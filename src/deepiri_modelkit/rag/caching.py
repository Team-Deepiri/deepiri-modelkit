"""
Advanced Caching Layer for Universal RAG
Redis-based caching with intelligent invalidation and TTL management
"""

from typing import Optional, Any, List, Dict
import json
import hashlib
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

from .base import Document, RetrievalResult, RAGQuery


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime]
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.last_accessed is None:
            self.last_accessed = self.created_at
    
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "key": self.key,
            "value": self.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CacheEntry':
        """Create from dictionary"""
        return cls(
            key=data["key"],
            value=data["value"],
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            access_count=data.get("access_count", 0),
            last_accessed=datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else None,
            tags=data.get("tags", [])
        )


class AdvancedCacheManager:
    """
    Advanced cache manager with:
    - TTL management
    - Tag-based invalidation
    - Access tracking
    - Size limits
    - LRU eviction
    """
    
    def __init__(
        self,
        redis_client=None,
        default_ttl: int = 3600,
        max_size: int = 10000,
        enable_lru: bool = True
    ):
        self.redis_client = redis_client
        self.default_ttl = default_ttl
        self.max_size = max_size
        self.enable_lru = enable_lru
        
        # In-memory fallback if Redis unavailable
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.tag_index: Dict[str, List[str]] = {}  # tag -> [keys]
    
    def _get_key_prefix(self, namespace: str = "rag") -> str:
        """Get key prefix"""
        return f"{namespace}:"
    
    def _serialize_value(self, value: Any) -> str:
        """Serialize value for storage"""
        if isinstance(value, (list, dict)):
            return json.dumps(value)
        return str(value)
    
    def _deserialize_value(self, value: str, value_type: type = None) -> Any:
        """Deserialize value from storage"""
        try:
            if value_type == list or (isinstance(value, str) and value.startswith('[')):
                return json.loads(value)
            elif value_type == dict or (isinstance(value, str) and value.startswith('{')):
                return json.loads(value)
            return value
        except (json.JSONDecodeError, TypeError):
            return value
    
    def get(
        self,
        key: str,
        namespace: str = "rag",
        update_access: bool = True
    ) -> Optional[Any]:
        """Get value from cache"""
        full_key = f"{self._get_key_prefix(namespace)}{key}"
        
        # Try Redis first
        if self.redis_client:
            try:
                cached = self.redis_client.get(full_key)
                if cached:
                    entry_data = json.loads(cached)
                    entry = CacheEntry.from_dict(entry_data)
                    
                    if entry.is_expired():
                        self.delete(key, namespace)
                        return None
                    
                    if update_access:
                        entry.access_count += 1
                        entry.last_accessed = datetime.now()
                        self._update_redis_entry(full_key, entry)
                    
                    return entry.value
            except Exception as e:
                # Fallback to memory
                pass
        
        # Fallback to memory cache
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            
            if entry.is_expired():
                del self.memory_cache[key]
                return None
            
            if update_access:
                entry.access_count += 1
                entry.last_accessed = datetime.now()
            
            return entry.value
        
        return None
    
    def set(
        self,
        key: str,
        value: Any,
        namespace: str = "rag",
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Set value in cache"""
        full_key = f"{self._get_key_prefix(namespace)}{key}"
        ttl = ttl or self.default_ttl
        tags = tags or []
        
        expires_at = datetime.now() + timedelta(seconds=ttl)
        entry = CacheEntry(
            key=full_key,
            value=value,
            created_at=datetime.now(),
            expires_at=expires_at,
            tags=tags
        )
        
        # Try Redis first
        if self.redis_client:
            try:
                entry_data = json.dumps(entry.to_dict())
                self.redis_client.setex(full_key, ttl, entry_data)
                
                # Update tag index
                for tag in tags:
                    tag_key = f"{self._get_key_prefix(namespace)}tag:{tag}"
                    if tag_key not in self.tag_index:
                        self.tag_index[tag_key] = []
                    if full_key not in self.tag_index[tag_key]:
                        self.tag_index[tag_key].append(full_key)
                        self.redis_client.sadd(tag_key, full_key)
                
                return True
            except Exception as e:
                # Fallback to memory
                pass
        
        # Fallback to memory cache
        # Check size limit
        if len(self.memory_cache) >= self.max_size:
            if self.enable_lru:
                self._evict_lru()
            else:
                # Remove oldest
                oldest_key = min(
                    self.memory_cache.keys(),
                    key=lambda k: self.memory_cache[k].created_at
                )
                del self.memory_cache[oldest_key]
        
        self.memory_cache[key] = entry
        
        # Update tag index
        for tag in tags:
            if tag not in self.tag_index:
                self.tag_index[tag] = []
            if key not in self.tag_index[tag]:
                self.tag_index[tag].append(key)
        
        return True
    
    def delete(self, key: str, namespace: str = "rag") -> bool:
        """Delete key from cache"""
        full_key = f"{self._get_key_prefix(namespace)}{key}"
        
        # Try Redis
        if self.redis_client:
            try:
                self.redis_client.delete(full_key)
                return True
            except Exception:
                pass
        
        # Memory cache
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            # Remove from tag indexes
            for tag in entry.tags:
                if tag in self.tag_index and key in self.tag_index[tag]:
                    self.tag_index[tag].remove(key)
            del self.memory_cache[key]
            return True
        
        return False
    
    def invalidate_by_tag(self, tag: str, namespace: str = "rag") -> int:
        """Invalidate all keys with given tag"""
        tag_key = f"{self._get_key_prefix(namespace)}tag:{tag}"
        count = 0
        
        # Get keys from tag index
        keys_to_delete = []
        
        if self.redis_client:
            try:
                keys_to_delete = list(self.redis_client.smembers(tag_key))
                self.redis_client.delete(tag_key)
            except Exception:
                pass
        
        if tag in self.tag_index:
            keys_to_delete.extend(self.tag_index[tag])
            del self.tag_index[tag]
        
        # Delete all keys
        for key in keys_to_delete:
            if self.delete(key.replace(self._get_key_prefix(namespace), ""), namespace):
                count += 1
        
        return count
    
    def invalidate_by_pattern(self, pattern: str, namespace: str = "rag") -> int:
        """Invalidate keys matching pattern"""
        full_pattern = f"{self._get_key_prefix(namespace)}{pattern}"
        count = 0
        
        if self.redis_client:
            try:
                keys = self.redis_client.keys(full_pattern)
                if keys:
                    count = self.redis_client.delete(*keys)
            except Exception:
                pass
        
        # Memory cache
        keys_to_delete = [
            k for k in self.memory_cache.keys()
            if self._match_pattern(k, pattern)
        ]
        for key in keys_to_delete:
            self.delete(key, namespace)
            count += 1
        
        return count
    
    def _match_pattern(self, key: str, pattern: str) -> bool:
        """Simple pattern matching (supports * wildcard)"""
        import fnmatch
        return fnmatch.fnmatch(key, pattern)
    
    def _evict_lru(self):
        """Evict least recently used entry"""
        if not self.memory_cache:
            return
        
        lru_key = min(
            self.memory_cache.keys(),
            key=lambda k: (
                self.memory_cache[k].last_accessed or
                self.memory_cache[k].created_at
            )
        )
        del self.memory_cache[lru_key]
    
    def _update_redis_entry(self, key: str, entry: CacheEntry):
        """Update Redis entry with new access info"""
        if not self.redis_client:
            return
        
        try:
            entry_data = json.dumps(entry.to_dict())
            # Get remaining TTL
            ttl = self.redis_client.ttl(key)
            if ttl > 0:
                self.redis_client.setex(key, ttl, entry_data)
        except Exception:
            pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {
            "memory_cache_size": len(self.memory_cache),
            "max_size": self.max_size,
            "tag_index_size": len(self.tag_index),
            "redis_available": self.redis_client is not None,
        }
        
        if self.memory_cache:
            total_access = sum(e.access_count for e in self.memory_cache.values())
            stats["total_accesses"] = total_access
            stats["avg_access_per_entry"] = total_access / len(self.memory_cache)
        
        return stats
    
    def clear(self, namespace: str = "rag"):
        """Clear all cache entries in namespace"""
        pattern = f"{self._get_key_prefix(namespace)}*"
        return self.invalidate_by_pattern(pattern, namespace)


class EmbeddingCache:
    """Specialized cache for embeddings"""
    
    def __init__(self, cache_manager: AdvancedCacheManager):
        self.cache_manager = cache_manager
        self.namespace = "rag:embeddings"
    
    def get_embedding_key(self, text: str) -> str:
        """Generate cache key for embedding"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"emb:{text_hash}"
    
    def get(self, text: str) -> Optional[Any]:
        """Get cached embedding"""
        key = self.get_embedding_key(text)
        return self.cache_manager.get(key, namespace=self.namespace)
    
    def set(self, text: str, embedding: Any, ttl: int = 86400):
        """Cache embedding (24 hour default TTL)"""
        key = self.get_embedding_key(text)
        return self.cache_manager.set(
            key,
            embedding,
            namespace=self.namespace,
            ttl=ttl,
            tags=["embedding"]
        )


class QueryResultCache:
    """Specialized cache for query results"""
    
    def __init__(self, cache_manager: AdvancedCacheManager):
        self.cache_manager = cache_manager
        self.namespace = "rag:queries"
    
    def get_query_key(self, query: RAGQuery) -> str:
        """Generate cache key for query"""
        query_dict = query.to_dict()
        query_str = json.dumps(query_dict, sort_keys=True)
        query_hash = hashlib.md5(query_str.encode()).hexdigest()
        return f"query:{query_hash}"
    
    def get(self, query: RAGQuery) -> Optional[List[RetrievalResult]]:
        """Get cached query results"""
        key = self.get_query_key(query)
        cached = self.cache_manager.get(key, namespace=self.namespace)
        
        if cached:
            # Reconstruct RetrievalResult objects
            results = []
            for item in cached:
                doc = Document.from_dict(item["document"])
                result = RetrievalResult(
                    document=doc,
                    score=item["score"],
                    rerank_score=item.get("rerank_score")
                )
                results.append(result)
            return results
        
        return None
    
    def set(
        self,
        query: RAGQuery,
        results: List[RetrievalResult],
        ttl: int = 3600,
        tags: Optional[List[str]] = None
    ):
        """Cache query results"""
        key = self.get_query_key(query)
        
        # Serialize results
        serialized = [
            {
                "document": r.document.to_dict(),
                "score": r.score,
                "rerank_score": r.rerank_score
            }
            for r in results
        ]
        
        # Add query tags
        query_tags = tags or []
        if query.industry:
            query_tags.append(f"industry:{query.industry.value if hasattr(query.industry, 'value') else query.industry}")
        if query.doc_types:
            for dt in query.doc_types:
                query_tags.append(f"doctype:{dt.value if hasattr(dt, 'value') else dt}")
        
        return self.cache_manager.set(
            key,
            serialized,
            namespace=self.namespace,
            ttl=ttl,
            tags=query_tags
        )
    
    def invalidate_by_industry(self, industry: str):
        """Invalidate all queries for an industry"""
        return self.cache_manager.invalidate_by_tag(
            f"industry:{industry}",
            namespace=self.namespace
        )
    
    def invalidate_by_doc_type(self, doc_type: str):
        """Invalidate all queries for a document type"""
        return self.cache_manager.invalidate_by_tag(
            f"doctype:{doc_type}",
            namespace=self.namespace
        )

