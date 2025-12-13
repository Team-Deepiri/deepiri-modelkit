"""
Redis Streams client for event-driven architecture
"""
import redis.asyncio as redis
from typing import Dict, Any, Optional, AsyncIterator, Callable
import json
import asyncio
from datetime import datetime

from .topics import StreamTopics
from ..contracts.events import BaseEvent


class StreamingClient:
    """
    Redis Streams client for publishing and subscribing to events
    """
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        redis_host: str = "redis",
        redis_port: int = 6379,
        redis_password: Optional[str] = None
    ):
        """
        Initialize streaming client
        
        Args:
            redis_url: Full Redis URL (redis://password@host:port)
            redis_host: Redis host (if not using redis_url)
            redis_port: Redis port (if not using redis_url)
            redis_password: Redis password (if not using redis_url)
        """
        if redis_url:
            self.redis = redis.from_url(redis_url, decode_responses=True)
        else:
            self.redis = redis.Redis(
                host=redis_host,
                port=redis_port,
                password=redis_password,
                decode_responses=True
            )
        self._running = False
        self._subscriptions = {}
    
    async def connect(self):
        """Connect to Redis"""
        await self.redis.ping()
    
    async def disconnect(self):
        """Disconnect from Redis"""
        await self.redis.close()
    
    async def publish(
        self,
        topic: str,
        event: Dict[str, Any],
        max_length: Optional[int] = 10000
    ) -> str:
        """
        Publish event to stream
        
        Args:
            topic: Stream topic name
            event: Event data (dict)
            max_length: Max stream length (truncate old messages)
        
        Returns:
            Message ID
        """
        # Ensure event has timestamp
        if "timestamp" not in event:
            event["timestamp"] = datetime.utcnow().isoformat()
        
        # Publish to stream
        message_id = await self.redis.xadd(
            topic,
            event,
            maxlen=max_length,
            approximate=True
        )
        
        return message_id
    
    async def subscribe(
        self,
        topic: str,
        callback: Callable[[Dict[str, Any]], None],
        consumer_group: Optional[str] = None,
        consumer_name: Optional[str] = None,
        last_id: str = "0",
        block_ms: int = 1000
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Subscribe to stream and yield events
        
        Args:
            topic: Stream topic name
            callback: Optional callback function
            consumer_group: Consumer group name (for load balancing)
            consumer_name: Consumer name (unique per consumer)
            last_id: Last message ID to read from
            block_ms: Block time in milliseconds
        
        Yields:
            Event data (dict)
        """
        # Create consumer group if specified
        if consumer_group:
            try:
                await self.redis.xgroup_create(
                    topic,
                    consumer_group,
                    id="0",
                    mkstream=True
                )
            except redis.ResponseError as e:
                if "BUSYGROUP" not in str(e):
                    raise
        
        self._running = True
        
        while self._running:
            try:
                if consumer_group and consumer_name:
                    # Read from consumer group
                    messages = await self.redis.xreadgroup(
                        consumer_group,
                        consumer_name,
                        {topic: ">"},
                        count=10,
                        block=block_ms
                    )
                else:
                    # Direct read
                    messages = await self.redis.xread(
                        {topic: last_id},
                        count=10,
                        block=block_ms
                    )
                
                for stream_name, stream_messages in messages:
                    for msg_id, data in stream_messages:
                        # Yield event
                        yield data
                        
                        # Call callback if provided
                        if callback:
                            try:
                                await callback(data) if asyncio.iscoroutinefunction(callback) else callback(data)
                            except Exception as e:
                                print(f"Callback error: {e}")
                        
                        # Update last_id for next read
                        last_id = msg_id
                        
                        # Acknowledge if using consumer group
                        if consumer_group and consumer_name:
                            await self.redis.xack(topic, consumer_group, msg_id)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Stream subscription error: {e}")
                await asyncio.sleep(1)
    
    async def subscribe_async(
        self,
        topic: str,
        callback: Callable[[Dict[str, Any]], None],
        consumer_group: Optional[str] = None,
        consumer_name: Optional[str] = None
    ):
        """
        Subscribe to stream in background task
        
        Args:
            topic: Stream topic name
            callback: Callback function
            consumer_group: Consumer group name
            consumer_name: Consumer name
        """
        async for event in self.subscribe(
            topic,
            callback,
            consumer_group,
            consumer_name
        ):
            pass  # Callback handles events
    
    def stop(self):
        """Stop all subscriptions"""
        self._running = False
    
    async def get_stream_info(self, topic: str) -> Dict[str, Any]:
        """Get stream information"""
        info = await self.redis.xinfo_stream(topic)
        return dict(info)
    
    async def get_stream_length(self, topic: str) -> int:
        """Get number of messages in stream"""
        return await self.redis.xlen(topic)

