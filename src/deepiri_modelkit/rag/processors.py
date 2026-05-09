"""
Document Processors for Universal RAG
Handles preprocessing, chunking, and metadata extraction for different document types
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import re
from datetime import datetime

from .base import Document, DocumentType, IndustryNiche


class DocumentProcessor(ABC):
    """Base class for document processing"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    @abstractmethod
    def process(self, raw_content: str, metadata: Dict[str, Any]) -> List[Document]:
        """
        Process raw content into structured documents
        
        Args:
            raw_content: Raw text content
            metadata: Document metadata
            
        Returns:
            List of processed document chunks
        """
        pass
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk text into smaller pieces with overlap
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + self.chunk_size
            
            # Find the last sentence boundary within chunk_size
            if end < text_length:
                # Look for sentence endings
                for sep in ['. ', '.\n', '! ', '!\n', '? ', '?\n']:
                    last_sep = text.rfind(sep, start, end)
                    if last_sep != -1:
                        end = last_sep + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start forward with overlap
            start = end - self.chunk_overlap if end - self.chunk_overlap > start else end
        
        return chunks
    
    def extract_metadata(self, content: str) -> Dict[str, Any]:
        """Extract metadata from content (can be overridden)"""
        return {}


class RegulationProcessor(DocumentProcessor):
    """
    Processor for regulations, policies, and compliance documents
    Common across insurance, healthcare, manufacturing, etc.
    """
    
    def process(self, raw_content: str, metadata: Dict[str, Any]) -> List[Document]:
        """Process regulation documents"""
        # Extract sections and subsections
        sections = self._extract_sections(raw_content)
        
        documents = []
        base_id = metadata.get('id', 'reg_' + str(hash(raw_content[:100])))
        industry = IndustryNiche(metadata.get('industry', 'generic'))
        
        for idx, section in enumerate(sections):
            doc = Document(
                id=f"{base_id}_chunk_{idx}",
                content=section['content'],
                doc_type=DocumentType.REGULATION,
                industry=industry,
                title=metadata.get('title', 'Regulation Document'),
                source=metadata.get('source'),
                created_at=self._parse_date(metadata.get('created_at')),
                metadata={
                    **metadata,
                    'section': section.get('section'),
                    'subsection': section.get('subsection'),
                },
                chunk_index=idx,
                total_chunks=len(sections),
            )
            documents.append(doc)
        
        return documents
    
    def _extract_sections(self, content: str) -> List[Dict[str, Any]]:
        """Extract sections from regulation text"""
        # Match section headers like "Section 1.2.3", "Article 5", etc.
        section_pattern = r'(Section|Article|Part|Chapter)\s+(\d+(?:\.\d+)*)'
        
        sections = []
        current_section = {'section': None, 'content': ''}
        
        lines = content.split('\n')
        for line in lines:
            match = re.search(section_pattern, line, re.IGNORECASE)
            if match:
                # Save previous section
                if current_section['content']:
                    sections.append(current_section)
                # Start new section
                current_section = {
                    'section': match.group(0),
                    'content': line + '\n'
                }
            else:
                current_section['content'] += line + '\n'
        
        # Add last section
        if current_section['content']:
            sections.append(current_section)
        
        # If no sections found, treat entire content as one chunk
        if not sections:
            chunks = self.chunk_text(content)
            sections = [{'section': f'Chunk {i+1}', 'content': chunk} 
                       for i, chunk in enumerate(chunks)]
        
        return sections
    
    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse date string to datetime"""
        if not date_str:
            return None
        try:
            return datetime.fromisoformat(date_str)
        except (ValueError, AttributeError):
            return None


class HistoricalDataProcessor(DocumentProcessor):
    """
    Processor for historical operational data
    - Work orders, maintenance logs, claim records, service history
    """
    
    def process(self, raw_content: str, metadata: Dict[str, Any]) -> List[Document]:
        """Process historical data records"""
        doc_type_str = metadata.get('doc_type', 'work_order')
        doc_type = DocumentType(doc_type_str) if doc_type_str else DocumentType.WORK_ORDER
        
        # Historical data is typically already structured
        # We may need minimal chunking
        chunks = self.chunk_text(raw_content) if len(raw_content) > self.chunk_size else [raw_content]
        
        documents = []
        base_id = metadata.get('id', 'hist_' + str(hash(raw_content[:100])))
        industry = IndustryNiche(metadata.get('industry', 'generic'))
        
        for idx, chunk in enumerate(chunks):
            doc = Document(
                id=f"{base_id}_chunk_{idx}",
                content=chunk,
                doc_type=doc_type,
                industry=industry,
                title=metadata.get('title', f'{doc_type.value.replace("_", " ").title()}'),
                source=metadata.get('source'),
                created_at=self._parse_date(metadata.get('created_at')),
                updated_at=self._parse_date(metadata.get('updated_at')),
                metadata={
                    **metadata,
                    'record_type': doc_type.value,
                    'record_id': metadata.get('record_id'),
                    'status': metadata.get('status'),
                    'priority': metadata.get('priority'),
                    'assigned_to': metadata.get('assigned_to'),
                },
                chunk_index=idx,
                total_chunks=len(chunks),
            )
            documents.append(doc)
        
        return documents
    
    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse date string to datetime"""
        if not date_str:
            return None
        if isinstance(date_str, datetime):
            return date_str
        try:
            return datetime.fromisoformat(date_str)
        except (ValueError, AttributeError):
            try:
                return datetime.strptime(date_str, '%Y-%m-%d')
            except (ValueError, AttributeError):
                return None


class KnowledgeBaseProcessor(DocumentProcessor):
    """
    Processor for knowledge base articles, FAQs, and guides
    - Equipment repair guides, compliance advice, troubleshooting steps
    """
    
    def process(self, raw_content: str, metadata: Dict[str, Any]) -> List[Document]:
        """Process knowledge base articles"""
        doc_type = DocumentType.KNOWLEDGE_BASE if metadata.get('doc_type') == 'knowledge_base' else DocumentType.FAQ
        
        # Extract Q&A pairs if FAQ format
        if doc_type == DocumentType.FAQ:
            qa_pairs = self._extract_qa_pairs(raw_content)
            if qa_pairs:
                return self._process_qa_pairs(qa_pairs, metadata)
        
        # Otherwise, process as regular article
        chunks = self.chunk_text(raw_content)
        
        documents = []
        base_id = metadata.get('id', 'kb_' + str(hash(raw_content[:100])))
        industry = IndustryNiche(metadata.get('industry', 'generic'))
        
        for idx, chunk in enumerate(chunks):
            doc = Document(
                id=f"{base_id}_chunk_{idx}",
                content=chunk,
                doc_type=doc_type,
                industry=industry,
                title=metadata.get('title', 'Knowledge Base Article'),
                source=metadata.get('source'),
                created_at=self._parse_date(metadata.get('created_at')),
                updated_at=self._parse_date(metadata.get('updated_at')),
                author=metadata.get('author'),
                metadata={
                    **metadata,
                    'category': metadata.get('category'),
                    'tags': metadata.get('tags', []),
                    'difficulty_level': metadata.get('difficulty_level'),
                },
                chunk_index=idx,
                total_chunks=len(chunks),
            )
            documents.append(doc)
        
        return documents
    
    def _extract_qa_pairs(self, content: str) -> List[Dict[str, str]]:
        """Extract Q&A pairs from FAQ content"""
        qa_pairs = []
        
        # Try different FAQ formats
        # Format 1: Q: ... A: ...
        pattern1 = r'Q:\s*(.+?)\s*A:\s*(.+?)(?=Q:|$)'
        matches = re.findall(pattern1, content, re.DOTALL | re.IGNORECASE)
        if matches:
            qa_pairs.extend([{'question': q.strip(), 'answer': a.strip()} for q, a in matches])
        
        # Format 2: Question/Answer headers
        pattern2 = r'Question:\s*(.+?)\s*Answer:\s*(.+?)(?=Question:|$)'
        matches = re.findall(pattern2, content, re.DOTALL | re.IGNORECASE)
        if matches:
            qa_pairs.extend([{'question': q.strip(), 'answer': a.strip()} for q, a in matches])
        
        return qa_pairs
    
    def _process_qa_pairs(self, qa_pairs: List[Dict[str, str]], metadata: Dict[str, Any]) -> List[Document]:
        """Process Q&A pairs into documents"""
        documents = []
        base_id = metadata.get('id', 'faq_' + str(hash(str(qa_pairs[0]))))
        industry = IndustryNiche(metadata.get('industry', 'generic'))
        
        for idx, qa in enumerate(qa_pairs):
            content = f"Question: {qa['question']}\n\nAnswer: {qa['answer']}"
            doc = Document(
                id=f"{base_id}_qa_{idx}",
                content=content,
                doc_type=DocumentType.FAQ,
                industry=industry,
                title=qa['question'][:100],  # Use question as title
                source=metadata.get('source'),
                metadata={
                    **metadata,
                    'question': qa['question'],
                    'answer': qa['answer'],
                },
                chunk_index=idx,
                total_chunks=len(qa_pairs),
            )
            documents.append(doc)
        
        return documents
    
    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse date string to datetime"""
        if not date_str:
            return None
        if isinstance(date_str, datetime):
            return date_str
        try:
            return datetime.fromisoformat(date_str)
        except (ValueError, AttributeError):
            return None


class ManualProcessor(DocumentProcessor):
    """
    Processor for equipment manuals, operation guides, technical specifications
    """
    
    def process(self, raw_content: str, metadata: Dict[str, Any]) -> List[Document]:
        """Process manual documents"""
        # Extract chapters and sections
        sections = self._extract_sections(raw_content)
        
        documents = []
        base_id = metadata.get('id', 'manual_' + str(hash(raw_content[:100])))
        industry = IndustryNiche(metadata.get('industry', 'generic'))
        
        for idx, section in enumerate(sections):
            doc = Document(
                id=f"{base_id}_chunk_{idx}",
                content=section['content'],
                doc_type=DocumentType.MANUAL,
                industry=industry,
                title=metadata.get('title', 'Equipment Manual'),
                source=metadata.get('source'),
                version=metadata.get('version'),
                metadata={
                    **metadata,
                    'chapter': section.get('chapter'),
                    'section': section.get('section'),
                    'equipment_model': metadata.get('equipment_model'),
                    'manufacturer': metadata.get('manufacturer'),
                },
                chunk_index=idx,
                total_chunks=len(sections),
            )
            documents.append(doc)
        
        return documents
    
    def _extract_sections(self, content: str) -> List[Dict[str, Any]]:
        """Extract sections from manual text"""
        # Match chapter/section headers
        section_pattern = r'(Chapter|Section)\s+(\d+(?:\.\d+)*):?\s*(.+?)(?=\n)'
        
        sections = []
        current_section = {'chapter': None, 'section': None, 'content': ''}
        
        lines = content.split('\n')
        for line in lines:
            match = re.search(section_pattern, line, re.IGNORECASE)
            if match:
                # Save previous section
                if current_section['content']:
                    sections.append(current_section)
                # Start new section
                section_type = match.group(1).lower()
                section_num = match.group(2)
                section_title = match.group(3).strip() if match.group(3) else ''
                current_section = {
                    section_type: f"{section_type.title()} {section_num}",
                    'section_title': section_title,
                    'content': line + '\n'
                }
            else:
                current_section['content'] += line + '\n'
        
        # Add last section
        if current_section['content']:
            sections.append(current_section)
        
        # If no sections found, chunk the content
        if not sections:
            chunks = self.chunk_text(content)
            sections = [{'section': f'Chunk {i+1}', 'content': chunk} 
                       for i, chunk in enumerate(chunks)]
        
        return sections


def get_processor(doc_type: DocumentType, **kwargs) -> DocumentProcessor:
    """
    Factory function to get appropriate processor for document type
    
    Args:
        doc_type: Type of document
        **kwargs: Additional configuration for processor
        
    Returns:
        Configured document processor
    """
    processor_map = {
        DocumentType.REGULATION: RegulationProcessor,
        DocumentType.POLICY: RegulationProcessor,  # Similar processing
        DocumentType.WORK_ORDER: HistoricalDataProcessor,
        DocumentType.CLAIM_RECORD: HistoricalDataProcessor,
        DocumentType.MAINTENANCE_LOG: HistoricalDataProcessor,
        DocumentType.FAQ: KnowledgeBaseProcessor,
        DocumentType.KNOWLEDGE_BASE: KnowledgeBaseProcessor,
        DocumentType.MANUAL: ManualProcessor,
        DocumentType.TECHNICAL_SPEC: ManualProcessor,  # Similar processing
        DocumentType.PROCEDURE: ManualProcessor,  # Similar processing
    }
    
    processor_class = processor_map.get(doc_type, DocumentProcessor)
    return processor_class(**kwargs)

