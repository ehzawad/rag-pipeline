"""
Simple Document Loader Module

This module handles loading and preprocessing documents without LangChain dependencies.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any
import PyPDF2
from config.config import DATA_PATH, SUPPORTED_FILE_TYPES, CHUNK_SIZE, CHUNK_OVERLAP

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Document:
    """Simple document class to replace LangChain Document."""
    
    def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
        self.page_content = page_content
        self.metadata = metadata or {}


class SimpleDocumentLoader:
    """Simple document loader without LangChain dependencies."""

    def __init__(self, data_path: str = DATA_PATH):
        """
        Initialize the document loader.

        Args:
            data_path: Path to the directory containing documents
        """
        self.data_path = Path(data_path)

    def load_text_file(self, file_path: Path) -> List[Document]:
        """Load a text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            metadata = {
                "source": str(file_path),
                "file_type": file_path.suffix,
                "file_size": file_path.stat().st_size
            }
            
            return [Document(page_content=content, metadata=metadata)]
        
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {e}")
            return []

    def load_pdf_file(self, file_path: Path) -> List[Document]:
        """Load a PDF file."""
        try:
            documents = []
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    content = page.extract_text()
                    if content.strip():
                        metadata = {
                            "source": str(file_path),
                            "file_type": file_path.suffix,
                            "page_number": page_num + 1,
                            "total_pages": len(pdf_reader.pages)
                        }
                        documents.append(Document(page_content=content, metadata=metadata))
            
            return documents
        
        except Exception as e:
            logger.error(f"Error loading PDF file {file_path}: {e}")
            return []

    def load_documents(self) -> List[Document]:
        """
        Load all documents from the data directory.

        Returns:
            List of Document objects
        """
        documents = []

        if not self.data_path.exists():
            logger.warning(f"Data path {self.data_path} does not exist. Creating it...")
            self.data_path.mkdir(parents=True, exist_ok=True)
            return documents

        # Load documents based on file type
        for file_path in self.data_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_FILE_TYPES:
                try:
                    if file_path.suffix.lower() == ".txt" or file_path.suffix.lower() == ".md":
                        docs = self.load_text_file(file_path)
                    elif file_path.suffix.lower() == ".pdf":
                        docs = self.load_pdf_file(file_path)
                    else:
                        # Try as text file
                        docs = self.load_text_file(file_path)

                    documents.extend(docs)
                    logger.info(f"Loaded {len(docs)} documents from {file_path}")

                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
                    continue

        logger.info(f"Total documents loaded: {len(documents)}")
        return documents

    def split_text(self, text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
        """
        Split text into chunks with overlap.

        Args:
            text: Text to split
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks

        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            
            # If this isn't the last chunk, try to break at a sentence or word boundary
            if end < len(text):
                # Look for sentence boundary
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start + chunk_size // 2:
                    end = sentence_end + 1
                else:
                    # Look for word boundary
                    word_end = text.rfind(' ', start, end)
                    if word_end > start + chunk_size // 2:
                        end = word_end

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start position with overlap
            start = max(start + 1, end - chunk_overlap)

        return chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks.

        Args:
            documents: List of Document objects to split

        Returns:
            List of split Document objects
        """
        if not documents:
            logger.warning("No documents provided for splitting")
            return []

        split_docs = []
        
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            
            for i, chunk in enumerate(chunks):
                # Create new metadata for each chunk
                chunk_metadata = doc.metadata.copy()
                chunk_metadata.update({
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                })
                
                split_docs.append(Document(page_content=chunk, metadata=chunk_metadata))

        logger.info(f"Split {len(documents)} documents into {len(split_docs)} chunks")
        return split_docs

    def process_documents(self) -> List[Document]:
        """
        Complete pipeline: load documents and split them into chunks.

        Returns:
            List of processed Document objects ready for embedding
        """
        logger.info("Starting document processing pipeline...")

        # Load documents
        documents = self.load_documents()

        if not documents:
            logger.warning("No documents found to process")
            return []

        # Split documents
        split_documents = self.split_documents(documents)

        logger.info(f"Document processing complete. Total chunks: {len(split_documents)}")
        return split_documents


def main():
    """Example usage of the SimpleDocumentLoader module."""
    loader = SimpleDocumentLoader()
    documents = loader.process_documents()

    print(f"Processed {len(documents)} document chunks")
    if documents:
        print(f"First chunk preview: {documents[0].page_content[:200]}...")


if __name__ == "__main__":
    main()
