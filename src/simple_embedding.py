"""
Simple Embedding Module for RAG Pipeline

This module handles converting text documents into vector embeddings
using Sentence Transformers models without LangChain dependencies.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from src.document_loader import Document
from config.config import EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleTextEmbedder:
    """Handles text embedding using Sentence Transformers."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        """
        Initialize the text embedder.

        Args:
            model_name: Name of the Sentence Transformers model to use
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the Sentence Transformers model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise

    def encode_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Encode a list of texts into embeddings.

        Args:
            texts: List of text strings to encode
            batch_size: Batch size for encoding (affects memory usage)

        Returns:
            List of embedding vectors
        """
        if not texts:
            logger.warning("No texts provided for encoding")
            return []

        if self.model is None:
            raise ValueError("Embedding model not loaded")

        try:
            logger.info(f"Encoding {len(texts)} texts with batch size {batch_size}")

            # Encode texts in batches to manage memory
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )

            # Convert to list for ChromaDB compatibility
            embeddings_list = embeddings.tolist()

            logger.info(f"Successfully encoded {len(texts)} texts. Embedding shape: {embeddings.shape}")
            return embeddings_list

        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            raise

    def encode_documents(self, documents: List[Document], batch_size: int = 32) -> Dict[str, Any]:
        """
        Encode a list of Document objects into embeddings.

        Args:
            documents: List of Document objects
            batch_size: Batch size for encoding

        Returns:
            Dictionary containing embeddings and metadata
        """
        if not documents:
            logger.warning("No documents provided for encoding")
            return {"embeddings": [], "texts": [], "metadata": []}

        # Extract texts and metadata from documents
        texts = [doc.page_content for doc in documents]
        metadata = [doc.metadata for doc in documents]

        # Encode texts
        embeddings = self.encode_texts(texts, batch_size=batch_size)

        result = {
            "embeddings": embeddings,
            "texts": texts,
            "metadata": metadata,
            "documents": documents
        }

        logger.info(f"Encoded {len(documents)} documents successfully")
        return result

    def encode_query(self, query: str) -> List[float]:
        """
        Encode a single query string for similarity search.

        Args:
            query: Query string to encode

        Returns:
            List representing the query embedding
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")

        try:
            embedding = self.model.encode([query], convert_to_numpy=True)
            return embedding[0].tolist()

        except Exception as e:
            logger.error(f"Error encoding query '{query}': {e}")
            raise

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings produced by this model.

        Returns:
            Embedding dimension
        """
        if self.model is None:
            return EMBEDDING_DIMENSION  # Fallback to config value

        return self.model.get_sentence_embedding_dimension()

    def __del__(self):
        """Cleanup method to free model resources."""
        if hasattr(self, 'model') and self.model is not None:
            # Clear model from memory
            del self.model


def main():
    """Example usage of the SimpleTextEmbedder module."""
    embedder = SimpleTextEmbedder()

    # Example documents
    sample_docs = [
        Document(page_content="This is a sample document about artificial intelligence."),
        Document(page_content="Machine learning is a subset of AI that focuses on algorithms.")
    ]

    # Encode documents
    result = embedder.encode_documents(sample_docs)

    print(f"Embeddings count: {len(result['embeddings'])}")
    print(f"Embedding dimension: {embedder.get_embedding_dimension()}")

    # Encode a query
    query_embedding = embedder.encode_query("What is AI?")
    print(f"Query embedding dimension: {len(query_embedding)}")


if __name__ == "__main__":
    main()
