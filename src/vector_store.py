"""
Vector Store Module for RAG Pipeline

This module handles storing and retrieving vector embeddings
using ChromaDB (a vector database for embeddings).
"""

import os
import json
import logging
import chromadb
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from langchain.schema import Document
from config.config import VECTOR_STORE_PATH, TOP_K_RETRIEVAL

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """ChromaDB-based vector store for efficient similarity search."""

    def __init__(self, store_path: str = VECTOR_STORE_PATH, collection_name: str = "rag_documents"):
        """
        Initialize the ChromaDB vector store.

        Args:
            store_path: Path to the vector store directory
            collection_name: Name of the ChromaDB collection
        """
        self.store_path = Path(store_path)
        self.collection_name = collection_name

        # Initialize ChromaDB client with persistent storage
        self.client = chromadb.PersistentClient(path=str(self.store_path))

        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(name=collection_name)
            logger.info(f"Created new collection: {collection_name}")

        self.texts = []
        self.metadata = []
        self.documents = []

    def build_index(self, embeddings: List[List[float]], texts: List[str],
                   metadata: List[Dict], documents: List[Document]):
        """
        Build ChromaDB index from embeddings and associated data.

        Args:
            embeddings: List of embedding vectors
            texts: List of text strings
            metadata: List of metadata dictionaries
            documents: List of Document objects
        """
        if not embeddings:
            raise ValueError("Cannot build index from empty embeddings")

        # Store data locally for retrieval
        self.texts = texts
        self.metadata = metadata
        self.documents = documents

        # Prepare data for ChromaDB
        ids = [f"doc_{i}" for i in range(len(embeddings))]

        # Add to ChromaDB collection
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadata,
            ids=ids
        )

        logger.info(f"Built ChromaDB index with {len(embeddings)} vectors")

    def search(self, query_embedding: List[float], k: int = TOP_K_RETRIEVAL) -> List[Tuple[Document, float]]:
        """
        Search for most similar documents to the query embedding.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return

        Returns:
            List of tuples (Document, similarity_score)
        """
        try:
            # Search ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )

            # Format results
            formatted_results = []
            for i, (doc_id, distance) in enumerate(zip(results['ids'][0], results['distances'][0])):
                # Convert distance to similarity score (ChromaDB returns cosine distance)
                similarity_score = 1.0 - distance

                # Get document index from id
                doc_index = int(doc_id.split('_')[1])

                if doc_index < len(self.documents):
                    formatted_results.append((self.documents[doc_index], similarity_score))

            logger.info(f"Found {len(formatted_results)} relevant documents")
            return formatted_results

        except Exception as e:
            logger.error(f"Error searching ChromaDB: {e}")
            return []

    def delete_index(self):
        """Delete the ChromaDB collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with store statistics
        """
        try:
            count = self.collection.count()
            stats = {
                "collection_name": self.collection_name,
                "total_documents": count,
                "store_path": str(self.store_path)
            }
            return stats
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}


def main():
    """Example usage of the ChromaVectorStore module."""
    import numpy as np
    from langchain.schema import Document

    # Create sample data
    embeddings = np.random.rand(10, 384).tolist()  # Convert to list for ChromaDB
    texts = [f"Sample text {i}" for i in range(10)]
    metadata = [{"id": i} for i in range(10)]
    documents = [Document(page_content=text, metadata=meta)
                for text, meta in zip(texts, metadata)]

    # Initialize vector store
    vector_store = ChromaVectorStore("./data/test_vector_store")

    # Build index
    vector_store.build_index(embeddings, texts, metadata, documents)

    # Search
    query_embedding = np.random.rand(384).tolist()
    results = vector_store.search(query_embedding, k=3)

    print(f"Found {len(results)} results")
    for doc, score in results:
        print(f"Score: {score:.4f}, Content: {doc.page_content[:50]}...")

    # Get stats
    print(f"Store stats: {vector_store.get_stats()}")


if __name__ == "__main__":
    main()
