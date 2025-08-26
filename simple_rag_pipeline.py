"""
Simple RAG Pipeline Main Script

This is the main entry point for the simplified RAG pipeline without LangChain dependencies.
It uses only OpenAI API directly and simple document processing.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent))

from src.document_loader import SimpleDocumentLoader
from src.simple_embedding import SimpleTextEmbedder
from src.vector_store import ChromaVectorStore
from src.contextual_retrieval import ContextualRetrieval
from src.simple_generation import SimpleResponseGenerator
from config.config import (
    VECTOR_STORE_PATH, DATA_PATH, TOP_K_RETRIEVAL,
    MODEL_NAME, TEMPERATURE, MAX_TOKENS
)

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleRAGPipeline:
    """Main RAG pipeline with contextual retrieval capabilities."""

    def __init__(self, vector_store_path: str = VECTOR_STORE_PATH):
        """
        Initialize the RAG pipeline with contextual retrieval.

        Args:
            vector_store_path: Path to the vector store
        """
        self.vector_store_path = vector_store_path

        # Initialize components
        self.contextual_retriever = ContextualRetrieval(vector_store_path)
        self.generator = SimpleResponseGenerator()

        logger.info("Simple RAG Pipeline with Contextual Retrieval initialized")

    def build_knowledge_base(self, data_path: str = DATA_PATH,
                           rebuild: bool = False) -> bool:
        """
        Build or rebuild the knowledge base with contextual retrieval.

        Args:
            data_path: Path to documents directory
            rebuild: Whether to rebuild even if index exists

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Building contextual knowledge base from {data_path}")

            # Check if index already exists
            if not rebuild:
                try:
                    stats = self.contextual_retriever.get_stats()
                    if stats.get("contextual_retrieval", {}).get("vector_store", {}).get("total_documents", 0) > 0:
                        logger.info("Knowledge base already exists, skipping build")
                        return True
                except:
                    pass  # Continue with build if stats check fails

            # Build contextual knowledge base
            success = self.contextual_retriever.build_contextual_knowledge_base(data_path, rebuild)

            if success:
                logger.info("Contextual knowledge base built successfully")
            else:
                logger.error("Failed to build contextual knowledge base")

            return success

        except Exception as e:
            logger.error(f"Error building knowledge base: {e}")
            return False

    def query(self, user_query: str, k: int = TOP_K_RETRIEVAL,
             temperature: float = TEMPERATURE, max_tokens: int = MAX_TOKENS,
             include_sources: bool = False) -> Dict[str, Any]:
        """
        Process a user query through the contextual RAG pipeline.

        Args:
            user_query: User's question or query
            k: Number of documents to retrieve
            temperature: Generation temperature (ignored for gpt-5-mini)
            max_tokens: Maximum tokens for generation
            include_sources: Whether to include source documents in response

        Returns:
            Dictionary containing response and metadata
        """
        if not user_query.strip():
            raise ValueError("Query cannot be empty")

        logger.info(f"Processing query with contextual retrieval: '{user_query}'")

        try:
            # Step 1: Retrieve relevant context using contextual retrieval
            retrieval_result = self.contextual_retriever.search_with_context(
                user_query, k=k, include_metadata=include_sources
            )

            context = retrieval_result["context"]
            context_docs = retrieval_result["results"] if include_sources else []

            # Step 2: Generate response
            if include_sources:
                response_result = self.generator.generate_with_sources(
                    user_query, context_docs, temperature, max_tokens
                )
            else:
                response_result = self.generator.generate_response(
                    user_query, context, temperature, max_tokens
                )

            # Combine results
            final_result = {
                "query": user_query,
                "response": response_result["response"],
                "context_used": response_result["context_used"],
                "retrieval_stats": {
                    "total_documents_found": retrieval_result["total_results"],
                    "documents_requested": k,
                    "search_type": retrieval_result["search_type"]
                },
                "generation_stats": {
                    "model": response_result["model"],
                    "temperature": response_result["temperature"],
                    "usage": response_result.get("usage", {})
                }
            }

            if include_sources:
                final_result["sources"] = response_result["sources"]

            logger.info("Query processed successfully with contextual retrieval")
            return final_result

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the pipeline.

        Returns:
            Dictionary with pipeline statistics
        """
        contextual_stats = self.contextual_retriever.get_stats()
        
        stats = {
            "pipeline_type": "Contextual RAG with Hybrid Search",
            "components": {
                "contextual_retriever": contextual_stats["contextual_retrieval"],
                "generator": f"Model: {self.generator.model}"
            },
            "knowledge_base": {
                "path": self.vector_store_path,
                "documents_count": contextual_stats["contextual_retrieval"]["vector_store"].get("total_documents", 0),
                "bm25_enabled": contextual_stats["contextual_retrieval"]["bm25_enabled"]
            }
        }

        return stats

    def rebuild_knowledge_base(self) -> bool:
        """
        Completely rebuild the contextual knowledge base.

        Returns:
            True if successful, False otherwise
        """
        logger.info("Rebuilding contextual knowledge base...")
        return self.build_knowledge_base(rebuild=True)


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Simple RAG Pipeline")
    parser.add_argument("--query", "-q", help="Query to process")
    parser.add_argument("--build", "-b", action="store_true", help="Build knowledge base")
    parser.add_argument("--rebuild", "-r", action="store_true", help="Rebuild knowledge base")
    parser.add_argument("--stats", "-s", action="store_true", help="Show pipeline statistics")
    parser.add_argument("--k", type=int, default=TOP_K_RETRIEVAL, help="Number of documents to retrieve")
    parser.add_argument("--sources", action="store_true", help="Include sources in response")

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = SimpleRAGPipeline()

    if args.build:
        success = pipeline.build_knowledge_base()
        print(f"Knowledge base build: {'Success' if success else 'Failed'}")

    elif args.rebuild:
        success = pipeline.rebuild_knowledge_base()
        print(f"Knowledge base rebuild: {'Success' if success else 'Failed'}")

    elif args.stats:
        stats = pipeline.get_stats()
        print("Pipeline Statistics:")
        for category, data in stats.items():
            print(f"\n{category.upper()}:")
            if isinstance(data, dict):
                for key, value in data.items():
                    print(f"  {key}: {value}")
            else:
                print(f"  {data}")

    elif args.query:
        try:
            result = pipeline.query(args.query, k=args.k, include_sources=args.sources)
            print(f"\nQuery: {result['query']}")
            print(f"\nResponse: {result['response']}")

            if args.sources and "sources" in result:
                print(f"\nSources ({len(result['sources'])}):")
                for source in result["sources"]:
                    print(f"  Source {source['source_id']} (Score: {source['similarity_score']:.4f})")
                    print(f"    Preview: {source['content_preview']}")

            print(f"\nRetrieval Stats: {result['retrieval_stats']}")
            print(f"Generation Stats: {result['generation_stats']}")

        except Exception as e:
            print(f"Error processing query: {e}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
