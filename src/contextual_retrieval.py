"""
Contextual Retrieval Module for RAG Pipeline

This module implements contextual retrieval using both contextual embeddings
and contextual BM25 to improve retrieval accuracy by 49-67%.
"""

import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from rank_bm25 import BM25Okapi
from src.document_loader import Document, SimpleDocumentLoader
from src.simple_embedding import SimpleTextEmbedder
from src.simple_generation import SimpleResponseGenerator
from src.vector_store import ChromaVectorStore
from config.config import CHUNK_SIZE, CHUNK_OVERLAP

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContextualRetrieval:
    """
    Advanced retrieval system using contextual embeddings and BM25 for improved accuracy.
    
    Based on Anthropic's research showing 49% improvement in retrieval accuracy
    when combining contextual embeddings with contextual BM25.
    """

    def __init__(self, vector_store_path: str = "./data/vector_store"):
        """
        Initialize the contextual retrieval system.

        Args:
            vector_store_path: Path to the vector store
        """
        self.vector_store_path = vector_store_path
        self.embedder = SimpleTextEmbedder()
        self.vector_store = ChromaVectorStore(vector_store_path)
        self.contextualizer = SimpleResponseGenerator()
        
        # BM25 components
        self.bm25 = None
        self.bm25_corpus = []
        self.bm25_documents = []
        
        # Store contextualized content
        self.contextualized_chunks = []
        self.original_chunks = []

    def generate_chunk_context(self, chunk: str, full_document: str) -> str:
        """
        Generate contextual information for a chunk using Claude.

        Args:
            chunk: The text chunk to contextualize
            full_document: The complete document containing the chunk

        Returns:
            Contextual information to prepend to the chunk
        """
        prompt = f"""<document>
{full_document}
</document>

Here is the chunk we want to situate within the whole document
<chunk>
{chunk}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

        try:
            result = self.contextualizer.generate_response(
                query="Generate context",
                context=prompt,
                max_tokens=100
            )
            context = result["response"].strip()
            logger.debug(f"Generated context: {context[:100]}...")
            return context
        except Exception as e:
            logger.error(f"Error generating context: {e}")
            return ""

    def process_documents_with_context(self, documents: List[Document]) -> List[Document]:
        """
        Process documents to add contextual information to each chunk.

        Args:
            documents: List of Document objects

        Returns:
            List of contextualized Document objects
        """
        logger.info(f"Processing {len(documents)} documents with contextual information...")
        
        contextualized_docs = []
        
        # Group chunks by source document
        doc_groups = {}
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            if source not in doc_groups:
                doc_groups[source] = {"chunks": [], "full_content": ""}
            doc_groups[source]["chunks"].append(doc)
        
        # Reconstruct full documents for context generation
        for source, group in doc_groups.items():
            # Sort chunks by chunk_index if available
            chunks = sorted(group["chunks"], 
                          key=lambda x: x.metadata.get("chunk_index", 0))
            
            # Reconstruct full document
            full_document = "\n\n".join([chunk.page_content for chunk in chunks])
            
            logger.info(f"Generating context for {len(chunks)} chunks from {source}")
            
            # Generate context for each chunk
            for chunk in chunks:
                try:
                    context = self.generate_chunk_context(chunk.page_content, full_document)
                    
                    # Create contextualized content
                    if context:
                        contextualized_content = f"{context}\n\n{chunk.page_content}"
                    else:
                        contextualized_content = chunk.page_content
                    
                    # Create new document with contextualized content
                    contextualized_doc = Document(
                        page_content=contextualized_content,
                        metadata={
                            **chunk.metadata,
                            "original_content": chunk.page_content,
                            "context": context,
                            "contextualized": True
                        }
                    )
                    
                    contextualized_docs.append(contextualized_doc)
                    
                except Exception as e:
                    logger.error(f"Error contextualizing chunk: {e}")
                    # Fallback to original chunk
                    contextualized_docs.append(chunk)
        
        logger.info(f"Successfully contextualized {len(contextualized_docs)} documents")
        return contextualized_docs

    def build_bm25_index(self, documents: List[Document]):
        """
        Build BM25 index from contextualized documents.

        Args:
            documents: List of contextualized Document objects
        """
        logger.info("Building BM25 index...")
        
        # Prepare corpus for BM25
        self.bm25_corpus = []
        self.bm25_documents = documents
        
        for doc in documents:
            # Tokenize the contextualized content
            tokens = doc.page_content.lower().split()
            self.bm25_corpus.append(tokens)
        
        # Build BM25 index
        self.bm25 = BM25Okapi(self.bm25_corpus)
        logger.info(f"BM25 index built with {len(self.bm25_corpus)} documents")

    def build_contextual_knowledge_base(self, data_path: str = "./data", rebuild: bool = False):
        """
        Build knowledge base with contextual retrieval capabilities.

        Args:
            data_path: Path to documents directory
            rebuild: Whether to rebuild even if index exists
        """
        logger.info("Building contextual knowledge base...")
        
        # Load and process documents
        loader = SimpleDocumentLoader(data_path)
        documents = loader.process_documents()
        
        if not documents:
            logger.warning("No documents found to process")
            return False
        
        # Add contextual information
        contextualized_docs = self.process_documents_with_context(documents)
        
        # Build vector store with contextualized embeddings
        embedder_result = self.embedder.encode_documents(contextualized_docs)
        
        self.vector_store.build_index(
            embeddings=embedder_result["embeddings"],
            texts=embedder_result["texts"],
            metadata=embedder_result["metadata"],
            documents=embedder_result["documents"]
        )
        
        # Build BM25 index
        self.build_bm25_index(contextualized_docs)
        
        logger.info("Contextual knowledge base built successfully")
        return True

    def hybrid_search(self, query: str, k: int = 20, alpha: float = 0.5) -> List[Tuple[Document, float]]:
        """
        Perform hybrid search combining contextual embeddings and BM25.

        Args:
            query: Search query
            k: Number of results to return
            alpha: Weight for semantic search (1-alpha for BM25)

        Returns:
            List of (Document, score) tuples
        """
        if not self.bm25 or not self.bm25_documents:
            logger.warning("BM25 index not built, falling back to semantic search only")
            return self.vector_store.search(self.embedder.encode_query(query), k)
        
        # Semantic search with contextual embeddings
        query_embedding = self.embedder.encode_query(query)
        semantic_results = self.vector_store.search(query_embedding, k=k*2)  # Get more for fusion
        
        # BM25 search with contextual content
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # Get top BM25 results
        bm25_indices = sorted(range(len(bm25_scores)), 
                             key=lambda i: bm25_scores[i], reverse=True)[:k*2]
        
        # Combine results using rank fusion
        combined_results = {}
        
        # Add semantic results
        for i, (doc, sem_score) in enumerate(semantic_results):
            doc_id = id(doc)
            rank_score = 1.0 / (i + 1)  # Reciprocal rank
            combined_results[doc_id] = {
                "document": doc,
                "semantic_score": sem_score,
                "semantic_rank": rank_score,
                "bm25_score": 0.0,
                "bm25_rank": 0.0
            }
        
        # Add BM25 results
        for i, doc_idx in enumerate(bm25_indices):
            if doc_idx < len(self.bm25_documents):
                doc = self.bm25_documents[doc_idx]
                doc_id = id(doc)
                rank_score = 1.0 / (i + 1)
                bm25_score = bm25_scores[doc_idx]
                
                if doc_id in combined_results:
                    combined_results[doc_id]["bm25_score"] = bm25_score
                    combined_results[doc_id]["bm25_rank"] = rank_score
                else:
                    combined_results[doc_id] = {
                        "document": doc,
                        "semantic_score": 0.0,
                        "semantic_rank": 0.0,
                        "bm25_score": bm25_score,
                        "bm25_rank": rank_score
                    }
        
        # Calculate final scores using weighted combination
        final_results = []
        for doc_id, result in combined_results.items():
            # Combine ranks (reciprocal rank fusion)
            final_score = (alpha * result["semantic_rank"] + 
                          (1 - alpha) * result["bm25_rank"])
            
            final_results.append((result["document"], final_score))
        
        # Sort by final score and return top k
        final_results.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Hybrid search returned {len(final_results[:k])} results")
        return final_results[:k]

    def retrieve_with_context(self, query: str, k: int = 20, 
                             include_metadata: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieve documents with contextual retrieval.
        
        Args:
            query: Search query
            k: Number of results to return
            include_metadata: Whether to include metadata
            
        Returns:
            List of document dictionaries
        """
        search_results = self.search_with_context(query, k, include_metadata)
        return search_results["results"]

    def search_with_context(self, query: str, k: int = 20, 
                           include_metadata: bool = True) -> Dict[str, Any]:
        """
        Search with contextual retrieval and return formatted results.

        Args:
            query: Search query
            k: Number of results to return
            include_metadata: Whether to include metadata

        Returns:
            Dictionary with search results and context
        """
        results = self.hybrid_search(query, k=k)
        
        formatted_results = []
        context_texts = []
        
        for doc, score in results:
            # Use original content for context if available
            original_content = doc.metadata.get("original_content", doc.page_content)
            context_info = doc.metadata.get("context", "")
            
            result_dict = {
                "content": original_content,
                "similarity_score": score,
                "context_info": context_info
            }
            
            if include_metadata:
                result_dict["metadata"] = doc.metadata
            
            formatted_results.append(result_dict)
            context_texts.append(original_content)
        
        return {
            "query": query,
            "results": formatted_results,
            "context": "\n\n".join(context_texts),
            "total_results": len(results),
            "search_type": "contextual_hybrid"
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the contextual retrieval system."""
        vector_stats = self.vector_store.get_stats()
        
        stats = {
            "contextual_retrieval": {
                "vector_store": vector_stats,
                "bm25_enabled": self.bm25 is not None,
                "bm25_documents": len(self.bm25_documents) if self.bm25_documents else 0,
                "embedding_model": self.embedder.model_name,
                "contextualizer_model": self.contextualizer.model
            }
        }
        
        return stats


def main():
    """Example usage of contextual retrieval."""
    # Initialize contextual retrieval
    contextual_retriever = ContextualRetrieval()
    
    # Build contextual knowledge base
    success = contextual_retriever.build_contextual_knowledge_base()
    
    if success:
        print("Contextual knowledge base built successfully!")
        
        # Test search
        query = "What is artificial intelligence?"
        results = contextual_retriever.search_with_context(query, k=5)
        
        print(f"\nQuery: {results['query']}")
        print(f"Search type: {results['search_type']}")
        print(f"Total results: {results['total_results']}")
        
        for i, result in enumerate(results['results'][:3], 1):
            print(f"\nResult {i} (Score: {result['similarity_score']:.4f}):")
            print(f"Context: {result['context_info'][:100]}...")
            print(f"Content: {result['content'][:200]}...")
        
        # Show stats
        stats = contextual_retriever.get_stats()
        print(f"\nStats: {stats}")
    
    else:
        print("Failed to build contextual knowledge base")


if __name__ == "__main__":
    main()
