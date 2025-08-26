"""
Simple Generation Module for RAG Pipeline

This module handles generating responses using OpenAI GPT models
without LangChain dependencies.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI
from config.config import MODEL_NAME, TEMPERATURE, MAX_TOKENS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleResponseGenerator:
    """Handles response generation using OpenAI GPT models."""

    def __init__(self, api_key: Optional[str] = None, model: str = MODEL_NAME):
        """
        Initialize the response generator.

        Args:
            api_key: OpenAI API key (uses environment variable if None)
            model: OpenAI model to use for generation
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")

        self.model = model
        self.client = OpenAI(api_key=self.api_key)

        logger.info(f"Initialized SimpleResponseGenerator with model: {model}")

    def generate_response(self, query: str, context: str,
                         temperature: float = TEMPERATURE,
                         max_tokens: int = MAX_TOKENS) -> Dict[str, Any]:
        """
        Generate a response based on the query and retrieved context.

        Args:
            query: User query string
            context: Retrieved context from knowledge base
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate

        Returns:
            Dictionary containing response and metadata
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")

        # Create the prompt
        prompt = self._create_prompt(query, context)

        try:
            logger.info(f"Generating response for query: '{query}'")

            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. If the context doesn't contain enough information to answer the question, say so clearly."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=max_tokens,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )

            # Extract response content
            generated_text = response.choices[0].message.content.strip()

            # Get usage statistics
            usage = response.usage.model_dump() if response.usage else {}

            result = {
                "response": generated_text,
                "query": query,
                "context_used": bool(context.strip()),
                "model": self.model,
                "usage": usage,
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            logger.info("Response generated successfully")
            return result

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def _create_prompt(self, query: str, context: str) -> str:
        """
        Create a formatted prompt for the language model.

        Args:
            query: User query
            context: Retrieved context

        Returns:
            Formatted prompt string
        """
        if context.strip():
            prompt = f"""Based on the following context, please answer the question.

Context:
{context}

Question: {query}

Answer:"""
        else:
            prompt = f"""Please answer the following question to the best of your ability.

Question: {query}

Answer:"""

        return prompt

    def generate_with_sources(self, query: str, context_docs: List[Dict[str, Any]],
                            temperature: float = TEMPERATURE,
                            max_tokens: int = MAX_TOKENS) -> Dict[str, Any]:
        """
        Generate response with source attribution.

        Args:
            query: User query string
            context_docs: List of context documents with metadata
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens

        Returns:
            Dictionary containing response and source information
        """
        # Extract context text and keep source information
        context_parts = []
        sources = []

        for i, doc_dict in enumerate(context_docs):
            content = doc_dict.get("content", "")
            score = doc_dict.get("similarity_score", 0.0)
            metadata = doc_dict.get("metadata", {})

            context_parts.append(f"[Source {i+1}, Score: {score:.4f}] {content}")

            sources.append({
                "source_id": i + 1,
                "similarity_score": score,
                "content_preview": content[:200] + "..." if len(content) > 200 else content,
                "metadata": metadata
            })

        context = "\n\n".join(context_parts)

        # Generate response
        result = self.generate_response(query, context, temperature, max_tokens)

        # Add source information
        result["sources"] = sources
        result["total_sources"] = len(sources)

        return result

    def validate_api_key(self) -> bool:
        """
        Validate the OpenAI API key.

        Returns:
            True if API key is valid, False otherwise
        """
        try:
            # Make a simple API call to validate the key
            response = self.client.models.list()
            logger.info("OpenAI API key is valid")
            return True
        except Exception as e:
            logger.error(f"OpenAI API key validation failed: {e}")
            return False

    def get_available_models(self) -> List[str]:
        """
        Get list of available OpenAI models.

        Returns:
            List of available model names
        """
        try:
            models = self.client.models.list()
            model_names = [model.id for model in models.data]
            return sorted(model_names)
        except Exception as e:
            logger.error(f"Error fetching available models: {e}")
            return []


def main():
    """Example usage of the SimpleResponseGenerator module."""
    # Initialize generator
    generator = SimpleResponseGenerator()

    # Validate API key
    if not generator.validate_api_key():
        print("Please set a valid OpenAI API key in your .env file")
        return

    # Example query and context
    query = "What are the main benefits of Python programming?"
    context = """
    Python is a high-level programming language known for its simplicity and readability.
    It has a large standard library and extensive ecosystem of third-party packages.
    Python is widely used in web development, data science, machine learning, and automation.
    """

    # Generate response
    result = generator.generate_response(query, context)

    print(f"Query: {result['query']}")
    print(f"Response: {result['response']}")
    print(f"Model: {result['model']}")
    print(f"Usage: {result['usage']}")


if __name__ == "__main__":
    main()
