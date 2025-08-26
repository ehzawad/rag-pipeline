# RAG Pipeline Configuration

# OpenAI API Key - set this in your .env file
OPENAI_API_KEY = "your_openai_api_key_here"

# Embedding model settings
EMBEDDING_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION=384

# Vector store settings
VECTOR_STORE_PATH="./data/vector_store"
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Retrieval settings
TOP_K_RETRIEVAL=5
SEARCH_TYPE="similarity"

# Generation settings
MODEL_NAME="gpt-5-mini"
TEMPERATURE=0.1
MAX_TOKENS=500

# Data settings
DATA_PATH="./data"
SUPPORTED_FILE_TYPES=[".txt", ".pdf", ".md"]
