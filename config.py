import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.llmod.ai/v1")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ted-talks")

# RAG Hyperparameters (Defaults, can be overridden)
CHUNK_SIZE = 1000  # Starting conservative, requirement max is 2048 tokens
CHUNK_OVERLAP = 200 # Starting conservative, requirement max is 30%
TOP_K = 20

# Paths
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "ted_talks_en.csv")
