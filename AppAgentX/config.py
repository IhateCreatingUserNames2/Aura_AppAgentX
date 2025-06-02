import os
from dotenv import load_dotenv
import logging

logger_appagentx_config = logging.getLogger(__name__)

CONFIG_FILE_PATH = os.path.abspath(__file__) # Full path to this config.py
APPAGENTX_DIR = os.path.dirname(CONFIG_FILE_PATH) # AppAgentX/ directory
PROJECT_ROOT = os.path.dirname(APPAGENTX_DIR) # Aura2/ directory
DOTENV_PATH = os.path.join(PROJECT_ROOT, '.env')

logger_appagentx_config.info("AppAgentX/config.py: Attempting to load .env...") # Log attempt
if os.path.exists(DOTENV_PATH):
    load_dotenv(DOTENV_PATH, override=True)
    logger_appagentx_config.info(f"AppAgentX/config.py: Successfully loaded .env from {DOTENV_PATH}")
else:
    logger_appagentx_config.warning(f"AppAgentX/config.py: .env file not found at {DOTENV_PATH}. Using defaults or existing env vars.")


if os.path.exists(DOTENV_PATH):
    print(f"AppAgentX/config.py: Loading .env from {DOTENV_PATH}")
    # override=True ensures that if .env is loaded multiple times,
    # the values from this load (if the variables are in .env) will take precedence
    # or be updated if already set by an earlier load_dotenv call in the same process.
    load_dotenv(DOTENV_PATH, override=True)
else:
    print(f"AppAgentX/config.py: .env file not found at {DOTENV_PATH}. Relying on pre-set environment variables or defaults.")

# LLM Configuration
# These settings control the connection and behavior of the Large Language Model API
# Please fill in your own API information below

LLM_BASE_URL = os.getenv("APPAGENTX_LLM_BASE_URL", "")
# Base URL for the LLM API service, using a proxy to access OpenAI's API
# Please enter your LLM service base URL here

LLM_API_KEY = os.getenv("APPAGENTX_LLM_API_KEY", "sk-proj-")
# API key for authentication with the LLM service
# Please enter your LLM API key here

LLM_MODEL = os.getenv("APPAGENTX_LLM_MODEL", "gpt-4o")
# Specific LLM model version to be used for inference
# You can use OpenAI models like "gpt-4o" or DeepSeek models like "deepseek-chat"

LLM_MAX_TOKEN = int(os.getenv("APPAGENTX_LLM_MAX_TOKEN", 1500))
# Maximum number of tokens allowed in a single LLM request

LLM_REQUEST_TIMEOUT = int(os.getenv("APPAGENTX_LLM_REQUEST_TIMEOUT", 500))
# Timeout in seconds for LLM API requests

LLM_MAX_RETRIES = int(os.getenv("APPAGENTX_LLM_MAX_RETRIES", 3))
# Maximum number of retry attempts for failed LLM API calls

# LangChain Configuration
# Settings for LangChain integration and monitoring
# Uncomment and fill in the following settings if you need LangSmith functionality

LANGCHAIN_TRACING_V2 = os.getenv("APPAGENTX_LANGCHAIN_TRACING_V2", "false")
# Enables LangSmith tracing for debugging and monitoring

LANGCHAIN_ENDPOINT = os.getenv("APPAGENTX_LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
# Endpoint URL for LangSmith API services

LANGCHAIN_API_KEY = os.getenv("APPAGENTX_LANGCHAIN_API_KEY", "lsv2_")
# API key for authentication with LangSmith services
# Please enter your LangSmith API key here if needed

LANGCHAIN_PROJECT = os.getenv("APPAGENTX_LANGCHAIN_PROJECT", "AppAgentX_Project")
# Project name for organizing LangSmith resources

# Neo4j Configuration
# Settings for connecting to the Neo4j graph database
# Please update these settings according to your Neo4j installation
logger_appagentx_config.info("AppAgentX/config.py: Defining configuration variables...")
Neo4j_URI = os.getenv("APPAGENTX_NEO4J_URI", "neo4j://127.0.0.1:7687") # Fallback to localhost if not in .env or env
_neo4j_user = os.getenv("APPAGENTX_NEO4J_AUTH_USER", "neo4j")
_neo4j_pass = os.getenv("APPAGENTX_NEO4J_AUTH_PASS", "12345678") # Be careful with default passwords
Neo4j_AUTH = (_neo4j_user, _neo4j_pass)
# Authentication credentials (username, password) for Neo4j
# Please update with your actual Neo4j credentials
logger_appagentx_config.info("AppAgentX/config.py: Configuration variables defined.")
# Feature Extractor Configuration
# Settings for the feature extraction service
# Please ensure this service is running at the specified address

Feature_URI = os.getenv("APPAGENTX_FEATURE_URI", "http://127.0.0.1:8001")
# URI for the feature extraction service API
# Default is localhost port 8001, update if needed

# Screen Parser Configuration
# Settings for the screen parsing service
# Please ensure this service is running at the specified address

Omni_URI = os.getenv("APPAGENTX_OMNI_URI", "http://127.0.0.1:8000")

# URI for the Omni screen parsing service API
# Default is localhost port 8000, update if needed

# Vector Storage Configuration
# Settings for the vector database used for embeddings storage
# Please fill in your vector database information

PINECONE_API_KEY = "pcsk_"
# API key for authentication with Pinecone vector database service
# Please enter your Pinecone API key here
