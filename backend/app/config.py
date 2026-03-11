import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_KEY")
    AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_VERSION")
    azure_embedding_deployment=os.getenv("EMBEDDING_SERVICE_DEPLOYMENT")
    azure_embedding_endpoint=os.getenv("EMBEDDING_SERVICE_ENDPOINT")
    embedding_api_key=os.getenv("EMBEDDING_SERVICE_KEY")
    embedding_api_version=os.getenv("EMBEDDING_SERVICE_VERSION")
    AZURE_SEARCH_ENDPOINT = os.getenv("SEARCH_SERVICE_NAME")
    AZURE_SEARCH_API_KEY = os.getenv("SEARCH_API_KEY")
    AZURE_SEARCH_INDEX = os.getenv("SEARCH_INDEX_NAME")

settings = Settings()
