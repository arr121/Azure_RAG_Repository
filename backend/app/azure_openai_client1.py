from azure.ai.openai import OpenAIClient
from openai import OpenAI
from azure.core.credentials import AzureKeyCredential
from .config import settings

# Create the Azure OpenAI client
client=OpenAIClient(
    endpoint=settings.AZURE_OPENAI_ENDPOINT,
    credential=AzureKeyCredential(settings.AZURE_OPENAI_API_KEY)
)
# Function to generate embeddings using Azure OpenAI
def generate_embedding(text:str):
    response = client.embeddings.create(
        model=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        input=text
    )
    return response.data[0].embedding
def generate_chat_complete(prompt:str):
    response =client.chat.completions.create(
        model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
        messages=[{"role":"user","content":prompt}]
    )
    return response.choices[0].message.content
