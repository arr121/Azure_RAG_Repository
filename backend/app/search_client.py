from azure.search.documents import SearchClient
from .config import settings
from azure.core.credentials import AzureKeyCredential

# Configure the Azure Search
search_client = SearchClient(
    endpoint=settings.AZURE_SEARCH_ENDPOINT,
    index_name=settings.AZURE_SEARCH_INDEX,
    credential=AzureKeyCredential(settings.AZURE_SEARCH_API_KEY)
)
# Function to perform vector search
def vector_search(embedding, top_k=5):
    results = search_client.search(
        search_text="",  # Empty search text for vector search
        vectors=[
            {
                "field": "content_vector",  # Must match your index schema
                "value": embedding,
                "k": top_k
            }
        ],
        top_k=top_k,
        include_total_count=True
        docs=[]
        for result in results:
            docs.append({
                "id": result["id"],
                "bill_id": result.get("bill_id"),
                "title": result.get("title"),
                "summary": result.get("summary"),
                "content": result.get("content"),
                "metadata": result.get("metadata"),
                "score": result["@search.score"]
            })


    )
    return docs