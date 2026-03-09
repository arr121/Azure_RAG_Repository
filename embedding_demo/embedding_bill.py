import os
import gc
from dotenv import load_dotenv
from operator import itemgetter
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 1. Setup Embeddings
def get_embeddings():
    return AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("EMBEDDING_SERVICE_DEPLOYMENT"),
        azure_endpoint=os.getenv("EMBEDDING_SERVICE_ENDPOINT"),
        api_key=os.getenv("EMBEDDING_SERVICE_KEY"),
        api_version=os.getenv("EMBEDDING_SERVICE_VERSION")
    )

# 2. Setup Vector Store (Connect to EXISTING index, DO NOT RECREATE)
def get_vector_store():
    embedding = get_embeddings()
    return AzureSearch(
        azure_search_endpoint=os.getenv("SEARCH_SERVICE_NAME"),
        azure_search_key=os.getenv("SEARCH_API_KEY"),
        index_name="bills-index",  # Must match your JSON
        embedding_function=embedding.embed_query,
        # Map your custom schema fields from your JSON
        content_key="content",
        vector_key="content_vector",
        metadata_key="metadata"
    )

# 3. Setup Chain
def create_rag_chain():
    retriever = get_vector_store().as_retriever(search_type="hybrid")
    
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version=os.getenv("AZURE_OPENAI_VERSION"),
        temperature=0.0,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use ONLY the provided context."),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}")
    ])

    # Helper to clean retrieved documents
    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    # LCEL Chain
    chain = (
        {"context": itemgetter("question") | retriever | format_docs, 
         "question": itemgetter("question")}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

if __name__ == "__main__":
    try:
        rag_chain = create_rag_chain()
        response = rag_chain.invoke({"question": "What are the key points of bill id 113?"})
        print("RAG Answer:", response)
    finally:
        # Force garbage collection to avoid the AzureSearch __del__ shutdown error
        gc.collect()