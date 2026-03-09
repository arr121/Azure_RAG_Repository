import os
import gc
from dotenv import load_dotenv
from operator import itemgetter

from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

def create_index(csv_path: str):
    loader = CSVLoader(file_path=csv_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    embedding = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("EMBEDDING_SERVICE_DEPLOYMENT"),
        azure_endpoint=os.getenv("EMBEDDING_SERVICE_ENDPOINT"),
        api_key=os.getenv("EMBEDDING_SERVICE_KEY"),
        api_version=os.getenv("EMBEDDING_SERVICE_VERSION")
    )

    vector_store = AzureSearch(
        azure_search_endpoint=os.getenv("SEARCH_SERVICE_NAME"),
        azure_search_key=os.getenv("SEARCH_API_KEY"),
        index_name=os.getenv("SEARCH_INDEX_NAME"),
        embedding_function=embedding.embed_query,
    )
    vector_store.add_documents(splits)
    return vector_store.as_retriever(search_type="hybrid")

def create_rag_chain(csv_path: str):
    retriever = create_index(csv_path)
    
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version=os.getenv("AZURE_OPENAI_VERSION"),
        temperature=0.0,
    )

    # Clean prompt structure
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful legal assistant. Answer using ONLY the provided context."),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}")
    ])

    # The Chain: 
    # 1. Takes input dict {"question": "..."}
    # 2. Uses "question" to fetch context
    # 3. Formats context
    # 4. Passes everything to the prompt
    chain = (
        {"context": itemgetter("question") | retriever | format_docs, 
         "question": itemgetter("question")}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

if __name__ == "__main__":
    rag_chain = create_rag_chain("bill_sum_data.csv")
    
    # Pass the input as a dictionary matching the "question" key
    response = rag_chain.invoke({"question": "What are the key points of bill id 113?"})
    print("RAG Answer:", response)
    
    gc.collect()