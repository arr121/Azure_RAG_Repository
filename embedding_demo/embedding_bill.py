import os
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import (
    AzureOpenAIEmbeddings, 
    AzureChatOpenAI
)
from langchain_classic.chains import RetrievalQA
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
#from langchain_community.chains import RetrievalQA
#from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import create_retriever_tool

  # Load environment variables from .env file

# 1. Load and index CSV for RAG (creates/updates index)
def create_index(csv_path: str, index_name: str = "bills-index"):
    """Load CSV, split, embed, and index in Azure AI Search."""
    loader = CSVLoader(file_path=csv_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    #OpenAIEmbeddings (or AzureOpenAIEmbeddings when using Azure) is an embedding generator. Converts text → vector
    embedding=AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("EMBEDDING_SERVICE_DEPLOYMENT"),  # Your embedding deployment
        azure_endpoint=os.getenv("EMBEDDING_SERVICE_ENDPOINT"),
        api_key=os.getenv("EMBEDDING_SERVICE_KEY"),
        api_version=os.getenv("EMBEDDING_SERVICE_VERSION")
    )

    #AzureSearch is a vector database. It stores the vectors and allows you to search for similar vectors.
    vector_store = AzureSearch.from_documents(
        splits,  # CSV chunks (bill_id, summary, title → content)
        embedding=embedding,  # embedding generator
        index_name=os.getenv("SEARCH_INDEX_NAME"),
        azure_search_endpoint=os.getenv("SEARCH_SERVICE_NAME"),
        azure_search_key=os.getenv("SEARCH_API_KEY"),
    )
    retriever = vector_store.as_retriever(search_type="hybrid")
    print(f"Indexed {len(splits)} chunks into {index_name}")
    return retriever

# 2. Create RAG Agent
def create_rag_agent(csv_path: str):
    retriever = create_index(csv_path)
    
    # 4. Initialize Azure Chat model
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version=os.getenv("AZURE_OPENAI_VERSION"),
        temperature=0.0,
    )
     # RAG Chain (RetrievalQA - stable replacement)
    system_message = """
        You are a helpful assistant specialized in answering questions about legal bills.
        Answer the user's question using ONLY the provided context below.
        If the answer is not in the context, state that you do not have enough information.
        """

    # 2. Human Prompt: Contains the dynamic data (context + question)
    human_message = """
    Context:
    {context}

    Question:
    {question}
    """

    # Combine into a ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", human_message)
    ])
        
    '''rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )'''
    chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    )
    
    # Test RAG chain
    response = chain.invoke({"query": "How do you explain about Taxpayer's Right to View Act of 1993?"})
    print("RAG Answer:", response["result"])
    return response
#4 . Helper function to format retrieved documents for the prompt
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])
# 3. Usage
if __name__ == "__main__":
    agent_executor = create_rag_agent("bill_sum_data.csv")  # Upload your CSV here
    response = agent_executor.invoke({"input": "What are the key points of bill id 113?"})
    print(response["output"])