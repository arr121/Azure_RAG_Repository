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
    prompt_template = """Use the CSV data below to answer. If unknown, say so.
    Data :{context}
    Question: {question}
    Answer:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    # Test RAG chain
    response = rag_chain.invoke({"query": "What did the president say about Ketanji Brown Jackson?"})
    print("RAG Answer:", response["result"])
    
    # Agent Tool (retriever as tool)
    retriever_tool = create_retriever_tool(
        retriever,
        "csv_search",
        "Search CSV data for relevant information. Use for RAG queries."
    )
    tools = [retriever_tool]
    
    # ReAct Agent (stable)
    agent_prompt = PromptTemplate.from_template("""
    Answer questions about CSV data using tools. Be specific, cite sources.
    
    {chat_history}
    Question: {input}
    Thought: {agent_scratchpad}
    """)
    
    agent = create_react_agent(llm, tools, agent_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    return agent_executor

# 3. Usage
if __name__ == "__main__":
    agent_executor = create_rag_agent("bill_sum_data.csv")  # Upload your CSV here
    response = agent_executor.invoke({"input": "What are the key points of bill id 113?"})
    print(response["output"])