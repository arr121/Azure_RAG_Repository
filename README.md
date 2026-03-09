Retrieval Augmented Generation (RAG) with Azure
A Retrieval Augmented Generation example with Azure, using Azure OpenAI Service, Azure Cognitive Search, embeddings, and a sample CSV file to produce a powerful grounding to applications that want to deliver customized generative AI applications.

Install the prerequisites
Use the requirements.txt to install all dependencies

python -m venv .venv
./.venv/bin/pip install -r requirements.txt
Add your keys
Find the Azure OpenAI Keys in the Azure OpenAI Service. Note, that keys aren't in the studio, but in the resource itself. Add them to a local .env file. This repository ignores the .env file to prevent you (and me) from adding these keys by mistake.

Your .env file should look like this:
# Azure OpenAIEmbedding
First, ingest your documents, convert them into vector embeddings using AzureOpenAIEmbeddings, and store them in your AzureSearch vector index.
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="your-embedding-deployment",
    azure_endpoint="https://your-endpoint.openai.azure.com/",
    api_key="your-api-key",
)

# Azure Cognitive Search
vector_store = AzureSearch(
    azure_search_endpoint="https://your-search-service.search.windows.net",
    azure_search_key="your-search-key",
    index_name="your-index-name",
    embedding_function=embeddings.embed_query,
)
Note that the Azure Cognitive Search is only needed if you are following the Retrieval Augmented Guidance (RAG) demo. It isn't required for a simple Chat application.

# Configure retriever to perform semantic or hybrid search
retriever = vector_store.as_retriever(search_type="hybrid")

# Initialize the AzureChatOpenAI model and define a prompt template that provides the retrieved context to the LLM.
llm = AzureChatOpenAI(
    azure_deployment="your-chat-deployment",
    azure_endpoint="https://your-endpoint.openai.azure.com/",
    api_key="your-api-key",
)
prompt = ChatPromptTemplate.from_template("""
    Answer the question based only on the following context:
    {context}
    Question: {question}
""")
# Create a chain that combains retriever and LLM , to process input and generate output
  
chain = (
    # "context": retriever: It takes the user's input, passes it to the retriever 
    # It passes the context and question to prompt 
    {"context": retriever, "question": RunnablePassthrough()}
    #prompt takes the dictionary values and pass it to LLM
    | prompt
    #LLM receives the prompt message and it performs the inference and AI message is returned
    | llm
    # Output is parsed
    | StrOutputParser()
)
chain.invoke({"question": "what is most visited place in Frankfurt?"})
