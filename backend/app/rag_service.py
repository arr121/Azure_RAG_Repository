from .azure_openai_client import generate_embedding, generate_chat_completion
from .search_client import vector_search

def build_prompt(question, documents):
    context = "\n\n".join([doc["content"] for doc in documents])

    prompt = f"""
You are an AI assistant. Use ONLY the context below to answer the question.

Context:
{context}

Question:
{question}

Answer clearly and concisely.
"""
    return prompt

def answer_question(question: str):
    # 1. Generate embedding
    embedding = generate_embedding(question)

    # 2. Vector search
    docs = vector_search(embedding)

    # 3. Build prompt
    prompt = build_prompt(question, docs)

    # 4. Get LLM answer
    answer = generate_chat_completion(prompt)

    return {
        "answer": answer,
        "sources": docs
    }
