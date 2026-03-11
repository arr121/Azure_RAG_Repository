from fastapi import FastAPI
from pydantic import BaseModel
from .rag_service import answer_question

app = FastAPI()

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask_question(payload: Question):
    response = answer_question(payload.question)
    return response

@app.get("/")
def read_root():
    return {"message": "Backend is running!"}
