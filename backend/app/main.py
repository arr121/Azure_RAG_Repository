from fastapi import FastAPI
from uvicorn import run
app = FastAPI()
@app.get("/")
def read_root():
    return {"Hello": "World"}