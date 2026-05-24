from langchain_core.documents import Document
from pydantic import BaseModel





class RAGstate(BaseModel):
    "RAG state object"
    question: str
    retrieved_docs: list[Document]=[]
    answer:str=""

