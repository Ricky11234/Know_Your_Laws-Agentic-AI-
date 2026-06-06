from langchain_core.documents import Document
from pydantic import BaseModel


class RAGstate(BaseModel):

    question:str

    retrieved_docs:list[Document]=[]

    answer:str=""

    context_quality:str=""

    source_type:str=""

    external_context:str=""