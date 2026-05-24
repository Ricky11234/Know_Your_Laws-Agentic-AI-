from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

class Retriever:

 def __init__(self):
    self.embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    self.vectorstore=None
    self.retriever=None

 def create_retriever(self,docs:list[Document]): 
    if not docs:
      raise ValueError("No documents provided")
    self.vectorstore=FAISS.from_documents(docs,self.embeddings)
    self.retriever=self.vectorstore.as_retriever(
    search_kwargs={"k":8}
)



 def retrieve(self,query:str,k:int =4)-> list[Document]: 
    if(self.retriever==None):
       raise ValueError("Call create_retriever() first")
    else:
     response=self.retriever.invoke(query)  
     return response[:k]

        
