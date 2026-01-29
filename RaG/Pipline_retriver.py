from langchain_chroma import Chroma
import os 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage


load_dotenv()
chroma_path = os.getenv("path_save")

def get_embedding_function():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma(
    persist_directory=chroma_path,
    embedding_function=get_embedding_function(),
    collection_metadata={"hnsw" : "cosine"}
)

def Handle_query(query):
    retriver = db.as_retriever(searche_kwargs = {"k" : 3})
    revalent_docs = retriver.invoke(query)

    promt = f""" based on the following document , please answer this  question {query} documents:
    {chr(10).join([f"- {doc.page_content}" for doc in revalent_docs])}
    please provide a clear helpful answer using only the information from these documents . if you can't find the answer in
    documents . say "i don't have anough information based on the documentation please try other question
    """
    llm  = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        )
    messages = [
        SystemMessage(content = "You are  a helpful assistance"),
        HumanMessage(content = promt),
        ]
    result = llm.invoke(messages).content
    return result




query = " what is computer"

result = Handle_query(query)
print(result)





