from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import os 
from dotenv import load_dotenv

load_dotenv()
chroma_path = os.getenv("path_save")
path_pdf = os.getenv("pdf_path")



def Load_pdf(path_pdf):
    loder = PyPDFLoader(path_pdf)
    return loder.load()

document = Load_pdf(path_pdf)

def Split_document(docs):
    spliter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )
    return spliter.split_documents(docs)

chunks = Split_document(document)

def get_embedding_function():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def add_to_chroma(chunks,chroma_path):
    db = Chroma(
        persist_directory=chroma_path,
        embedding_function=get_embedding_function()
    )
    db.add_documents(chunks)
    return db

add_to_chroma(chunks, chroma_path)



