from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

import os
from dotenv import load_dotenv


load_dotenv()
PDF_PATH = os.getenv("pdf_path")
DB_PATH = "/Users/lait-zet/Desktop/RaG-IT-assistant/db_chroma"




def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    return loader.load()

documents = load_pdf(PDF_PATH)


def split_documents(docs, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

chunks = split_documents(documents)


def get_embedding_function():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def add_to_chroma(chunks : list[documents]):
    db = Chroma(
    persist_directory = DB_PATH,
    embedding_function = get_embedding_function())
    db.add_documents(chunks)
    print("Number of documents in DB:", db._collection.count())
    return db

# db = add_to_chroma(chunks)

db =  db = Chroma(
    persist_directory = DB_PATH,
    embedding_function = get_embedding_function()
)
print("Number of documents in DB:", db._collection.count())
retriever = db.as_retriever(search_kwargs={"k": 3}) 


prompt = ChatPromptTemplate.from_template("""
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{content}

Question:
{question}
""")

# Debugging step


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)


query = "What is a messi"

results = db.similarity_search_with_relevance_scores(query, k=3)

SIMILARITY_THRESHOLD = 0.4

def safe_retrieve(query: str):
    results = db.similarity_search_with_relevance_scores(query, k=3)

    filtered_docs = [
        doc for doc, score in results if score >= SIMILARITY_THRESHOLD
    ]

    if not filtered_docs:
        return None  

    return filtered_docs
query = "What is a computer"

docs = safe_retrieve(query)

if docs is None:
    print("I don't know (not in the PDF)")
else:
    print("Retrieved chunks:")
    for d in docs:
        print(d.page_content[:300])