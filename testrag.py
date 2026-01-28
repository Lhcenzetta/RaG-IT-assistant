# from langchain_community.document_loaders.pdf import PyPDFLoader
# from langchain_community.llms import openai
# from langchain_chroma import Chroma
# from langchain_community.embeddings import OpenAIEmbeddings  
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.embeddings.bedrock import BedrockEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
# import os 
# from dotenv import load_dotenv


# # load_dotenv()  

# # os.getenv["key_open_ia"]


# loader = PyPDFLoader ("/Users/lait-zet/Desktop/RaG-IT-assistant/Document/data.pdf")
# document = loader.load_and_split()

# split = RecursiveCharacterTextSplitter(
#     chunk_size = 500,
#     chunk_overlap = 50,
#     add_start_index = True,
#     length_function=len
# )

# chunks = split.split_documents(document)
# print(len(chunks))
# def get_embeding_function():
#     embeding=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
#     return embeding

# def add_to_chroma(chunks : list[document]):
#     db = Chroma(
#     persist_directory = "/Users/lait-zet/Desktop/RaG-IT-assistant/db_chroma",
#     embedding_function = get_embeding_function())
#     db.add_documents(chunks)
#     print("Number of documents in DB:", db._collection.count())
#     # db.persist()

# add_to_chroma(chunks)
    





# # def add_to_chroma(save_path,chunks):
# #     db = Chroma(
# #         persist_directory = save_path,
# #         embedding_function=get_embedding_function()
# #     )  
# #     db.add_documents(chunks)

# # add_to_chroma("/Users/lait-zet/Desktop/RaG-IT-assistant/db_chroma",chunks)

# # for chunk in chunks :
# #     source = chunk.metadata.get("source")
# #     page = chunk.metadata.get("page")
# #     current_shrunk_id = f"{source}:page{page}"
# #     print(current_shrunk_id)

# # def query_rag(query_text):
# #     db = Chroma(
# #         embedding_function=get_embedding_function(),
# #         persist_directory="/Users/lait-zet/Desktop/RaG-IT-assistant/db_chroma/chroma.sqlite3"
# #     )
# #     promt = """ Answer the question based only on the following context : {context}answer the question based on the above question : {question}"""
# #     result = db._similarity_search_with_relevance_scores(query_text, k = 4)
# #     print(query_text)


# query_text = "introduce to it"

# db = Chroma(
#     embedding_function=get_embedding_function(),
#     persist_directory="/Users/lait-zet/Desktop/RaG-IT-assistant/db_chroma"
# )

# prompt_template_text = """
# Answer the question using ONLY the context below.

# Context:
# {context}

# Question:
# {question}
# """

# results = db.similarity_search_with_relevance_scores(query_text, k=2)

# context_text = "\n\n--\n\n".join(
#     [doc.page_content for doc, score in results]
# )

# prompt_template = ChatPromptTemplate.from_template(prompt_template_text)

# prompt = prompt_template.format(
#     context=context_text,
#     question=query_text
# )