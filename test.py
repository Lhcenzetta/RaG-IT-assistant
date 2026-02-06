# from passlib.context import CryptContext

# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# def create_hash_mode_pass(password):
#     return pwd_context.hash(password)

# def verfiy_hash_passsword(new_password , hashed_password):
#     return pwd_context.verify(new_password, hashed_password)

# hash_code = create_hash_mode_pass("1234")
# print(hash_code)
# print(verfiy_hash_passsword("1234",hash_code))

def Handle_query(query):

    with mlflow.start_run(run_name="rag_query"):

        # -------------------
        # PARAMS
        # -------------------
        mlflow.log_param("llm_model", "gemini-2.5-flash")
        mlflow.log_param("temperature", 0.3)
        mlflow.log_param("vector_store", "chroma")
        mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")
        mlflow.log_param("top_k", 3)

        # -------------------
        # QUERY
        # -------------------
        mlflow.log_text(query, "input/query.txt")

        # -------------------
        # RETRIEVAL
        # -------------------
        t0 = time.time()

        retriver = db.as_retriever(search_kwargs={"k": 3})
        revalent_docs = retriver.invoke(query)

        retrieval_latency = (time.time() - t0) * 1000
        mlflow.log_metric("retrieval_latency_ms", retrieval_latency)

        # log chunks
        chunks_text = "\n\n".join(
            f"CHUNK {i+1}:\n{doc.page_content}"
            for i, doc in enumerate(revalent_docs)
        )
        mlflow.log_text(chunks_text, "rag/retrieved_chunks.txt")

        # -------------------
        # PROMPT
        # -------------------
        promt = f"""based on the following document, please answer this question:
{query}

documents:
{chr(10).join([f"- {doc.page_content}" for doc in revalent_docs])}

If the answer is not in the documents, say:
"I don't have enough information based on the documentation."
"""

        mlflow.log_text(promt, "prompt/final_prompt.txt")

        # -------------------
        # GENERATION
        # -------------------
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
        )

        messages = [
            SystemMessage(content="You are a helpful assistance"),
            HumanMessage(content=promt),
        ]

        t1 = time.time()
        result = llm.invoke(messages).content
        generation_latency = (time.time() - t1) * 1000

        mlflow.log_metric("generation_latency_ms", generation_latency)
        mlflow.log_metric(
            "total_latency_ms",
            retrieval_latency + generation_latency
        )

        # -------------------
        # OUTPUT
        # -------------------
        mlflow.log_text(result, "output/answer.txt")

        return result


query = "what is computer"
result = Handle_query(query)
print(result)