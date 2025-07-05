import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

#step 1 setup llm mistral with huggingface

HF_TOKEN=os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"

def load_llm():
    return HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        temperature=0.5,
        max_new_tokens=512,                    # ← directly here
        token=os.getenv("HF_TOKEN")            # ← directly here
    )



#step 2 connect llm with faiss and create chain


CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(CUSTOM_PROMPT_TEMPLATE):
    prompt=PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE,input_variables=["context","question"])
    
    return prompt
DB_FAISS_PATH="vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    pipeline="sentence-similarity",
    huggingfacehub_api_token=os.getenv("HF_TOKEN")
)

db=FAISS.load_local(DB_FAISS_PATH,embedding_model,allow_dangerous_deserialization=True) #idhar true isliye kyuki secured source of info h 

#create QA chain

qa_chain=RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k':3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

#NOW INVOKE WITH
user_query=input("Write Quer Here: ")
response=qa_chain.invoke({'query':user_query})

print("Result of the query:",response["result"])
print("Source Documents:",response["source_documents"])