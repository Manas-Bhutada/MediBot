from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter 

from langchain_huggingface import HuggingFaceEmbeddings 

from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
#step 1 pdf ke pages load kar rha hu
DATA_PATH='data/'
def load_pdf_files(data):  #idhar jitni bhi .pdf files hongi data m unko load karke return karega
    loader=DirectoryLoader(data,glob='*.pdf',loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents

documents=load_pdf_files(data=DATA_PATH)

print("length of pdf pages",len(documents))

#step 2 creating chunks
def create_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks=create_chunks(documents);

print("length of text chunks",len(text_chunks))

#step 3 now i am making embeddings mera model text chunks ko embedinngs me banayega


def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model=get_embedding_model()


#step 4 ab jo embedings h na unko faiss db m daalenge

DB_FAISS_PATH="vectorstore/db_faiss"
db=FAISS.from_documents(text_chunks,embedding_model)
db.save_local(DB_FAISS_PATH)
 