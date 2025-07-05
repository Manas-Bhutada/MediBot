import os
import streamlit as st

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Set up FAISS Vector Store Path
DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm():
    return HuggingFaceEndpoint(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        temperature=0.5,
        model_kwargs={
            "max_new_tokens": 512,
            "token": os.getenv("HF_TOKEN")
        }
    )

def main():
    st.title("🤖 AI Medical Chatbot")

    # Sidebar with chatbot information
    st.sidebar.title("⚙️ Chatbot Settings")
    st.sidebar.info("💡 Ask me medical-related queries! I fetch answers from trusted medical sources.")

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history using Streamlit's built-in chat formatting
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            role = 'user' if message['role'] == 'user' else 'assistant'
            with st.chat_message(role):
                st.markdown(message['content'])

    # Get user input
    prompt = st.chat_input("💬 Type your message here")

    if prompt:
        with st.chat_message("user"):
            st.markdown(f"**You:** {prompt}")
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        # Define the chatbot response template
        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer the user's question.
        If you don't know the answer, just say that you don't know; don't make up an answer.
        Stick strictly to the given context.

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk.
        """

        HUGGINGFACE_REPO_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"

        HF_TOKEN = os.environ.get("HF_TOKEN")

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("⚠️ Failed to load the vector store")

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, hf_token=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})

            result = response["result"]
            source_documents = response["source_documents"]
            result_to_show = f"**Chatbot:** {result}\n\n**📄 Source Docs:** {str(source_documents)}"

            with st.chat_message("assistant"):
                st.markdown(result_to_show)
            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
