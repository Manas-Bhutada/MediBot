import os
import streamlit as st
import warnings
from dotenv import load_dotenv, find_dotenv
from urllib.parse import urlencode

from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

from auth.google_oauth import get_login_url, fetch_tokens, get_user_info

# Load environment variables
load_dotenv(find_dotenv())

# Fix torch and HuggingFace warnings
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
warnings.filterwarnings("ignore", category=FutureWarning)

# Path to FAISS DB
DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    return db

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def load_llm():
    return HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        temperature=0.5,
        max_new_tokens=512,
        token=os.getenv("HF_TOKEN")
    )

def main():
    st.set_page_config(page_title="MediBot", page_icon="🧠")
    st.title("🤖 AI Medical Chatbot")

    # Initialize session states 
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user_info" not in st.session_state:
        st.session_state.user_info = {}

    # Handle OAuth redirect
    query_params = st.experimental_get_query_params()
    if "code" in query_params and not st.session_state.authenticated:
        with st.spinner("🔐 Logging in..."):
            code = query_params["code"][0]
            tokens = fetch_tokens(code)
            access_token = tokens.get("access_token")
            if access_token:
                user_info = get_user_info(access_token)
                st.session_state.authenticated = True
                st.session_state.user_info = user_info
                st.experimental_set_query_params()  # Clean the URL

    # Sidebar Login Section
    with st.sidebar:
        st.subheader("🔐 Login Required")
        if st.session_state.authenticated:
            user = st.session_state.user_info
            st.success(f"✅ Logged in as {user.get('email')}")
            st.markdown(f"👤 **{user.get('name')}**")
            st.image(user.get("picture"), width=50)
        else:
            login_url = get_login_url()
            st.markdown(f"[Login with Google]({login_url})", unsafe_allow_html=True)

        st.divider()
        st.title("⚙️ Chatbot Settings")
        st.info("💡 Ask me medical-related queries! I fetch answers from trusted medical sources.")

    # Require login
    if not st.session_state.authenticated:
        st.warning("🔒 Please log in to access the chatbot.")
        return

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Show chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            role = 'user' if message['role'] == 'user' else 'assistant'
            with st.chat_message(role):
                st.markdown(message['content'])

    # Chat input
    prompt = st.chat_input("💬 Type your medical question here")

    if prompt:
        with st.chat_message("user"):
            st.markdown(f"**You:** {prompt}")
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        # Prompt Template
        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer the user's question.
        If you don't know the answer, just say that you don't know; don't make up an answer.
        Stick strictly to the given context.

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk.
        """

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("⚠️ Failed to load the vector store")

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(),
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
