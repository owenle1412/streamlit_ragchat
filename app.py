import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

# --- Streamlit page setup ---
st.set_page_config(page_title="📄 RAG Chat with PDF", layout="wide")
st.title("📄 RAG Chatbot with PDF + Local Chroma")

# --- Paths and constants ---
PERSIST_DIR = "chroma_store"
os.makedirs(PERSIST_DIR, exist_ok=True)

# --- Load API key from Streamlit secrets ---
api_key = st.secrets["OPENAI_API_KEY"]

# --- Initialize LLM and embeddings ---
llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
embeddings = OpenAIEmbeddings(api_key=api_key)

# --- Create two columns ---
col1, col2 = st.columns([2, 1])

# ---------------------------------------------------
# Right column — file upload and embedding creation
# ---------------------------------------------------
with col2:
    st.subheader("📂 Document Upload")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    vectordb = None

    if uploaded_file:
        pdf_path = "temp.pdf"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        st.info("📚 Loading PDF and splitting into chunks...")
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(pages)
        st.success(f"✅ Loaded {len(chunks)} text chunks")

        # Create Chroma vectorstore
        vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_DIR)
        vectordb.persist()
        st.success("✅ Embeddings saved locally in 'chroma_store/'")

        st.caption("You can now start chatting with your document on the left →")

# ---------------------------------------------------
# Left column — chat interface
# ---------------------------------------------------
with col1:
    st.subheader("💬 Chat with your document")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if vectordb is None and not os.path.exists(PERSIST_DIR):
        st.warning("👈 Please upload a PDF first (right panel)")
    else:
        if vectordb is None:
            # Load existing Chroma DB if it exists
            vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

        user_query = st.text_input("Ask a question about the document:")

        if st.button("Ask"):
            if user_query.strip():
                with st.spinner("🔍 Searching document and generating answer..."):
                    docs = vectordb.similarity_search(user_query, k=3)
                    context = "\n\n".join([d.page_content for d in docs])

                    # Build RAG prompt
                    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Use only the provided context to answer.
If you can’t find the answer, say "I couldn’t find that in the document."

Context:
{context}

Question: {question}
Answer:
""")

                    chain = prompt | llm
                    response = chain.invoke({"context": context, "question": user_query})
                    answer = response.content

                    # Save conversation
                    st.session_state.chat_history.append((user_query, answer))

        # --- Display chat history ---
        for question, answer in reversed(st.session_state.chat_history):
            with st.chat_message("user"):
                st.markdown(f"**You:** {question}")
            with st.chat_message("assistant"):
                st.markdown(f"**Bot:** {answer}")
