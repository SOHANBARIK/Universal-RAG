import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# --- 1. CONFIGURATION & LANGSMITH SETUP ---
# Load environment variables first
load_dotenv()

# Explicitly map your specific .env key to LangSmith's expected variable
if os.getenv("Langchain_Key"):
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("Langchain_Key")
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "Gemini-RAG" # You can name this whatever you want
else:
    # Optional: Warn if the key is missing so you know why tracing isn't working
    print("Warning: 'Langchain_Key' not found in .env. LangSmith tracing is disabled.")

# Streamlit Page Config
st.set_page_config(page_title="Universal RAG Chatbot", layout="wide", initial_sidebar_state="expanded", page_icon="❓")
st.title("🤖 Gemini RAG: Chat with PDF, Website, or YouTube")

# --- 2. IMPORTS (Keep these after config) ---
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, YoutubeLoader

# --- SIDEBAR: SETTINGS ---
with st.sidebar:
    st.header("Data Source")
    
    # Check for Google API Key
    api_key = os.getenv("Google_API_KEY")
    if not api_key:
        st.error("⚠️ Google_API_KEY missing in .env")
    
    source_type = st.radio("Select Input Type:", ("Website URL", "YouTube URL", "PDF Document"))

    url_input = ""
    uploaded_file = None

    if source_type == "Website URL":
        url_input = st.text_input("Enter Website URL")
    elif source_type == "YouTube URL":
        url_input = st.text_input("Enter YouTube Video URL")
    elif source_type == "PDF Document":
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    process_btn = st.button("Process Data")

# --- HELPER FUNCTIONS ---
# ... (The rest of your functions: get_vectorstore_from_url, get_vectorstore_from_pdf, get_rag_chain remain exactly the same)

def get_vectorstore_from_url(url, source_type):
    """Loads text from Website or YouTube and returns a vector store."""
    try:
        if source_type == "Website URL":
            loader = WebBaseLoader(url)
        elif source_type == "YouTube URL":
            # 🔧 FIX: Check for multiple languages and translation
            loader = YoutubeLoader.from_youtube_url(
                url,
                add_video_info=False,  # Keep False to avoid Pytube errors
                language=["en", "hi", "bn", "es", "ur"],  # Prioritize languages (English, Hindi, Bengali, etc.)
                translation="en"  # Optional: Translate non-English transcripts to English
            )
        
        docs = loader.load()
        
        # Check if documents were actually loaded
        if not docs:
            st.error("No content found. The video might not have captions.")
            return None
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(docs)
        
        # Create Embeddings & Vector Store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        return vectorstore
        
    except Exception as e:
        st.error(f"Error processing URL: {e}")
        return None

def get_vectorstore_from_pdf(pdf_file):
    """Loads text from PDF and returns a vector store."""
    try:
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(pdf_file.getvalue())
            temp_pdf_path = temp_pdf.name

        loader = PyPDFLoader(temp_pdf_path)
        docs = loader.load()
        
        # Clean up temp file
        os.remove(temp_pdf_path)

        # Split text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(docs)

        # Create Embeddings & Vector Store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        return vectorstore

    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None

def get_rag_chain(vectorstore):
    """Creates the Conversational RAG Chain (LangChain 0.2+ compatible)."""
    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", google_api_key=api_key)

    retriever = vectorstore.as_retriever()

    # Prompt for answering with context
    prompt_answer = ChatPromptTemplate.from_messages([
        ("system", 
        "You are a helpful assistant. Answer clearly and concisely using the context below. "
        "If the answer isn't in the context, say you don't know.\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    # 🔧 CRITICAL FIXES
    get_query = RunnableLambda(lambda x: x["input"])
    get_chat_history = RunnableLambda(lambda x: x["chat_history"])

    rag_chain = (
        {
            "context": get_query | retriever,
            "input": get_query,
            "chat_history": get_chat_history,
        }
        | prompt_answer
        | llm
        | StrOutputParser()
    )

    return rag_chain

# --- UI STYLING ---
st.markdown(
    """
    <style>
        body { background-color:rgb(1, 1, 1); color: white; font-family: 'Arial', sans-serif; }
        .stApp { background-color:rgb(1, 1, 1); }
        .fixed-top-left {
            position: fixed; top: 10px; left: 10px; font-size: 14px; font-weight: bold;
            color: white; background-color:rgb(0, 0, 0); padding: 5px 10px;
            border-radius: 5px; z-index: 1000;
        }
        .fixed-bottom {
            position: fixed; bottom: 10px; left: 50%; transform: translateX(-50%);
            font-size: 12px; color: white; background-color:rgb(0, 0, 0);
            padding: 5px 10px; border-radius: 5px; z-index: 1000;
        }
        .message-box { padding: 15px; border-radius: 10px; margin: 10px 0; }
        .user-message { background-color:rgb(153, 150, 242); text-align: right; border: 1px solid #ffffff; color: black; }
        .ai-message { background-color: #ffffff; text-align: left; color: black; border: 1px solid #ffffff; }
        .stTextInput input { color: white; border: 1px solid #444; }
    </style>
    <div class="fixed-top-left">AI Chatbot (LangSmith Active)</div>
    <div class="fixed-bottom">Made with ❤️ from Sohan</div>
    """,
    unsafe_allow_html=True
)

# --- MAIN APP LOGIC ---

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# PROCESS DATA
if process_btn:
    if not api_key:
        st.warning("Please enter your Google API Key first.")
    else:
        with st.spinner("Processing..."):
            if source_type == "PDF Document" and uploaded_file:
                st.session_state.vectorstore = get_vectorstore_from_pdf(uploaded_file)
                st.success("PDF Processed!")
            elif source_type in ["Website URL", "YouTube URL"] and url_input:
                st.session_state.vectorstore = get_vectorstore_from_url(url_input, source_type)
                st.success(f"{source_type} Processed!")
            else:
                st.warning("Please provide a valid input.")

# CHAT INTERFACE
if st.session_state.vectorstore:
    # Display chat history
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("User"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.markdown(message.content)

    # User Input
    user_query = st.chat_input("Ask a question about the content...")
    
    if user_query:
        # Display user message immediately
        with st.chat_message("User"):
            st.markdown(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))

        # Generate Response
        with st.chat_message("AI"):
            rag_chain = get_rag_chain(st.session_state.vectorstore)
            
            response = rag_chain.invoke({
                "input": user_query,
                "chat_history": st.session_state.chat_history
            })
            
            answer = response
            st.markdown(answer)
        
        st.session_state.chat_history.append(AIMessage(content=answer))

else:
    st.info("Please process a document, website, or video to start chatting.")