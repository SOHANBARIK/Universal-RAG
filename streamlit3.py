import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# --- 1. CONFIGURATION & SETUP ---
load_dotenv()

model = os.getenv("MODEL_NAME")
# LangSmith Configuration (Optional)
if os.getenv("LANGCHAIN_KEY"):
    os.environ["LANGCHAIN_KEY"] = os.getenv("LANGCHAIN_KEY")
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("PROJECT_NAME")

# Page Config
st.set_page_config(page_title="Universal RAG Chatbot", layout="wide", initial_sidebar_state="expanded", page_icon="🤖")
st.title("🤖 Universal RAG: Chat with PDF, Web, YouTube, or Search")

# --- 2. IMPORTS ---
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from youtube_transcript_api import TranscriptsDisabled, NoTranscriptFound

# Loaders & Tools
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, YoutubeLoader
from langchain_community.tools.tavily_search import TavilySearchResults

# --- SIDEBAR: SETTINGS ---
with st.sidebar:
    st.header(" If you are using a YouTube URL - Please select that video which has captions enabled.")
    api_key = os.getenv("Google_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")

    st.divider()
    
    # Data Source Selection
    st.header("📂 Data Source")
    source_type = st.radio("Select Input Type:", ("Web Search", "Website URL", "YouTube URL", "PDF Document"))

    # DYNAMIC INPUT FIELDS
    user_input = ""
    uploaded_file = None

    if source_type == "Web Search":
        st.info("🔍 Google-Style Search Mode")
        user_input = st.text_input("Enter Search Query", placeholder="e.g., Who won the 2024 Cricket World Cup?")
    
    elif source_type == "Website URL":
        user_input = st.text_input("Enter Website URL", placeholder="https://example.com")
    
    elif source_type == "YouTube URL":
        st.info("Ensure the video has captions (CC) enabled.")
        user_input = st.text_input("Enter YouTube Video URL", placeholder="https://youtube.com/watch?v=...")
    
    elif source_type == "PDF Document":
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    process_btn = st.button("🚀 Process Data")

# --- HELPER FUNCTIONS ---

def get_vectorstore_from_url(input_text, source_type):
    """Loads text from Website, YouTube, or Tavily Search."""
    try:
        docs = []
        
        # A. TAVILY SEARCH (Strict Text Input)
        if source_type == "Web Search":
            if not os.environ.get("TAVILY_API_KEY"):
                st.error("❌ Tavily API Key is missing!")
                return None
                
            st.info(f"🔍 Searching the web for: '{input_text}'...")
            search = TavilySearchResults(max_results=10)
            results = search.invoke(input_text)
            
            # Tavily returns a list of dictionaries with 'content' and 'url'
            for res in results:
                content = res.get("content", "")
                url = res.get("url", "")
                if content:
                    docs.append(Document(page_content=content, metadata={"source": url}))
            
            if len(docs) > 0:
                st.write(f"✅ Found {len(docs)} results. Adding to knowledge base...")
            else:
                st.warning("No relevant results found on the web.")

        # B. WEBSITE URL
        elif source_type == "Website URL":
            st.info(f"🌐 Scraping: {input_text}")
            loader = WebBaseLoader(input_text)
            docs = loader.load()

        # C. YOUTUBE URL
        elif source_type == "YouTube URL":
            st.info(f"📹 Processing Video: {input_text}")
            try:
                loader = YoutubeLoader.from_youtube_url(
                    input_text, 
                    add_video_info=False, 
                    language=["en", "en-US", "en-IN", "hi", "bn", "es", "ur"], 
                )
                docs = loader.load()
            except (TranscriptsDisabled, NoTranscriptFound):
                st.warning("⚠️ No captions found! This looks like a music video or one without CC.")
                st.info("💡 Tip: Use 'Web Search' mode and search for 'Lyrics of [Song Name]' instead.")
                return None
            except Exception as e:
                st.error(f"YouTube Error: {e}")
                return None

        if not docs:
            st.error("❌ No content found to process.")
            return None

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(docs)

        # Create Embeddings & Vector Store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        return vectorstore

    except Exception as e:
        st.error(f"Error processing data: {e}")
        return None

def get_vectorstore_from_pdf(pdf_file):
    """Loads text from PDF."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(pdf_file.getvalue())
            temp_pdf_path = temp_pdf.name
        
        loader = PyPDFLoader(temp_pdf_path)
        docs = loader.load()
        os.remove(temp_pdf_path)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(docs)
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None

def get_rag_chain(vectorstore):
    """Creates the Chain."""
    # Using specific model name to avoid 404 errors
    llm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key)
    
    retriever = vectorstore.as_retriever()
    
    prompt_answer = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer based strictly on the context below:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    
    get_query = RunnableLambda(lambda x: x["input"])
    get_chat_history = RunnableLambda(lambda x: x["chat_history"])
    
    rag_chain = (
        {"context": get_query | retriever, "input": get_query, "chat_history": get_chat_history}
        | prompt_answer | llm | StrOutputParser()
    )
    return rag_chain

# --- UI STYLING ---
st.markdown("""
    <style>
        body { background-color: #0E1117; color: white; }
        .stTextInput input { color: white; border: 1px solid #444; }
        .fixed-bottom { position: fixed; bottom: 10px; left: 50%; transform: translateX(-50%); font-size: 12px; color: #888; }
    </style>
    <div class="fixed-bottom">Powered by Gemini & Tavily</div>
    """, unsafe_allow_html=True)

# --- MAIN APP LOGIC ---

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# PROCESS INPUT
if process_btn:
    if not api_key:
        st.warning("⚠️ Please provide a Google API Key.")
    else:
        with st.spinner("Processing..."):
            new_vs = None
            
            # Handle PDF
            if source_type == "PDF Document" and uploaded_file:
                new_vs = get_vectorstore_from_pdf(uploaded_file)
            
            # Handle Text Inputs (Search, Website, YouTube)
            elif source_type in ["Web Search", "Website URL", "YouTube URL"] and user_input:
                new_vs = get_vectorstore_from_url(user_input, source_type)
            
            # Success Handler
            if new_vs:
                st.session_state.vectorstore = new_vs
                st.success("Context Loaded Successfully! Chat below. 👇")

# CHAT INTERFACE
if st.session_state.vectorstore:
    # Render Chat History
    for message in st.session_state.chat_history:
        role = "user" if isinstance(message, HumanMessage) else "ai"
        with st.chat_message(role):
            st.markdown(message.content)

    # Chat Input
    user_query = st.chat_input("Ask a question about the context...")
    if user_query:
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        
        with st.chat_message("ai"):
            rag_chain = get_rag_chain(st.session_state.vectorstore)
            response = rag_chain.invoke({
                "input": user_query,
                "chat_history": st.session_state.chat_history
            })
            st.markdown(response)
        st.session_state.chat_history.append(AIMessage(content=response))