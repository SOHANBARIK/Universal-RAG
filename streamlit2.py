import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# --- 1. CONFIGURATION & SETUP ---
load_dotenv()

# LangSmith Configuration (Optional but recommended)
if os.getenv("Langchain_Key"):
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("Langchain_Key")
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "Gemini-RAG-Final"

# Page Config
st.set_page_config(page_title="Universal RAG Chatbot", layout="wide", initial_sidebar_state="expanded", page_icon="🤖")
st.title("🤖 Universal RAG: Chat with PDF, Web, YouTube, or Tavily Search")

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
    
    # 1. Google API Key
    api_key = os.getenv("Google_API_KEY")
    # if not api_key:
    #     api_key = st.text_input("Google API Key", type="password")
    
    # 2. Tavily API Key
    tavily_key = os.getenv("TAVILY_API_KEY")
    # if not tavily_key:
    #     tavily_key = st.text_input("Tavily API Key", type="password")
    #     if tavily_key:
    #         os.environ["TAVILY_API_KEY"] = tavily_key

    st.divider()
    
    # Data Source Selection
    st.header("📂 Data Source")
    source_type = st.radio("Select Input Type:", ("Web Search", "Website URL", "YouTube URL", "PDF Document"))

    url_input = ""
    uploaded_file = None

    if source_type == "Web Search":
        url_input = st.text_input("Enter Search Topic", placeholder="e.g., Latest updates in AI technology")
    elif source_type == "Website URL":
        url_input = st.text_input("Enter Website URL")
    elif source_type == "YouTube URL":
        url_input = st.text_input("Enter YouTube Video URL")
    elif source_type == "PDF Document":
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    elif source_type == "Web Search (Tavily)":
        url_input = st.text_input("Enter Search Topic", placeholder="e.g., Latest features of Gemini 1.5")

    process_btn = st.button("🚀 Process Data")

# --- HELPER FUNCTIONS ---

def get_vectorstore_from_url(url, source_type):
    """Loads text from Website, YouTube, or Tavily Search."""
    try:
        docs = []
        
        # A. TAVILY SEARCH
        if source_type == "Web Search":
            if not os.environ.get("TAVILY_API_KEY"):
                st.error("❌ Tavily API Key is missing!")
                return None
                
            st.info(f"🔍 Searching from web for: '{url}'...")
            search = TavilySearchResults(max_results=5)
            results = search.invoke(url)
            
            # Convert Tavily results directly to LangChain Documents
            # Tavily gives us the 'content' directly, so we don't need to scrape again!
            for res in results:
                content = res.get("content", "")
                source = res.get("url", "")
                if content:
                    docs.append(Document(page_content=content, metadata={"source": source}))
            
            st.write(f"✅ Found {len(docs)} relevant results.")

        # B. WEBSITE URL
        elif source_type == "Website URL":
            loader = WebBaseLoader(url)
            docs = loader.load()

        # C. YOUTUBE URL
        elif source_type == "YouTube URL":
            try:
                # Robust Loader: Checks English, Hindi, Bengali, etc., and auto-translates
                loader = YoutubeLoader.from_youtube_url(
                    url, 
                    add_video_info=False, 
                    language=["en", "en-US", "en-IN", "hi", "bn", "es", "ur"], 
                    # translation="en"
                )
                docs = loader.load()
            except (TranscriptsDisabled, NoTranscriptFound):
                        # 2. HANDLE MUSIC VIDEOS
                        st.warning("⚠️ No captions found! This looks like a music video without CC.")
                        st.info("💡 Pro Tip: Switch to 'Web Search' and search for 'Lyrics of [Song Name]' instead.")
                        return None
                
            except Exception as e:
                st.error(f"YouTube Error: {e}")
                return None

        if not docs:
            st.error("❌ No content found. Please check the URL or Search Query.")
            return None

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(docs)

        # Create Embeddings & Vector Store
        # Using text-embedding-004 to avoid "Limit 0" errors
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
    # Using Gemini 3 Flash Preview for speed and cost
    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", google_api_key=api_key)
    
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
            if source_type == "PDF Document" and uploaded_file:
                new_vs = get_vectorstore_from_pdf(uploaded_file)
            elif source_type in ["Website URL", "YouTube URL", "Web Search (Tavily)"] and url_input:
                new_vs = get_vectorstore_from_url(url_input, source_type)
            
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