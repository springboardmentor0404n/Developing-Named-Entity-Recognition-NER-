import streamlit as st
import os
import tempfile
import hashlib
import pickle
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import html
import uuid
import concurrent.futures

# --- 1. OCR (pytesseract/pdf2image) IMPORTS ---
from langchain_core.documents import Document

# --- Core LangChain Imports ---
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS

# --- LLM and Embeddings Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# --- Document Loader Imports ---
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    JSONLoader,
    UnstructuredExcelLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Load API Key from .env file ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found in .env file!")
    st.stop()

# --- FAISS Cache Directory ---
CACHE_DIR = Path("faiss_cache")
CACHE_DIR.mkdir(exist_ok=True)

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 5

def truncate_text(text, max_length=90):
    """Trim text for compact UI previews."""
    text = text.strip()
    if len(text) <= max_length:
        return text
    return text[: max_length - 1].rstrip() + "…"


def create_chat_session(title=None, file_hash=None, file_name=None):
    if not title:
        title = file_name or "New chat"
    session = {
        "id": str(uuid.uuid4()),
        "title": title,
        "file_hash": file_hash,
        "file_name": file_name,
        "messages": [],
        "created_at": datetime.now().isoformat(),
    }
    st.session_state.chat_sessions.append(session)
    return session


def get_session(session_id):
    for session in st.session_state.chat_sessions:
        if session["id"] == session_id:
            return session
    return None


def ensure_active_session():
    session_id = st.session_state.get("active_session_id")
    session = get_session(session_id) if session_id else None
    if session is None:
        session = create_chat_session(
            file_hash=st.session_state.get("file_hash"),
            file_name=st.session_state.get("file_name"),
        )
        st.session_state.active_session_id = session["id"]
        st.session_state.messages = []
    return session


def save_active_session():
    active_id = st.session_state.get("active_session_id")
    if not active_id:
        return
    session = get_session(active_id)
    if session is None:
        return
    session["messages"] = list(st.session_state.messages)
    session["file_hash"] = st.session_state.file_hash
    session["file_name"] = st.session_state.file_name

    if session["messages"]:
        first_user_msg = next((m["content"] for m in session["messages"] if m["role"] == "user"), None)
        if first_user_msg:
            session["title"] = truncate_text(first_user_msg, 40)


def load_session(session_id):
    save_active_session()
    if session_id == st.session_state.get("active_session_id"):
        return
    session = get_session(session_id)
    if session is None:
        return
    
    # Clear the file uploader so it doesn't override the loaded session
    st.session_state.uploader_key = f"document_uploader_{uuid.uuid4()}"
    
    st.session_state.active_session_id = session_id
    st.session_state.messages = list(session["messages"])

    st.session_state.file_hash = session.get("file_hash")
    st.session_state.file_name = session.get("file_name")
    st.session_state.view = "chat"
    st.session_state.retriever = None
    st.session_state.llm = None
    st.session_state.qa_prompt = None
    st.rerun()


def clear_active_chat():
    st.session_state.messages = []
    save_active_session()
    st.rerun()


def start_new_chat(file_hash=None, file_name=None, rerun=True):
    save_active_session()
    session = create_chat_session(
        file_hash=file_hash or st.session_state.get("file_hash"),
        file_name=file_name or st.session_state.get("file_name"),
    )
    st.session_state.active_session_id = session["id"]
    st.session_state.messages = []
    st.session_state.file_hash = session["file_hash"]
    st.session_state.file_name = session["file_name"]
    st.session_state.view = "chat"
    if rerun:
        st.rerun()


# --- Helper Functions ---
def compute_file_hash(file_bytes):
    """Compute SHA256 hash of file content for caching."""
    return hashlib.sha256(file_bytes).hexdigest()

def get_cache_path(file_hash):
    """Get the cache file path for a given file hash."""
    return CACHE_DIR / f"{file_hash}.faiss"

def save_vectorstore(vectorstore, file_hash):
    """Save FAISS vectorstore to disk."""
    cache_path = get_cache_path(file_hash)
    vectorstore.save_local(str(cache_path))
    # Save metadata
    metadata_path = CACHE_DIR / f"{file_hash}.meta"
    with open(metadata_path, 'wb') as f:
        pickle.dump({"created_at": datetime.now().isoformat()}, f)

def load_vectorstore(file_hash, embeddings):
    """Load FAISS vectorstore from disk if it exists."""
    cache_path = get_cache_path(file_hash)
    if cache_path.exists():
        try:
            vectorstore = FAISS.load_local(
                str(cache_path), 
                embeddings,
                allow_dangerous_deserialization=True
            )
            return vectorstore
        except Exception as e:
            st.warning(f"Failed to load cached vectorstore: {e}")
            return None
    return None

# --- 2. UPDATED: load_document (with better error handling) ---
def load_document(file_path):
    """
    Loads a document, using PyMuPDF for fast PDF processing and parallel OCR for images.
    """
    _, file_extension = os.path.splitext(file_path)
    
    if file_extension == ".pdf":
        try:
            import fitz  # PyMuPDF
            from PIL import Image
            import pytesseract
            import io
        except ImportError as import_err:
            st.error("❌ Required libraries not installed.")
            st.code("pip install PyMuPDF Pillow pytesseract", language="bash")
            return []

        try:
            doc = fitz.open(file_path)
        except Exception as e:
            st.error(f"❌ Error opening PDF: {str(e)}")
            return []
        
        final_docs = []
        images_to_ocr = []
        
        # 1. First pass: Extract text where possible
        with st.spinner("📄 Reading PDF pages..."):
            for i in range(len(doc)):
                page = doc[i]
                text = page.get_text()
                
                # Heuristic: If text is very short, assume it's an image/scan
                if len(text.strip()) < 50: 
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img_bytes = pix.tobytes("png")
                    images_to_ocr.append((i, img_bytes))
                else:
                    final_docs.append(Document(page_content=text, metadata={"source": file_path, "page": i+1}))
            
            doc.close()

        # 2. Second pass: Parallel OCR for image pages
        if images_to_ocr:
            st.info(f"🔍 OCR required for {len(images_to_ocr)} pages. Processing in background...")
            progress_bar = st.progress(0)
            
            def ocr_worker(args):
                idx, img_data = args
                try:
                    img = Image.open(io.BytesIO(img_data))
                    text = pytesseract.image_to_string(img)
                    return idx, text
                except Exception:
                    return idx, ""

            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = list(executor.map(ocr_worker, images_to_ocr))
                
            for idx, (page_num, text) in enumerate(results):
                if text.strip():
                    final_docs.append(Document(page_content=text, metadata={"source": file_path, "page": page_num+1}))
                progress_bar.progress((idx + 1) / len(results))
            
            progress_bar.empty()

        # Sort documents by page number
        final_docs.sort(key=lambda x: x.metadata["page"])
        
        if not final_docs:
             st.error("❌ No text could be extracted from this PDF.")
             
        return final_docs

    else:
        try:
            if file_extension == ".docx":
                return Docx2txtLoader(file_path).load()
            elif file_extension == ".txt":
                return TextLoader(file_path).load()
            elif file_extension == ".json":
                return JSONLoader(file_path=file_path, jq_schema='.. | select(type == "string")').load()
            elif file_extension == ".xlsx":
                return UnstructuredExcelLoader(file_path, mode="elements").load()
            elif file_extension == ".md":
                return UnstructuredMarkdownLoader(file_path).load()
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
        except Exception as e:
            st.error(f"❌ Error loading {file_extension} file: {str(e)}")
            return []


# --- 3. OPTIMIZED: get_rag_components with Caching ---
def get_rag_components(uploaded_file=None, api_key=None, chunk_size=1000, chunk_overlap=200, top_k=5, file_hash=None):
    """Return retriever, llm, prompt triple, using cache when possible."""

    if uploaded_file is None and file_hash is None:
        raise ValueError("Provide an uploaded_file or an existing file_hash.")

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_hash = compute_file_hash(file_bytes)
        file_name = uploaded_file.name
    else:
        file_bytes = None
        file_name = "cached document"

    # Initialize embeddings with error handling for event loop issues
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=api_key
        )
    except RuntimeError as e:
        if "event loop" in str(e).lower():
            # Fallback: create embeddings without async
            import nest_asyncio
            nest_asyncio.apply()
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001", 
                google_api_key=api_key
            )
        else:
            raise
    
    # Try to load from cache
    vector_store = load_vectorstore(file_hash, embeddings)

    if vector_store is not None:
        st.success(f"✅ Loaded `{file_name}` from cache (instant load!)")
    else:
        if uploaded_file is None:
            raise RuntimeError("Cached vector store missing. Please re-upload the document to rebuild it.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_file_path = tmp_file.name

        try:
            with st.spinner(f"📄 Processing `{uploaded_file.name}` for the first time..."):
                # Load, split, and embed
                docs = load_document(tmp_file_path)
                if not docs:
                    raise ValueError("Document loading failed or returned no content (check OCR fallback).")

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size, 
                    chunk_overlap=chunk_overlap
                )
                chunks = text_splitter.split_documents(docs)
                
                st.info(f"📊 Created {len(chunks)} chunks from document")
                
                # Create FAISS vectorstore
                vector_store = FAISS.from_documents(chunks, embeddings)
                
                # Save to cache
                save_vectorstore(vector_store, file_hash)
                st.success(f"✅ Processed and cached `{uploaded_file.name}` successfully!")
        
        except Exception as e:
            raise RuntimeError(f"Error processing file: {e}")
        finally:
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

    # --- Create and return the components ---
    retriever = vector_store.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": top_k}
    )
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp", 
        temperature=0.3, 
        google_api_key=api_key
    )
    
    # --- RAG PROMPT WITH CONVERSATION HISTORY ---
    system_template = """
You are an expert financial analyst and document assistant. Your task is to answer questions based on the provided context and conversation history.

**Rules:**
1. Analyze the text in the <context> section carefully.
2. Provide clear, concise, and accurate answers.
3. When extracting financial data (revenue, profit, metrics), quote exact figures with surrounding context.
4. If the answer is not in the context, state: "I'm sorry, but that information is not available in the provided document."
5. Use conversation history to maintain context across questions.
6. Do not make assumptions or use outside knowledge.
7. Be conversational and helpful while staying grounded in the document.

<context>
{context}
</context>
"""
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
    return retriever, llm, qa_prompt, file_hash


# --- Streamlit App ---

st.set_page_config(
    page_title="AI Document Assistant", 
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# Custom CSS for a sleek minimal UI
st.markdown("""
<style>
    :root {
        --bg-light: #f4f5f7;
        --bg-panel: #ffffff;
        --border: #e3e6ed;
        --text-primary: #1e1f23;
        --text-muted: #5f6270;
        --accent: #5b5bd6;
        --shadow: 0px 8px 30px rgba(15, 23, 42, 0.08);
    }

    body, .stApp {
        background: var(--bg-light);
        color: var(--text-primary);
        font-family: "Inter", "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont;
        letter-spacing: 0.01em;
    }

    [data-testid="stSidebar"] {
        background: var(--bg-panel);
        border-right: 1px solid var(--border);
        color: var(--text-primary);
        box-shadow: inset -1px 0 0 var(--border);
    }

    .stChatMessage {
        background: var(--bg-panel);
        border: 1px solid transparent;
        border-radius: 16px;
        padding: 1.1rem 1.35rem;
        margin: 0.75rem 0;
        box-shadow: var(--shadow);
        position: relative;
        overflow: hidden;
        animation: floatUp 0.4s ease;
    }

    .stChatMessage .stMarkdown p {
        color: var(--text-muted);
        line-height: 1.65;
    }

    div[data-testid="chat-message-user"] .stChatMessage {
        background: linear-gradient(135deg, rgba(91,91,214,0.15), rgba(144,97,249,0.1));
        border-left: 4px solid #5b5bd6;
    }

    div[data-testid="chat-message-assistant"] .stChatMessage {
        background: var(--bg-panel);
        border-left: 4px solid #09b58c;
    }

    div[data-testid="chat-message-assistant"] .stMarkdown p strong {
        color: #0f172a;
    }

    .stButton>button {
        background: var(--bg-panel);
        color: var(--text-primary);
        border: 1px solid var(--border);
        border-radius: 999px;
        padding: 0.5rem 1.2rem;
        font-weight: 500;
        transition: all 0.2s ease;
        box-shadow: var(--shadow);
    }

    .stButton>button:hover {
        border-color: var(--accent);
        color: var(--accent);
        transform: translateY(-1px);
    }

    [data-testid="stFileUploader"],
    [data-testid="stMetric"],
    .stAlert {
        background: var(--bg-panel);
        border: 1px solid transparent;
        border-radius: 14px;
        padding: 1rem;
        color: var(--text-primary);
        box-shadow: var(--shadow);
    }

    .stAlert p {
        color: var(--text-muted);
    }

    h1, h2, h3 {
        color: var(--text-primary);
        font-weight: 600;
        letter-spacing: -0.02em;
    }

    .main .block-container {
        background: transparent;
        padding: 1.5rem 2.5rem 4rem;
    }

    .stExpander {
        border-radius: 14px;
        border: 1px solid var(--border);
        background: var(--bg-panel);
        box-shadow: var(--shadow);
    }

    .stSlider > div > div {
        color: var(--text-muted);
    }

    .stSlider [aria-valuenow]::-webkit-slider-runnable-track {
        background: linear-gradient(90deg, var(--accent), #9f7aea);
        height: 5px;
        border-radius: 999px;
    }

    .hero-card {
        position: relative;
        background: radial-gradient(circle at top right, rgba(255,255,255,0.15), transparent 55%),
                    linear-gradient(130deg, #5b5bd6, #7f53ac, #647dee);
        padding: 2.75rem;
        border-radius: 28px;
        color: #fff;
        overflow: hidden;
        box-shadow: var(--shadow);
        animation: gradientShift 8s ease-in-out infinite;
    }

    .hero-card::after {
        content: "";
        position: absolute;
        inset: 0;
        background: radial-gradient(circle at 20% 20%, rgba(255,255,255,0.25), transparent 60%);
        opacity: 0.5;
        pointer-events: none;
    }

    .hero-text {
        position: relative;
        z-index: 1;
        max-width: 540px;
    }

    .hero-text h1 {
        font-size: 2.4rem;
        margin-bottom: 0.5rem;
        color: #fff;
    }

    .hero-text p {
        color: rgba(255,255,255,0.9);
        font-size: 1.05rem;
        margin-bottom: 1.4rem;
    }

    .hero-actions {
        display: flex;
        gap: 0.8rem;
        flex-wrap: wrap;
    }

    .hero-step {
        background: rgba(255,255,255,0.12);
        padding: 0.65rem 1rem;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.25);
        font-weight: 500;
    }

    .hero-cta {
        position: relative;
        z-index: 1;
        margin-top: 2rem;
        padding: 1.25rem 1.5rem;
        border-radius: 18px;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.25);
        max-width: 360px;
    }

    .hero-badges {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 1rem;
    }

    .hero-badges span {
        background: rgba(255,255,255,0.18);
        padding: 0.45rem 0.9rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }

    .feature-card {
        background: var(--bg-panel);
        border-radius: 18px;
        padding: 1.4rem;
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
        margin-bottom: 1rem;
        transition: transform 0.2s ease;
    }

    .feature-card:hover {
        transform: translateY(-4px);
        border-color: var(--accent);
    }

    .feature-card h3 {
        margin-bottom: 0.35rem;
    }

    .quick-list {
        list-style: none;
        padding-left: 0;
        margin: 0.75rem 0 0;
    }

    .quick-list li {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.35rem 0;
        color: rgba(255,255,255,0.95);
    }

    .quick-list li::before {
        content: "✦";
        font-size: 0.75rem;
        color: #ffe08a;
    }

    .chat-history {
        background: var(--bg-panel);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 0.9rem;
        max-height: 340px;
        overflow-y: auto;
        box-shadow: var(--shadow);
        animation: fadeIn 0.5s ease;
    }

    .chat-history::-webkit-scrollbar {
        width: 4px;
    }

    .chat-history::-webkit-scrollbar-thumb {
        background: var(--border);
        border-radius: 999px;
    }

    .chat-history-item {
        display: flex;
        gap: 0.6rem;
        padding: 0.65rem 0.7rem;
        border-radius: 12px;
        border: 1px solid transparent;
        background: var(--bg-light);
        transition: transform 0.2s ease, border 0.2s ease;
    }

    .chat-history-item:hover {
        transform: translateX(4px);
        border-color: var(--accent);
    }

    .chat-history-item span {
        font-size: 1.1rem;
    }

    .chat-history-empty {
        color: var(--text-muted);
        text-align: center;
        padding: 1rem 0;
    }

    .glow {
        position: absolute;
        width: 180px;
        height: 180px;
        top: -40px;
        right: -40px;
        background: radial-gradient(circle, rgba(255,255,255,0.45), transparent 70%);
        filter: blur(4px);
        animation: floatUp 6s ease-in-out infinite;
    }

    @keyframes gradientShift {
        0% { filter: hue-rotate(0deg); }
        100% { filter: hue-rotate(25deg); }
    }

    @keyframes floatUp {
        0% { transform: translateY(0px) scale(1); opacity: 0.7; }
        50% { transform: translateY(-8px) scale(1.02); opacity: 1; }
        100% { transform: translateY(0px) scale(1); opacity: 0.7; }
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "file_hash" not in st.session_state:
    st.session_state.file_hash = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "qa_prompt" not in st.session_state:
    st.session_state.qa_prompt = None
if "view" not in st.session_state:
    st.session_state.view = "home"
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = f"document_uploader_{uuid.uuid4()}"
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = []
if "active_session_id" not in st.session_state:
    initial_session = create_chat_session()
    st.session_state.active_session_id = initial_session["id"]
if "file_name" not in st.session_state:
    st.session_state.file_name = None

ensure_active_session()

def reset_to_home():
    save_active_session()
    st.session_state.uploader_key = f"document_uploader_{uuid.uuid4()}"
    st.session_state.retriever = None
    st.session_state.llm = None
    st.session_state.qa_prompt = None
    st.session_state.file_hash = None
    st.session_state.file_name = None
    st.session_state.messages = []
    st.session_state.view = "home"
    st.rerun()

# --- Sidebar ---
with st.sidebar:
    st.markdown("## 📂 Workspace")
    st.caption("Securely upload reports, manuals, or knowledge bases for instant answers.")

    st.markdown("### 📁 Upload Document")
    supported_types = ["pdf", "docx", "txt", "json", "xlsx", "md"]
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=supported_types,
        help="PDF, Word, Text, JSON, Excel, Markdown",
        key=st.session_state.uploader_key
    )

    if uploaded_file and st.session_state.view == "home":
        st.session_state.view = "chat"

    st.caption("Optimized retrieval settings are applied automatically for best results.")

    st.markdown("### 🕑 Chat History")
    sessions_container = st.container()
    with sessions_container:
        st.markdown("<div class='chat-history'>", unsafe_allow_html=True)
        if st.session_state.chat_sessions:
            for session in reversed(st.session_state.chat_sessions[-8:]):
                is_active = session["id"] == st.session_state.active_session_id
                title = truncate_text(session.get("title") or "Untitled", 40)
                meta = session.get("file_name") or "No file"
                label = f"{'●' if is_active else '○'} {title} — {meta}"
                if st.button(label, key=f"session_{session['id']}", help=meta, use_container_width=True):
                    if not is_active:
                        load_session(session["id"])
        else:
            st.markdown(
                "<div class='chat-history-empty'>No conversations yet. Start one above!</div>",
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

    if st.button("➕ New chat", use_container_width=True):
        start_new_chat()

    st.markdown("### ⚡ Quick Actions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            clear_active_chat()

    with col2:
        if st.button("🔄 Clear Cache", use_container_width=True):
            for cache_file in CACHE_DIR.glob("*"):
                cache_file.unlink()
            st.rerun()

chunk_size = DEFAULT_CHUNK_SIZE
chunk_overlap = DEFAULT_CHUNK_OVERLAP
top_k = DEFAULT_TOP_K

# --- Main Content ---
header_col, home_col = st.columns([10, 1])
with header_col:
    st.title("🤖 AI Document Assistant")
    st.markdown("Context-aware answers grounded in your PDFs, spreadsheets, and policies.")

with home_col:
    st.markdown("&nbsp;")
    if st.button("🏠", key="home_nav", help="Back to home", use_container_width=True):
        reset_to_home()

has_cached_file = st.session_state.file_hash is not None
show_chat = (
    st.session_state.view != "home"
    and (uploaded_file is not None or has_cached_file)
)

if show_chat:
    try:
        current_file_hash = st.session_state.file_hash
        new_file_uploaded = False

        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            current_file_hash = compute_file_hash(file_bytes)
            new_file_uploaded = (st.session_state.file_hash != current_file_hash)
            st.session_state.file_name = uploaded_file.name

        needs_components = (
            new_file_uploaded
            or st.session_state.retriever is None
            or st.session_state.llm is None
            or st.session_state.qa_prompt is None
        )

        if needs_components:
            with st.spinner("⏳ Processing document..."):
                retriever, llm, qa_prompt, file_hash = get_rag_components(
                    api_key=GOOGLE_API_KEY,
                    uploaded_file=uploaded_file if uploaded_file is not None else None,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    top_k=top_k,
                    file_hash=current_file_hash
                )

            st.session_state.retriever = retriever
            st.session_state.llm = llm
            st.session_state.qa_prompt = qa_prompt
            st.session_state.file_hash = file_hash

            if new_file_uploaded:
                st.session_state.messages = []
                st.success(f"✅ Ready! Ask me anything about `{uploaded_file.name}`")
            save_active_session()

    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.stop()

    # --- Chat Interface ---
    st.markdown("---")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input
    if prompt := st.chat_input("💭 Ask a question about your document..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate AI response with streaming
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            try:
                # Retrieve relevant documents
                with st.spinner("🔍 Searching document..."):
                    retrieved_docs = st.session_state.retriever.invoke(prompt)
                    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                
                # Build chat history for context
                chat_history = []
                for msg in st.session_state.messages[:-1]:  # Exclude current message
                    if msg["role"] == "user":
                        chat_history.append(HumanMessage(content=msg["content"]))
                    else:
                        chat_history.append(AIMessage(content=msg["content"]))
                
                # Format messages
                messages = st.session_state.qa_prompt.format_messages(
                    context=context,
                    chat_history=chat_history,
                    input=prompt
                )
                
                # Stream response
                full_response = ""
                for chunk in st.session_state.llm.stream(messages):
                    full_response += chunk.content
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
                # Add to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                save_active_session()
                
            except Exception as e:
                st.error(f"❌ Error generating response: {e}")
                
else:
    st.markdown("""
    <div class="hero-card">
        <div class="hero-text">
            <div class="hero-badges">
                <span>Gemini Powered</span>
                <span>FAISS Cached</span>
            </div>
            <h1>Bring your documents to life.</h1>
            <p>Drop in annual reports, technical manuals, policy decks, or research logs.
               Get trustworthy answers, grounded in citations, in seconds.</p>
            <div class="hero-actions">
                <div class="hero-step">1 · Upload a file</div>
                <div class="hero-step">2 · Ask in natural language</div>
                <div class="hero-step">3 · Explore sourced insights</div>
            </div>
        </div>
        <div class="hero-cta">
            <div class="glow"></div>
            <p><strong>Need inspiration?</strong></p>
            <ul class="quick-list">
                <li>"Summarize FY25 performance drivers."</li>
                <li>"Compare pricing clauses between sections."</li>
                <li>"List every risk mitigation strategy mentioned."</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Why teams love this workspace")
    features = [
        ("🎯", "Precision Retrieval", "Semantic search surfaces only the most relevant passages for each query."),
        ("⚡", "Instant Re-runs", "Upload once—subsequent sessions load from FAISS cache immediately."),
        ("🧠", "Conversation Memory", "Follow-up questions automatically reuse prior context for continuity."),
    ]
    cols = st.columns(len(features))
    for col, (icon, title, desc) in zip(cols, features):
        with col:
            st.markdown(
                f"""
                <div class='feature-card'>
                    <h3>{icon} {title}</h3>
                    <p>{desc}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("#### Quick start checklist")
    checklist = [
        "Use PDFs with selectable text for best accuracy.",
        "Stay under 50 pages for lightning-fast processing.",
        "Batch upload variations to compare policies or reports.",
    ]
    for item in checklist:
        st.markdown(f"- ✅ {item}")

    st.markdown(
        """<p style='color: var(--text-muted); margin-top: 1rem;'>Tip: Drag & drop a file into the sidebar uploader to get started instantly.</p>""",
        unsafe_allow_html=True
    )