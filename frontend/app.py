"""
Streamlit frontend — AI Chatbot Pro (RAG & Memory).

Start with:  streamlit run frontend/app.py
Backend URL is read from the BACKEND_URL environment variable
(defaults to http://localhost:8000 for local development).
"""

import os
import uuid
from datetime import datetime

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

BACKEND_URL: str = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

MODEL_NAMES_GROQ = ["llama-3.3-70b-versatile", "llama3-70b-8192"]
MODEL_NAMES_OPENAI = ["gpt-4o-mini"]

# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Assistant Pro — RAG & Memory",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ────────────────────────────────────────────────────

def _init_state() -> None:
    defaults = {
        "session_id": str(uuid.uuid4()),
        "user_id": "default",
        "chat_history": [],
        "uploaded_documents": [],
        "current_response": None,
        "show_chat_history": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

_init_state()

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .stApp { background: linear-gradient(135deg, #f0f4f8 0%, #e2e8f0 100%); }

    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #f1f5f9; border-radius: 10px; }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        border-radius: 10px;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 50%, #2563eb 100%);
        background-size: 200% 200%;
        animation: gradientShift 8s ease infinite;
        padding: 2.5rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(59, 130, 246, 0.3);
    }
    @keyframes gradientShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .main-header h1 {
        color: white; font-size: 2.8rem; font-weight: 700; margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2); font-family: 'Poppins', sans-serif;
    }
    .main-header p { color: rgba(255,255,255,0.95); font-size: 1.15rem; margin-top: 0.5rem; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border-right: 2px solid #e2e8f0;
        padding: 1.5rem 1rem;
    }
    .sidebar-section { margin-bottom: 2rem; padding-bottom: 1.5rem; border-bottom: 2px solid #e2e8f0; }
    .sidebar-section h3 {
        color: #3b82f6; font-weight: 600; font-size: 1.15rem; margin-bottom: 1.25rem;
        font-family: 'Poppins', sans-serif;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white !important; border: none; border-radius: 8px;
        padding: 0.45rem 0.75rem !important; font-weight: 600 !important;
        font-size: 0.75rem !important; width: 100%; height: 36px !important;
        box-shadow: 0 2px 8px rgba(59,130,246,0.3); transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59,130,246,0.4);
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        font-size: 0.95rem !important; height: 46px !important;
        box-shadow: 0 4px 15px rgba(16,185,129,0.4);
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
    }

    /* Inputs */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border: 2px solid #e2e8f0; border-radius: 8px;
        font-size: 0.9rem; transition: all 0.3s ease; background: white;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59,130,246,0.1);
        outline: none;
    }

    /* Select / radio */
    .stSelectbox > div > div {
        border-radius: 8px; border: 2px solid #e2e8f0; transition: all 0.3s ease; background: white;
    }
    .stRadio > div { background: white; padding: 1rem; border-radius: 10px; border: 2px solid #e2e8f0; }
    .stRadio [role="radiogroup"] { display: flex !important; flex-direction: row !important; gap: 0.75rem !important; }
    .stRadio [role="radiogroup"] > label {
        background: #ffffff !important; padding: 0.75rem 1rem !important;
        border-radius: 8px !important; border: 2px solid #cbd5e1 !important;
        flex: 1 !important; display: flex !important; align-items: center !important; transition: all 0.3s ease !important;
    }
    .stRadio [role="radiogroup"] > label:hover { border-color: #3b82f6 !important; background: #f8fafc !important; }
    .stRadio [role="radiogroup"] > label[data-checked="true"] { border-color: #3b82f6 !important; background: #eff6ff !important; }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: white; border: 2px dashed #3b82f6; border-radius: 10px;
        padding: 1.5rem; transition: all 0.3s ease;
    }
    [data-testid="stFileUploader"]:hover { border-color: #2563eb; background: #eff6ff; }

    /* Document list */
    .document-item {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 0.6rem 0.8rem; margin: 0.4rem 0; border-radius: 8px;
        border-left: 3px solid #3b82f6; box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        font-size: 0.85rem; transition: all 0.3s ease;
    }
    .document-item:hover { transform: translateX(3px); box-shadow: 0 3px 10px rgba(59,130,246,0.15); }

    /* Chat boxes */
    .chat-response-box {
        background: white; border: 2px solid #e2e8f0; border-radius: 12px;
        padding: 1.5rem; margin-top: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08); animation: fadeIn 0.5s ease;
    }
    .user-message {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        padding: 1rem; border-radius: 10px; margin-bottom: 1rem; border-left: 4px solid #3b82f6;
    }
    .user-message strong { color: #1e40af; display: block; margin-bottom: 0.5rem; }
    .assistant-message {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        padding: 1rem; border-radius: 10px; border-left: 4px solid #10b981;
    }
    .assistant-message strong { color: #047857; display: block; margin-bottom: 0.5rem; }

    /* Columns */
    [data-testid="column"] {
        background: white; padding: 1.5rem; border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }

    /* Footer */
    .footer {
        text-align: center; padding: 2rem;
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 15px; margin-top: 2rem;
    }
    .feature-badge {
        display: inline-block;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white; padding: 0.4rem 1rem; border-radius: 20px;
        margin: 0.25rem; font-size: 0.85rem; font-weight: 600;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    @media (max-width: 768px) {
        .main-header h1 { font-size: 2rem; }
        .stButton > button { font-size: 0.7rem !important; }
    }
</style>
""",
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown(
    """
<div class="main-header">
    <h1>🤖 AI Assistant Pro</h1>
    <p>Advanced RAG-powered AI with Memory &amp; Document Intelligence</p>
</div>
""",
    unsafe_allow_html=True,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _api(method: str, path: str, **kwargs) -> requests.Response | None:
    """Make an API call; return the Response or None on connection error."""
    url = f"{BACKEND_URL}{path}"
    try:
        return requests.request(method, url, timeout=120, **kwargs)
    except requests.exceptions.ConnectionError:
        st.error(
            f"❌ Cannot reach the backend at **{BACKEND_URL}**. "
            "Make sure `uvicorn backend.main:app` is running."
        )
        return None
    except Exception as exc:
        st.error(f"❌ Request error: {exc}")
        return None


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    # User & Session
    st.markdown('<div class="sidebar-section"><h3>👤 User &amp; Session</h3></div>', unsafe_allow_html=True)
    user_id_input = st.text_input("User ID:", value=st.session_state.user_id)
    if user_id_input != st.session_state.user_id:
        st.session_state.user_id = user_id_input

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 New Session"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.chat_history = []
            st.session_state.current_response = None
            st.session_state.show_chat_history = False
            st.rerun()
    with col2:
        if st.button("🗑️ Clear History"):
            resp = _api("POST", "/clear-history", json={"session_id": st.session_state.session_id})
            if resp and resp.status_code == 200:
                st.session_state.chat_history = []
                st.session_state.current_response = None
                st.session_state.show_chat_history = False
                st.success("✅ History cleared!")
            elif resp:
                st.error("❌ Failed to clear history.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Model Settings
    st.markdown('<div class="sidebar-section"><h3>🧠 Model Settings</h3></div>', unsafe_allow_html=True)
    provider = st.radio("Provider:", ("Groq", "OpenAI"), index=0)
    if provider == "Groq":
        selected_model = st.selectbox("Model:", MODEL_NAMES_GROQ)
    else:
        selected_model = st.selectbox("Model:", MODEL_NAMES_OPENAI)

    st.markdown("<br>", unsafe_allow_html=True)

    # Agent Settings
    st.markdown('<div class="sidebar-section"><h3>🎯 Agent Settings</h3></div>', unsafe_allow_html=True)
    allow_web_search = st.checkbox("🔍 Allow Web Search", value=True)
    similarity_threshold = st.slider(
        "📊 RAG Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Higher = stricter document matching before falling back to LLM.",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Document Management
    st.markdown('<div class="sidebar-section"><h3>📚 Document Management</h3></div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file is not None:
        if st.button("📤 Process PDF"):
            with st.spinner("Processing PDF…"):
                resp = _api(
                    "POST",
                    "/upload-pdf",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                    data={"user_id": st.session_state.user_id},
                )
                if resp and resp.status_code == 200:
                    result = resp.json()
                    st.success(f"✅ {result['message']}")
                    if uploaded_file.name not in st.session_state.uploaded_documents:
                        st.session_state.uploaded_documents.append(uploaded_file.name)
                elif resp:
                    st.error(f"❌ {resp.json().get('detail', 'Upload failed.')}")

    if st.button("📋 Refresh Documents"):
        resp = _api("POST", "/user-documents", json={"user_id": st.session_state.user_id})
        if resp and resp.status_code == 200:
            st.session_state.uploaded_documents = resp.json().get("documents", [])

    if st.session_state.uploaded_documents:
        st.markdown("**📄 Your Documents:**")
        for doc in st.session_state.uploaded_documents:
            st.markdown(f'<div class="document-item">📄 {doc}</div>', unsafe_allow_html=True)
    else:
        st.info("No documents uploaded yet.")

# ── Main area ─────────────────────────────────────────────────────────────────

col_chat, col_history = st.columns([2, 1])

with col_chat:
    st.markdown("### 💬 Chat Interface")

    system_prompt = st.text_area(
        "🎭 System Prompt:",
        height=100,
        placeholder="You are a helpful AI assistant…",
        value=(
            "You are a helpful AI assistant. Answer questions using uploaded "
            "documents when relevant, otherwise use your general knowledge. "
            "Be clear about your sources."
        ),
    )

    user_query = st.text_area(
        "💭 Your message:",
        height=120,
        placeholder="Ask anything — I can search your documents or the web…",
    )

    if st.button("🚀 Ask Agent!", type="primary"):
        if not user_query.strip():
            st.warning("⚠️ Please enter a message first.")
        else:
            with st.spinner("🤔 Thinking…"):
                payload = {
                    "model_name": selected_model,
                    "model_provider": provider,
                    "system_prompt": system_prompt,
                    "messages": [user_query],
                    "allow_search": allow_web_search,
                    "user_id": st.session_state.user_id,
                    "session_id": st.session_state.session_id,
                    "similarity_threshold": similarity_threshold,
                }
                resp = _api("POST", "/chat", json=payload)
                if resp and resp.status_code == 200:
                    data = resp.json()
                    st.session_state.current_response = {
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "user": user_query,
                        "assistant": data["response"],
                    }
                    st.rerun()
                elif resp:
                    st.error(f"❌ {resp.json().get('detail', 'Unknown error from backend.')}")

    # Show the latest response
    if st.session_state.current_response:
        r = st.session_state.current_response
        st.markdown('<div class="chat-response-box">', unsafe_allow_html=True)
        st.markdown(
            f"""
<div class="user-message">
    <strong>👤 You ({r['timestamp']}):</strong>
    <p>{r['user']}</p>
</div>
<div class="assistant-message">
    <strong>🤖 Assistant:</strong>
    <p>{r['assistant']}</p>
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

with col_history:
    st.markdown("### 📜 Chat History")

    if st.button("🔄 Load History"):
        resp = _api("POST", "/chat-history", json={"session_id": st.session_state.session_id})
        if resp and resp.status_code == 200:
            raw = resp.json().get("history", [])
            pairs = []
            for i in range(0, len(raw) - 1, 2):
                u, a = raw[i], raw[i + 1]
                if u.get("type") == "human" and a.get("type") == "ai":
                    pairs.append({
                        "timestamp": u.get("timestamp", ""),
                        "user": u.get("content", ""),
                        "assistant": a.get("content", ""),
                    })
            st.session_state.chat_history = pairs
            st.session_state.show_chat_history = True
            st.rerun()

    if st.session_state.show_chat_history and st.session_state.chat_history:
        for i, chat in enumerate(reversed(st.session_state.chat_history[-10:])):
            label = f"💬 Chat {len(st.session_state.chat_history) - i} — {chat['timestamp']}"
            with st.expander(label, expanded=(i == 0)):
                st.markdown("**👤 You:**")
                st.write(chat["user"])
                st.markdown("**🤖 Assistant:**")
                st.write(chat["assistant"])
    else:
        st.info("💡 Click 'Load History' to view past conversations.")

    with st.expander("💡 Tips & Features"):
        st.markdown(
            """
**🧠 Memory** — context carries across turns in the same session.
**📚 RAG** — upload a PDF; questions about it use document excerpts.
**🔍 Web Search** — enabled via Tavily when no document matches.
**🤖 Providers** — Groq (fast) or OpenAI GPT-4o-mini.

**Tips:**
- Upload PDFs before asking about them.
- Adjust the similarity threshold if RAG isn't triggering.
- Use "New Session" to start a fresh conversation.
"""
        )

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    """
<div class="footer">
    <p><strong>AI Assistant Pro</strong> | Powered by LangGraph + RAG</p>
    <p><small>Streamlit · FastAPI · Cohere · FAISS · Groq / OpenAI</small></p>
    <p style="margin-top:1rem;">
        <span class="feature-badge">📚 RAG</span>
        <span class="feature-badge">🧠 Memory</span>
        <span class="feature-badge">🔍 Web Search</span>
        <span class="feature-badge">🤖 Multi-Model</span>
    </p>
</div>
""",
    unsafe_allow_html=True,
)
