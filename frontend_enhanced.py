from dotenv import load_dotenv

load_dotenv()

import streamlit as st
import requests
import json
import uuid
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="AI Assistant Pro - RAG & Memory",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'user_id' not in st.session_state:
    st.session_state.user_id = "default"
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'uploaded_documents' not in st.session_state:
    st.session_state.uploaded_documents = []

# API Configuration
API_URL = "https://agenticai-chatbot-using-rag.onrender.com"

# Enhanced Professional CSS Styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Container Styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Animated Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        background-size: 200% 200%;
        animation: gradientShift 8s ease infinite;
        padding: 2.5rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        font-family: 'Poppins', sans-serif;
        position: relative;
        z-index: 1;
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.95);
        font-size: 1.2rem;
        margin-top: 0.5rem;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
        border-right: 2px solid #e9ecef;
    }
    
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] .stMarkdown h4 {
        color: #667eea;
        font-weight: 600;
        font-family: 'Poppins', sans-serif;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Enhanced Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        cursor: pointer;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Primary Button (Ask Agent) */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        box-shadow: 0 4px 15px rgba(245, 87, 108, 0.4);
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { box-shadow: 0 4px 15px rgba(245, 87, 108, 0.4); }
        50% { box-shadow: 0 6px 25px rgba(245, 87, 108, 0.6); }
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #f5576c 0%, #f093fb 100%);
        box-shadow: 0 6px 25px rgba(245, 87, 108, 0.6);
    }
    
    /* Input Fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border: 2px solid #e9ecef;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        background: white;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        outline: none;
    }
    
    /* Select Box */
    .stSelectbox > div > div {
        border-radius: 12px;
        border: 2px solid #e9ecef;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #667eea;
    }
    
    /* Radio Buttons */
    .stRadio > div {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        border: 2px solid #e9ecef;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Checkbox */
    .stCheckbox {
        background: white;
        padding: 0.75rem;
        border-radius: 10px;
        border: 2px solid #e9ecef;
        transition: all 0.3s ease;
    }
    
    .stCheckbox:hover {
        border-color: #667eea;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.1);
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: white;
        border: 2px dashed #667eea;
        border-radius: 12px;
        padding: 1.5rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #764ba2;
        background: #f8f9fa;
    }
    
    /* Document List Items */
    .document-item {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        font-size: 0.9rem;
    }
    
    .document-item:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
        border-left-color: #764ba2;
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 12px;
        border: 2px solid #e9ecef;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #667eea;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.1);
    }
    
    /* Metrics Cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1rem;
        border-radius: 12px;
        border: 2px solid #e9ecef;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
        border-color: #667eea;
    }
    
    [data-testid="stMetricLabel"] {
        font-weight: 600;
        color: #667eea;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2d3748;
    }
    
    /* Alert Boxes */
    .stAlert {
        border-radius: 12px;
        border: none;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    [data-baseweb="notification"] {
        border-radius: 12px;
    }
    
    /* Success Alert */
    .stSuccess {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 4px solid #28a745;
    }
    
    /* Error Alert */
    .stError {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 4px solid #dc3545;
    }
    
    /* Warning Alert */
    .stWarning {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border-left: 4px solid #ffc107;
    }
    
    /* Info Alert */
    .stInfo {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border-left: 4px solid #17a2b8;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, #667eea 50%, transparent 100%);
    }
    
    /* Footer Styling */
    .footer {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 15px;
        margin-top: 2rem;
        box-shadow: 0 -4px 20px rgba(0,0,0,0.05);
    }
    
    .footer p {
        margin: 0.5rem 0;
        color: #4a5568;
    }
    
    .footer strong {
        color: #667eea;
        font-weight: 700;
    }
    
    /* Feature Badges */
    .feature-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        margin: 0.25rem;
        font-size: 0.85rem;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .feature-badge:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.5);
    }
    
    /* Column Styling */
    [data-testid="column"] {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin: 0.5rem;
    }
    
    /* Markdown Content */
    .stMarkdown {
        color: #2d3748;
    }
    
    .stMarkdown h3 {
        color: #667eea;
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        margin-top: 1.5rem;
    }
    
    .stMarkdown strong {
        color: #764ba2;
    }
    
    /* Code Blocks */
    code {
        background: #f8f9fa;
        padding: 0.2rem 0.5rem;
        border-radius: 6px;
        color: #667eea;
        font-size: 0.9rem;
    }
    
    /* JSON Display */
    .stJson {
        background: #f8f9fa;
        border-radius: 12px;
        border: 2px solid #e9ecef;
        padding: 1rem;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: white;
        border-radius: 12px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Loading Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .element-container {
        animation: fadeIn 0.5s ease;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .main-header p {
            font-size: 1rem;
        }
        
        [data-testid="column"] {
            margin: 0.25rem;
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Professional header
st.markdown("""
<div class="main-header">
    <h1>🤖 AI Assistant Pro</h1>
    <p>Advanced RAG-powered AI with Memory & Document Intelligence</p>
</div>
""", unsafe_allow_html=True)


# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    # User & Session Management
    st.markdown("#### 👤 User & Session")
    user_id = st.text_input("User ID:", value=st.session_state.user_id, key="user_id_input")
    if user_id != st.session_state.user_id:
        st.session_state.user_id = user_id

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 New Session"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.chat_history = []
            st.rerun()

    with col2:
        if st.button("🗑️ Clear History"):
            try:
                response = requests.post(f"{API_URL}/clear-history",
                                         json={"session_id": st.session_state.session_id})
                if response.status_code == 200:
                    st.session_state.chat_history = []
                    st.success("History cleared!")
                else:
                    st.error("Failed to clear history")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    # Model Configuration
    st.markdown("#### 🧠 Model Settings")
    MODEL_NAMES_GROQ = ["llama-3.3-70b-versatile", "llama3-70b-8192"]
    MODEL_NAMES_OPENAI = ["gpt-4o-mini"]

    provider = st.radio("Select Provider:", ("Groq", "OpenAI"))

    if provider == "Groq":
        selected_model = st.selectbox("Select Groq Model:", MODEL_NAMES_GROQ)
    elif provider == "OpenAI":
        selected_model = st.selectbox("Select OpenAI Model:", MODEL_NAMES_OPENAI)

    # Agent Configuration
    st.markdown("#### 🎯 Agent Settings")
    allow_web_search = st.checkbox("🔍 Allow Web Search", value=True)
    similarity_threshold = st.slider(
        "📊 RAG Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Higher values = more strict document matching"
    )

    # Document Management
    st.markdown("#### 📚 Document Management")

    # PDF Upload
    uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])

    if uploaded_file is not None:
        if st.button("📤 Process PDF"):
            with st.spinner("Processing PDF..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    data = {"user_id": st.session_state.user_id}
                    response = requests.post(f"{API_URL}/upload-pdf", files=files, data=data)

                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ {result['message']}")
                        st.session_state.uploaded_documents.append(uploaded_file.name)
                    else:
                        error_detail = response.json().get('detail', 'Unknown error')
                        st.error(f"❌ Upload failed: {error_detail}")
                except Exception as e:
                    st.error(f"❌ Error uploading PDF: {str(e)}")

    if st.button("📋 Refresh Documents"):
        try:
            response = requests.post(f"{API_URL}/user-documents",
                                     json={"user_id": st.session_state.user_id})
            if response.status_code == 200:
                result = response.json()
                st.session_state.uploaded_documents = result.get('documents', [])
        except Exception as e:
            st.error(f"Error fetching documents: {str(e)}")

    if st.session_state.uploaded_documents:
        st.markdown("**📄 Your Documents:**")
        for doc in st.session_state.uploaded_documents:
            st.markdown(f'<div class="document-item">📄 {doc}</div>', unsafe_allow_html=True)
    else:
        st.info("*No documents uploaded yet*")

# Main Chat Interface
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 💬 Chat Interface")

    # System Prompt
    system_prompt = st.text_area(
        "🎭 Define your AI Agent:",
        height=100,
        placeholder="You are a helpful AI assistant with access to uploaded documents and web search...",
        value="You are a helpful AI assistant. You can answer questions using uploaded documents or general knowledge. Be clear about your sources."
    )

    # Chat Input
    user_query = st.text_area(
        "💭 Enter your query:",
        height=120,
        placeholder="Ask anything! I can use your uploaded documents or search the web..."
    )

    if st.button("🚀 Ask Agent!", type="primary"):
        if user_query.strip():
            with st.spinner("🤔 Agent is thinking..."):
                payload = {
                    "model_name": selected_model,
                    "model_provider": provider,
                    "system_prompt": system_prompt,
                    "messages": [user_query],
                    "allow_search": allow_web_search,
                    "user_id": st.session_state.user_id,
                    "session_id": st.session_state.session_id,
                    "similarity_threshold": similarity_threshold
                }

                try:
                    response = requests.post(f"{API_URL}/chat", json=payload)
                    if response.status_code == 200:
                        data = response.json()
                        if "error" in data:
                            st.error(f"❌ {data['error']}")
                        else:
                            st.session_state.chat_history.append({
                                "timestamp": datetime.now().strftime("%H:%M:%S"),
                                "user": user_query,
                                "assistant": data['response'],
                                "session_id": data.get('session_id', st.session_state.session_id)
                            })
                            st.rerun()
                    else:
                        st.error("❌ Error: Could not get response from backend.")
                except Exception as e:
                    st.error(f"❌ Exception: {str(e)}")
        else:
            st.warning("⚠️ Please enter a query!")

with col2:
    st.markdown("### 📜 Chat History")

    if st.button("🔄 Load History"):
        try:
            response = requests.post(f"{API_URL}/chat-history",
                                     json={"session_id": st.session_state.session_id})
            if response.status_code == 200:
                result = response.json()
                history = result.get('history', [])
                st.session_state.chat_history = []

                for i in range(0, len(history), 2):
                    if i + 1 < len(history):
                        user_msg = history[i]
                        ai_msg = history[i + 1]
                        if user_msg.get('type') == 'human' and ai_msg.get('type') == 'ai':
                            st.session_state.chat_history.append({
                                "timestamp": user_msg.get('timestamp', ''),
                                "user": user_msg.get('content', ''),
                                "assistant": ai_msg.get('content', ''),
                                "session_id": st.session_state.session_id
                            })
        except Exception as e:
            st.error(f"Error loading history: {str(e)}")

    if st.session_state.chat_history:
        for i, chat in enumerate(reversed(st.session_state.chat_history[-10:])):
            with st.expander(f"💬 Chat {len(st.session_state.chat_history) - i} - {chat['timestamp']}",
                             expanded=(i == 0)):
                st.markdown("**👤 You:**")
                st.write(chat['user'])
                st.markdown("**🤖 Assistant:**")
                st.write(chat['assistant'])
    else:
        st.info("💡 No chat history yet. Start a conversation!")

    # Session Info
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("👤 User", st.session_state.user_id[:8] + "...")
    with col2:
        st.metric("💬 Chats", len(st.session_state.chat_history))
    with col3:
        st.metric("📚 Docs", len(st.session_state.uploaded_documents))

    # Tips & Debug
    with st.expander("🔧 Advanced Features & Tips"):
        st.markdown("""
        ### 🌟 Features Available:

        **1. 🧠 Memory & Context**
        - Maintains conversation history across sessions
        - Context-aware responses based on previous interactions

        **2. 📚 RAG (Retrieval-Augmented Generation)**
        - Upload PDFs and ask questions about their content
        - Smart document retrieval using Cohere embeddings
        - Automatic reranking for better relevance

        **3. 🎯 Smart Answer Switching**
        - Automatically chooses between document content and general knowledge
        - Adjustable similarity threshold for fine-tuning

        **4. 🔍 Web Search Integration**
        - Falls back to web search for questions outside document scope
        - Powered by Tavily Search API

        **5. 🤖 Multiple AI Providers**
        - Groq: Fast inference with Llama and Mixtral models
        - OpenAI: GPT-4o-mini for high-quality responses

        ### 💡 Tips for Best Results:

        **For Document Questions:**
        - Upload relevant PDFs first
        - Use specific questions about document content
        - Lower similarity threshold (0.3-0.5) for broader matching

        **For General Questions:**
        - Higher similarity threshold (0.7-0.9) to avoid document interference
        - Enable web search for current information

        **For Conversations:**
        - Use the same session ID to maintain context
        - System prompts help define agent behavior
        """)

    if st.checkbox("🐛 Show Debug Info"):
        st.markdown("#### Debug Information")
        st.json({
            "user_id": st.session_state.user_id,
            "session_id": st.session_state.session_id,
            "chat_history_length": len(st.session_state.chat_history),
            "uploaded_documents": st.session_state.uploaded_documents,
            "selected_model": selected_model,
            "provider": provider,
            "similarity_threshold": similarity_threshold,
            "allow_web_search": allow_web_search
        })

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>🤖 <strong>AI Assistant Pro</strong> | Powered by RAG Technology & Advanced Memory Systems</p>
    <p><small>Built with Streamlit • Enhanced with Professional UI/UX Design</small></p>
    <p style="margin-top: 1rem;">
        <span class="feature-badge">RAG Enabled</span>
        <span class="feature-badge">Memory System</span>
        <span class="feature-badge">Web Search</span>
        <span class="feature-badge">Multi-Model</span>
    </p>
</div>
""", unsafe_allow_html=True)
