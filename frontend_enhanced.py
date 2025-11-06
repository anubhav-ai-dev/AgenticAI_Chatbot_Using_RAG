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
if 'current_response' not in st.session_state:
    st.session_state.current_response = None
if 'show_chat_history' not in st.session_state:
    st.session_state.show_chat_history = False

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
    
    /* Main Container */
    .stApp {
        background: linear-gradient(135deg, #f0f4f8 0%, #e2e8f0 100%);
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        border-radius: 10px;
    }
    
    /* Light Blue Header */
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
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        font-family: 'Poppins', sans-serif;
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.95);
        font-size: 1.15rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border-right: 2px solid #e2e8f0;
        padding: 1.5rem 1rem;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        padding: 0.5rem;
    }
    
    /* Section Headers */
    .sidebar-section {
        margin-bottom: 2rem;
        padding-bottom: 1.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    .sidebar-section h3 {
        color: #3b82f6;
        font-weight: 600;
        font-size: 1.15rem;
        margin-bottom: 1.25rem;
        font-family: 'Poppins', sans-serif;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .sidebar-section h4 {
        color: #1e293b;
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 0.75rem;
        margin-top: 0;
    }
    
    /* Input Labels */
    .stTextInput > label,
    .stTextArea > label,
    .stSelectbox > label,
    .stSlider > label {
        color: #475569 !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Text Inputs */
    .stTextInput > div > div > input {
        border: 2px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.65rem 0.85rem;
        font-size: 0.9rem;
        transition: all 0.3s ease;
        background: white;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        outline: none;
    }
    
    /* Improved button styling with smaller font and proper alignment */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.45rem 0.75rem !important;
        font-weight: 600 !important;
        font-size: 0.75rem !important;
        line-height: 1.2 !important;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
        cursor: pointer;
        width: 100%;
        height: 36px !important;
        min-height: 36px !important;
        max-height: 36px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 0.3rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Primary Button (Ask Agent) */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
        font-size: 0.95rem !important;
        padding: 0.7rem 1.5rem !important;
        height: 46px !important;
        min-height: 46px !important;
        max-height: 46px !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5);
    }
    
    /* Radio buttons styled to display inline with proper spacing */
    .stRadio > div {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    
    .stRadio > label {
        font-weight: 600 !important;
        color: #475569 !important;
        font-size: 0.9rem !important;
        margin-bottom: 0.75rem !important;
        display: block !important;
    }
    
    /* Improved radio button alignment - both options in same row with equal width */
    .stRadio [role="radiogroup"] {
        display: flex !important;
        flex-direction: row !important;
        gap: 0.75rem !important;
        flex-wrap: nowrap !important;
        align-items: stretch !important;
        width: 100% !important;
    }
    
    .stRadio [role="radiogroup"] > label {
        background: #f8fafc !important;
        padding: 0.7rem 1rem !important;
        border-radius: 8px !important;
        margin: 0 !important;
        border: 2px solid #e2e8f0 !important;
        transition: all 0.3s ease !important;
        cursor: pointer !important;
        flex: 1 1 0 !important;
        min-width: 0 !important;
        max-width: none !important;
        text-align: center !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 0.4rem !important;
        white-space: nowrap !important;
    }
    
    .stRadio [role="radiogroup"] > label:hover {
        background: #eff6ff !important;
        border-color: #3b82f6 !important;
        transform: translateY(-1px);
    }
    
    .stRadio [role="radiogroup"] > label[data-checked="true"] {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%) !important;
        border-color: #3b82f6 !important;
        font-weight: 600 !important;
        color: #1e40af !important;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.2);
    }
    
    .stRadio [role="radiogroup"] > label > div {
        display: flex !important;
        align-items: center !important;
        gap: 0.4rem !important;
        width: 100% !important;
        justify-content: center !important;
    }
    
    .stRadio [role="radiogroup"] > label > div > div:first-child {
        flex-shrink: 0 !important;
    }
    
    .stRadio [role="radiogroup"] > label > div > div:last-child {
        flex-grow: 0 !important;
        white-space: nowrap !important;
    }
    
    /* Select Box */
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        transition: all 0.3s ease;
        background: white;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #3b82f6;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
    }
    
    /* Checkbox */
    .stCheckbox {
        background: white;
        padding: 0.75rem;
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .stCheckbox:hover {
        border-color: #3b82f6;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.1);
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: white;
        border: 2px dashed #3b82f6;
        border-radius: 10px;
        padding: 1.5rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #2563eb;
        background: #eff6ff;
    }
    
    /* Text Area */
    .stTextArea > div > div > textarea {
        border: 2px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.75rem;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        background: white;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        outline: none;
    }
    
    /* Document List */
    .document-item {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 0.6rem 0.8rem;
        margin: 0.4rem 0;
        border-radius: 8px;
        border-left: 3px solid #3b82f6;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        font-size: 0.85rem;
    }
    
    .document-item:hover {
        transform: translateX(3px);
        box-shadow: 0 3px 10px rgba(59, 130, 246, 0.15);
    }
    
    /* Chat Response Box */
    .chat-response-box {
        background: white;
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        animation: fadeIn 0.5s ease;
    }
    
    .user-message {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #3b82f6;
    }
    
    .user-message strong {
        color: #1e40af;
        display: block;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #10b981;
    }
    
    .assistant-message strong {
        color: #047857;
        display: block;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #3b82f6;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.1);
    }
    
    /* Alerts */
    .stSuccess {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 4px solid #10b981;
        border-radius: 8px;
    }
    
    .stError {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 4px solid #ef4444;
        border-radius: 8px;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
    }
    
    /* Section Headers in Main Area */
    .section-header {
        color: #1e293b;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        font-family: 'Poppins', sans-serif;
    }
    
    /* Column Styling */
    [data-testid="column"] {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 15px;
        margin-top: 2rem;
        box-shadow: 0 -4px 20px rgba(0,0,0,0.05);
    }
    
    .footer p {
        margin: 0.5rem 0;
        color: #475569;
    }
    
    .feature-badge {
        display: inline-block;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        margin: 0.25rem;
        font-size: 0.85rem;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .main-header h1 { font-size: 2rem; }
        .main-header p { font-size: 1rem; }
        
        .stButton > button {
            font-size: 0.7rem !important;
            padding: 0.4rem 0.6rem !important;
        }
        
        .stRadio [role="radiogroup"] > label {
            font-size: 0.8rem !important;
            padding: 0.55rem 1rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🤖 AI Assistant Pro</h1>
    <p>Advanced RAG-powered AI with Memory & Document Intelligence</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.markdown('<div class="sidebar-section"><h3>⚙️ Configuration</h3></div>', unsafe_allow_html=True)
    
    # User & Session
    st.markdown('<h4>👤 User & Session</h4>', unsafe_allow_html=True)
    user_id = st.text_input("User ID:", value=st.session_state.user_id, key="user_id_input")
    if user_id != st.session_state.user_id:
        st.session_state.user_id = user_id
    
    # Buttons in same row
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 NEW SESSION"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.chat_history = []
            st.session_state.current_response = None
            st.session_state.show_chat_history = False
            st.rerun()
    
    with col2:
        if st.button("🗑️ CLEAR HISTORY"):
            try:
                response = requests.post(f"{API_URL}/clear-history",
                                       json={"session_id": st.session_state.session_id})
                if response.status_code == 200:
                    st.session_state.chat_history = []
                    st.session_state.current_response = None
                    st.session_state.show_chat_history = False
                    st.success("✅ History cleared!")
                else:
                    st.error("❌ Failed to clear history")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)
    
    # Model Settings
    st.markdown('<div class="sidebar-section"><h3>🧠 Model Settings</h3></div>', unsafe_allow_html=True)
    
    MODEL_NAMES_GROQ = ["llama-3.3-70b-versatile", "llama3-70b-8192"]
    MODEL_NAMES_OPENAI = ["gpt-4o-mini"]
    
    provider = st.radio("Select Provider:", ("Groq", "OpenAI"), index=0, key="provider_radio")
    
    if provider == "Groq":
        selected_model = st.selectbox("Select Groq Model:", MODEL_NAMES_GROQ)
    else:
        selected_model = st.selectbox("Select OpenAI Model:", MODEL_NAMES_OPENAI)
    
    st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)
    
    # Agent Settings
    st.markdown('<div class="sidebar-section"><h3>🎯 Agent Settings</h3></div>', unsafe_allow_html=True)
    
    allow_web_search = st.checkbox("🔍 Allow Web Search", value=True)
    similarity_threshold = st.slider(
        "📊 RAG Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Higher values = more strict document matching"
    )
    
    st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)
    
    # Document Management
    st.markdown('<div class="sidebar-section"><h3>📚 Document Management</h3></div>', unsafe_allow_html=True)
    
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
                    st.error(f"❌ Error: {str(e)}")
    
    if st.button("📋 Refresh Documents"):
        try:
            response = requests.post(f"{API_URL}/user-documents",
                                   json={"user_id": st.session_state.user_id})
            if response.status_code == 200:
                result = response.json()
                st.session_state.uploaded_documents = result.get('documents', [])
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    if st.session_state.uploaded_documents:
        st.markdown("**📄 Your Documents:**")
        for doc in st.session_state.uploaded_documents:
            st.markdown(f'<div class="document-item">📄 {doc}</div>', unsafe_allow_html=True)
    else:
        st.info("*No documents uploaded yet*")

# Main Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="section-header">💬 Chat Interface</div>', unsafe_allow_html=True)
    
    system_prompt = st.text_area(
        "🎭 Define your AI Agent:",
        height=100,
        placeholder="You are a helpful AI assistant...",
        value="You are a helpful AI assistant. You can answer questions using uploaded documents or general knowledge. Be clear about your sources."
    )
    
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
                            # Store ONLY current response, don't auto-load history
                            st.session_state.current_response = {
                                "timestamp": datetime.now().strftime("%H:%M:%S"),
                                "user": user_query,
                                "assistant": data['response']
                            }
                            st.rerun()
                    else:
                        st.error("❌ Error: Could not get response from backend.")
                except Exception as e:
                    st.error(f"❌ Exception: {str(e)}")
        else:
            st.warning("⚠️ Please enter a query!")
    
    # Display Current Response ONLY
    if st.session_state.current_response:
        st.markdown('<div class="chat-response-box">', unsafe_allow_html=True)
        st.markdown(f'''
        <div class="user-message">
            <strong>👤 You:</strong>
            <p>{st.session_state.current_response['user']}</p>
        </div>
        <div class="assistant-message">
            <strong>🤖 Assistant:</strong>
            <p>{st.session_state.current_response['assistant']}</p>
        </div>
        ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-header">📜 Chat History</div>', unsafe_allow_html=True)
    
    # Manual Load History Button
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
                                "assistant": ai_msg.get('content', '')
                            })
                
                st.session_state.show_chat_history = True
                st.rerun()
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    # Display chat history ONLY if manually loaded
    if st.session_state.show_chat_history and st.session_state.chat_history:
        for i, chat in enumerate(reversed(st.session_state.chat_history[-10:])):
            with st.expander(f"💬 Chat {len(st.session_state.chat_history) - i} - {chat['timestamp']}", 
                           expanded=(i == 0)):
                st.markdown("**👤 You:**")
                st.write(chat['user'])
                st.markdown("**🤖 Assistant:**")
                st.write(chat['assistant'])
    else:
        st.info("💡 Click 'Load History' to view past conversations")
    
    # Tips
    with st.expander("💡 Tips & Features"):
        st.markdown("""
        ### 🌟 Features:
        
        **🧠 Memory & Context**
        - Maintains conversation history
        - Context-aware responses
        
        **📚 RAG System**
        - Upload PDFs for Q&A
        - Smart document retrieval
        
        **🔍 Web Search**
        - Falls back to web search
        - Current information access
        
        **🤖 Multiple Providers**
        - Groq: Fast Llama models
        - OpenAI: GPT-4o-mini
        
        ### 💡 Best Practices:
        
        - Upload relevant PDFs first
        - Use specific questions
        - Adjust similarity threshold
        - Enable web search for general queries
        """)

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p><strong>AI Assistant Pro</strong> | Powered by RAG Technology</p>
    <p><small>Built with Streamlit • Professional UI/UX Design</small></p>
    <p style="margin-top: 1rem;">
        <span class="feature-badge">📚 RAG Enabled</span>
        <span class="feature-badge">🧠 Memory System</span>
        <span class="feature-badge">🔍 Web Search</span>
        <span class="feature-badge">🤖 Multi-Model</span>
    </p>
</div>
""", unsafe_allow_html=True)
