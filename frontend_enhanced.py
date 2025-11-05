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

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@400;500;600;700&display=swap');

    /* Root Variables - Professional Color Palette */
    :root {
        --primary-color: #4F46E5;
        --primary-dark: #4338CA;
        --primary-light: #818CF8;
        --secondary-color: #06B6D4;
        --secondary-dark: #0891B2;
        --success-color: #10B981;
        --warning-color: #F59E0B;
        --error-color: #EF4444;
        --info-color: #3B82F6;
        --background-color: #F8FAFC;
        --card-bg: #FFFFFF;
        --text-primary: #1E293B;
        --text-secondary: #64748B;
        --border-color: #E2E8F0;
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        --border-radius: 12px;
        --border-radius-lg: 16px;
        --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        color: var(--text-primary);
        background-color: var(--background-color);
    }

    /* Hide Streamlit Branding */
    #MainMenu, header, footer {
        visibility: hidden;
    }

    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }

    /* Animated Gradient Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        background-size: 200% 200%;
        animation: gradientShift 8s ease infinite;
        padding: 2.5rem 3rem;
        border-radius: var(--border-radius-lg);
        margin-bottom: 2.5rem;
        color: white;
        text-align: center;
        box-shadow: var(--shadow-xl);
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
        margin: 0;
        font-size: 3rem;
        font-weight: 800;
        font-family: 'Poppins', sans-serif;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        position: relative;
        z-index: 1;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        margin: 0.75rem 0 0 0;
        font-size: 1.25rem;
        opacity: 0.95;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }

    /* Enhanced Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
        border-right: 1px solid var(--border-color);
        box-shadow: var(--shadow-lg);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding: 2rem 1.5rem;
    }

    /* Section Headers in Sidebar */
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: var(--primary-color);
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--primary-light);
    }

    /* Enhanced Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: var(--border-radius);
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
        transition: var(--transition);
        box-shadow: var(--shadow-md);
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-xl);
        background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary-color) 100%);
    }

    .stButton > button:active {
        transform: translateY(0);
        box-shadow: var(--shadow-md);
    }

    /* Primary Button Variant */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        font-size: 1.1rem;
        padding: 1rem 2rem;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        50% { box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
    }

    /* Input Fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select {
        border: 2px solid var(--border-color);
        border-radius: var(--border-radius);
        padding: 0.75rem 1rem;
        font-size: 1rem;
        transition: var(--transition);
        background: var(--card-bg);
        color: var(--text-primary);
    }

    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stSelectbox > div > div > select:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
        outline: none;
    }

    /* Enhanced Cards */
    .card {
        background: var(--card-bg);
        border-radius: var(--border-radius-lg);
        padding: 1.5rem;
        box-shadow: var(--shadow-md);
        transition: var(--transition);
        border: 1px solid var(--border-color);
        margin-bottom: 1.5rem;
    }

    .card:hover {
        box-shadow: var(--shadow-xl);
        transform: translateY(-2px);
    }

    /* Chat Messages */
    .chat-message {
        background: var(--card-bg);
        padding: 1.25rem;
        border-radius: var(--border-radius);
        box-shadow: var(--shadow-sm);
        margin-bottom: 1rem;
        border-left: 4px solid transparent;
        transition: var(--transition);
        animation: slideIn 0.3s ease-out;
    }

    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    .user-message {
        border-left-color: var(--primary-color);
        background: linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 100%);
    }

    .assistant-message {
        border-left-color: var(--secondary-color);
        background: linear-gradient(135deg, #ECFEFF 0%, #CFFAFE 100%);
    }

    /* Metrics Cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, var(--card-bg) 0%, #F8FAFC 100%);
        padding: 1rem;
        border-radius: var(--border-radius);
        box-shadow: var(--shadow-md);
        border: 1px solid var(--border-color);
        transition: var(--transition);
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
    }

    [data-testid="stMetric"] label {
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--primary-color);
    }

    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%);
        border: 2px dashed var(--primary-color);
        border-radius: var(--border-radius-lg);
        padding: 2rem;
        transition: var(--transition);
    }

    [data-testid="stFileUploader"]:hover {
        background: linear-gradient(135deg, #E0F2FE 0%, #BAE6FD 100%);
        border-color: var(--primary-dark);
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, var(--card-bg) 0%, #F8FAFC 100%);
        border-radius: var(--border-radius);
        padding: 1rem 1.25rem;
        font-weight: 600;
        color: var(--text-primary);
        border: 1px solid var(--border-color);
        transition: var(--transition);
    }

    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
        box-shadow: var(--shadow-md);
    }

    /* Checkbox & Radio */
    .stCheckbox, .stRadio {
        padding: 0.5rem 0;
    }

    .stCheckbox label, .stRadio label {
        font-weight: 500;
        color: var(--text-primary);
    }

    /* Slider */
    .stSlider > div > div > div {
        background: var(--primary-light);
    }

    .stSlider > div > div > div > div {
        background: var(--primary-color);
    }

    /* Status Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.375rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 600;
        margin: 0.25rem;
    }

    .status-online {
        background: #D1FAE5;
        color: #065F46;
    }

    .status-processing {
        background: #FEF3C7;
        color: #92400E;
    }

    .status-offline {
        background: #FEE2E2;
        color: #991B1B;
    }

    /* Loading Spinner */
    .stSpinner > div {
        border-color: var(--primary-light);
        border-top-color: var(--primary-color);
    }

    /* Alert Boxes */
    .stAlert {
        border-radius: var(--border-radius);
        border: none;
        box-shadow: var(--shadow-md);
    }

    /* Success Alert */
    [data-baseweb="notification"][kind="success"] {
        background: linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%);
        border-left: 4px solid var(--success-color);
    }

    /* Error Alert */
    [data-baseweb="notification"][kind="error"] {
        background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%);
        border-left: 4px solid var(--error-color);
    }

    /* Warning Alert */
    [data-baseweb="notification"][kind="warning"] {
        background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
        border-left: 4px solid var(--warning-color);
    }

    /* Info Alert */
    [data-baseweb="notification"][kind="info"] {
        background: linear-gradient(135deg, #DBEAFE 0%, #BFDBFE 100%);
        border-left: 4px solid var(--info-color);
    }

    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, var(--border-color) 50%, transparent 100%);
    }

    /* Columns */
    .stColumns > div {
        padding: 0 0.75rem;
    }

    /* Document List */
    .document-item {
        background: var(--card-bg);
        padding: 0.75rem 1rem;
        border-radius: var(--border-radius);
        margin: 0.5rem 0;
        border-left: 3px solid var(--secondary-color);
        box-shadow: var(--shadow-sm);
        transition: var(--transition);
        display: flex;
        align-items: center;
    }

    .document-item:hover {
        box-shadow: var(--shadow-md);
        transform: translateX(4px);
    }

    /* Footer */
    .footer {
        text-align: center;
        color: var(--text-secondary);
        padding: 2rem 1rem;
        margin-top: 3rem;
        border-top: 2px solid var(--border-color);
        background: linear-gradient(180deg, transparent 0%, #F8FAFC 100%);
    }

    .footer strong {
        color: var(--primary-color);
        font-weight: 700;
    }

    /* Tooltip */
    [data-testid="stTooltipIcon"] {
        color: var(--primary-color);
    }

    /* Session Info Card */
    .session-info {
        background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
        padding: 1rem;
        border-radius: var(--border-radius);
        border-left: 4px solid var(--warning-color);
        margin: 1rem 0;
        font-size: 0.875rem;
        box-shadow: var(--shadow-sm);
    }

    /* Feature Badge */
    .feature-badge {
        display: inline-block;
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 0.25rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }

    ::-webkit-scrollbar-track {
        background: var(--background-color);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--primary-light) 0%, var(--primary-color) 100%);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, var(--primary-color) 0%, var(--primary-dark) 100%);
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .main-header p {
            font-size: 1rem;
        }
        
        .main .block-container {
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
