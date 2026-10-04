# 🤖 AI Assistant Pro — RAG & Memory

A production-ready AI chatbot with **Retrieval-Augmented Generation (RAG)**, **conversation memory**, **PDF processing**, and **smart answer routing**.

[![Live Demo](https://img.shields.io/badge/Live-Demo-blue?style=for-the-badge&logo=render&logoColor=white)](https://agenticai-chatbot-using-rag-1.onrender.com/)

---

## 🏗️ Project Structure

```
AgenticAI_Chatbot_Using_RAG/
├── backend/
│   ├── __init__.py
│   ├── config.py       ← env vars, constants, allowed model names
│   ├── models.py       ← Pydantic request / response schemas
│   ├── agent.py        ← LangGraph workflow, RAG manager, memory manager
│   └── main.py         ← FastAPI application & all HTTP endpoints
├── frontend/
│   ├── __init__.py
│   └── app.py          ← Streamlit UI
├── .devcontainer/
│   └── devcontainer.json
├── .env.example        ← copy to .env and fill in API keys
├── .gitignore
├── Procfile            ← Render / Railway deploy command
├── requirements.txt
└── README.md
```

---

## 🌟 Features

| Feature | Details |
|---------|---------|
| **RAG** | Upload PDFs → Cohere embeddings → FAISS index → Cohere reranking → page-cited answers |
| **Memory** | Per-session chat history, last 10 turns carried into every request |
| **Smart routing** | Automatically chooses document RAG, plain LLM, or Tavily web search |
| **Multi-provider** | Groq (Llama 3.3 70B) or OpenAI (GPT-4o-mini) |
| **REST API** | Clean FastAPI backend with Pydantic-validated endpoints |

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/BrainstormerAI/AgenticAI_Chatbot_Using_RAG.git
cd AgenticAI_Chatbot_Using_RAG
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

Required keys:

| Variable | Get it from |
|----------|------------|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com/) |
| `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com/) |
| `TAVILY_API_KEY` | [tavily.com](https://tavily.com/) |
| `COHERE_API_KEY` | [cohere.ai](https://cohere.ai/) |

### 3. Run locally

Open **two terminals** from the project root:

**Terminal 1 — Backend:**
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 — Frontend:**
```bash
streamlit run frontend/app.py
```

Then open [http://localhost:8501](http://localhost:8501).

---

## ⚙️ Configuration

All configuration lives in [`backend/config.py`](backend/config.py) and is driven by environment variables (see `.env.example`).

| Variable | Default | Description |
|----------|---------|-------------|
| `BACKEND_PORT` | `8000` | Port the FastAPI server listens on |
| `BACKEND_URL` | `http://localhost:8000` | URL the Streamlit frontend calls |
| `ALLOWED_ORIGINS` | `http://localhost:8501` | CORS allowed origins (comma-separated) |

For production, set `BACKEND_URL` in the frontend's environment and `ALLOWED_ORIGINS` in the backend's environment to your deployed URLs.

---

## 🔌 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | API info |
| `GET` | `/health` | Health check |
| `POST` | `/chat` | Send a message, get an AI response |
| `POST` | `/upload-pdf` | Upload a PDF for RAG indexing |
| `POST` | `/chat-history` | Retrieve session history |
| `POST` | `/clear-history` | Clear session history |
| `POST` | `/user-documents` | List indexed documents for a user |

### Example `/chat` request

```json
{
  "model_name": "llama-3.3-70b-versatile",
  "model_provider": "Groq",
  "system_prompt": "You are a helpful assistant.",
  "messages": ["What does the document say about pricing?"],
  "allow_search": true,
  "user_id": "alice",
  "session_id": "session-abc",
  "similarity_threshold": 0.5
}
```

---

## 🏛️ Architecture

```
┌─────────────────┐   HTTP   ┌──────────────────┐   Python   ┌─────────────────────┐
│  Streamlit UI   │─────────►│  FastAPI Backend  │───────────►│  LangGraph Agent    │
│  frontend/app.py│◄─────────│  backend/main.py  │◄───────────│  backend/agent.py   │
└─────────────────┘          └──────────────────┘            └──────────┬──────────┘
                                                                         │
                              ┌──────────────────────────────────────────┤
                              │                                          │
                    ┌─────────▼────────┐                      ┌─────────▼─────────┐
                    │  MemoryManager   │                      │    RAGManager      │
                    │  (session dict)  │                      │  FAISS + Cohere    │
                    └──────────────────┘                      └───────────────────┘
```

### LangGraph workflow

```
router ──► rag ──► agent ──► END
       └──────────────────►
```

- **router** — computes FAISS similarity score; routes to `rag` if score > threshold, else straight to `agent`.
- **rag** — builds a numbered-excerpt prompt with page citations from the top-k reranked documents.
- **agent** — calls the LLM directly (RAG path) or wraps it in a Tavily ReAct loop (search path).

---

## 🛠️ Extending

### Add a new LLM provider

In [`backend/agent.py`](backend/agent.py), add a branch in `get_response()`:

```python
elif provider == "Anthropic":
    from langchain_anthropic import ChatAnthropic
    llm = ChatAnthropic(model=llm_id, temperature=LLM_TEMPERATURE)
```

Then add the model name to `ALLOWED_MODELS` in [`backend/config.py`](backend/config.py).

### Persistent storage

Replace `MemoryManager` with a Redis or SQLite backend; replace the in-memory FAISS store with a saved index (call `store.save_local()` / `FAISS.load_local()`).

---

## 🐛 Troubleshooting

| Symptom | Fix |
|---------|-----|
| Frontend shows "Cannot reach the backend" | Make sure `uvicorn backend.main:app` is running and `BACKEND_URL` is set correctly |
| "COHERE_API_KEY not set" in logs | Add the key to `.env`; RAG and reranking are disabled without it |
| PDF upload returns 422 | PDF may be scanned/image-only; text extraction requires selectable text |
| Slow responses | Switch to Groq; disable web search for pure document queries |

---

## 📄 License

MIT
