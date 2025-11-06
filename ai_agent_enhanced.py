from dotenv import load_dotenv

load_dotenv()

import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_cohere import CohereEmbeddings, CohereRerank
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
import cohere

# Initialize Cohere client
cohere_client = cohere.Client(COHERE_API_KEY) if COHERE_API_KEY else None


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    session_id: str
    use_rag: bool
    similarity_threshold: float
    retrieved_docs: List[Document]


class MemoryManager:
    """Manages chat history and session state"""

    def __init__(self):
        self.sessions = {}

    def get_session_history(self, session_id: str) -> List[Dict]:
        return self.sessions.get(session_id, [])

    def add_to_session(self, session_id: str, message: Dict):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append({
            **message,
            "timestamp": datetime.now().isoformat()
        })

    def clear_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]


class RAGManager:
    """Manages document storage and retrieval"""

    def __init__(self):
        # Initialize embeddings with proper model parameter
        if COHERE_API_KEY:
            try:
                self.embeddings = CohereEmbeddings(
                    cohere_api_key=COHERE_API_KEY,
                    model="embed-english-v3.0"  # Specify the model
                )
                self.reranker = CohereRerank(
                    cohere_api_key=COHERE_API_KEY,
                    model="rerank-english-v3.0"  # Specify rerank model
                )
                self.cohere_available = True
            except Exception as e:
                print(f"Warning: Cohere initialization failed: {e}")
                print("Falling back to basic similarity search without Cohere")
                self.embeddings = None
                self.reranker = None
                self.cohere_available = False
        else:
            print("Warning: COHERE_API_KEY not found. RAG features will be limited.")
            self.embeddings = None
            self.reranker = None
            self.cohere_available = False

        self.vector_stores = {}  # user_id -> FAISS store
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

    def process_pdf_content(self, user_id: str, pdf_content: str, filename: str):
        """Process PDF content and store in vector database"""
        if not self.cohere_available:
            print("Warning: Cohere not available. Cannot process PDF for RAG.")
            return False

        try:
            # Split text into chunks
            chunks = self.text_splitter.split_text(pdf_content)

            # Create documents with metadata
            documents = [
                Document(
                    page_content=chunk,
                    metadata={
                        "source": filename,
                        "user_id": user_id,
                        "chunk_id": i,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                for i, chunk in enumerate(chunks)
            ]

            # Create or update vector store for user
            if user_id in self.vector_stores:
                # Add to existing store
                self.vector_stores[user_id].add_documents(documents)
            else:
                # Create new store
                self.vector_stores[user_id] = FAISS.from_documents(
                    documents, self.embeddings
                )

            return True
        except Exception as e:
            print(f"Error processing PDF content: {e}")
            return False

    def retrieve_relevant_docs(self, user_id: str, query: str, k: int = 3) -> List[Document]:
        """Retrieve relevant documents for a query"""
        if user_id not in self.vector_stores:
            return []

        try:
            # Retrieve similar documents
            docs = self.vector_stores[user_id].similarity_search(query, k=k * 2)  # Get more for reranking

            if not docs:
                return []

            # Rerank using Cohere if available
            if self.cohere_available and self.reranker:
                try:
                    doc_texts = [doc.page_content for doc in docs]
                    reranked = self.reranker.rerank(doc_texts, query)

                    # Return top k reranked documents
                    reranked_docs = []
                    for result in reranked[:k]:
                        original_doc = docs[result["index"]]
                        original_doc.metadata["relevance_score"] = result["relevance_score"]
                        reranked_docs.append(original_doc)

                    return reranked_docs

                except Exception as e:
                    print(f"Reranking failed: {e}")
                    return docs[:k]
            else:
                return docs[:k]
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []

    def calculate_similarity_score(self, user_id: str, query: str) -> float:
        """Calculate similarity score to determine if RAG should be used"""
        if user_id not in self.vector_stores:
            return 0.0

        try:
            docs = self.retrieve_relevant_docs(user_id, query, k=1)
            if docs and "relevance_score" in docs[0].metadata:
                return docs[0].metadata.get('relevance_score', 0.0)

            # Fallback: use vector similarity
            similar_docs = self.vector_stores[user_id].similarity_search_with_score(query, k=1)
            if similar_docs:
                return 1.0 - similar_docs[0][1]  # Convert distance to similarity

            return 0.0
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0


# Global instances
memory_manager = MemoryManager()
rag_manager = RAGManager()


def router_node(state: AgentState) -> AgentState:
    """Determines whether to use RAG, LLM, or Search based on query"""
    messages = state["messages"]
    user_query = messages[-1].content if messages else ""
    user_id = state.get("user_id", "default")

    # Only use RAG if Cohere is available and user has documents
    if rag_manager.cohere_available and user_id in rag_manager.vector_stores:
        # Calculate similarity with stored documents
        similarity_score = rag_manager.calculate_similarity_score(user_id, user_query)
        threshold = state.get("similarity_threshold", 0.5)

        print(f"Similarity score: {similarity_score}, Threshold: {threshold}")

        if similarity_score > threshold:
            # Use RAG
            retrieved_docs = rag_manager.retrieve_relevant_docs(user_id, user_query)
            state["retrieved_docs"] = retrieved_docs
            state["use_rag"] = True
            print("Router decision: Using RAG")
            return state

    # Use regular LLM/Search
    state["use_rag"] = False
    print("Router decision: Using LLM/Search")
    return state


def rag_node(state: AgentState) -> AgentState:
    """Handles RAG-based responses"""
    messages = state["messages"]
    retrieved_docs = state.get("retrieved_docs", [])

    if not retrieved_docs:
        return state

    # Prepare context from retrieved documents
    context = "\n\n".join([
        f"Document: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
        for doc in retrieved_docs
    ])

    # Create enhanced prompt with context
    user_query = messages[-1].content
    enhanced_prompt = f"""Based on the following context from uploaded documents, please answer the user's question. If the context doesn't contain relevant information, say so clearly.

Context:
{context}

User Question: {user_query}

Please provide a comprehensive answer based on the context provided."""

    # Replace the last message with enhanced prompt
    enhanced_messages = messages[:-1] + [HumanMessage(content=enhanced_prompt)]
    state["messages"] = enhanced_messages

    return state


def create_enhanced_agent(llm, tools=None, use_search=True):
    """Creates an enhanced agent with RAG, memory, and routing capabilities"""

    def agent_node(state: AgentState) -> AgentState:
        messages = state["messages"]

        if state.get("use_rag", False):
            # RAG response
            response = llm.invoke(messages)
            return {"messages": [response]}
        else:
            # Regular LLM response with optional search
            if use_search and tools:
                try:
                    from langgraph.prebuilt import create_react_agent
                    react_agent = create_react_agent(llm, tools)
                    result = react_agent.invoke({"messages": messages})
                    return {"messages": result["messages"]}
                except Exception as e:
                    print(f"Search agent failed: {e}")
                    # Fallback to regular LLM
                    response = llm.invoke(messages)
                    return {"messages": [response]}
            else:
                response = llm.invoke(messages)
                return {"messages": [response]}

    # Create the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("agent", agent_node)

    # Add edges
    workflow.set_entry_point("router")

    def should_use_rag(state: AgentState) -> str:
        return "rag" if state.get("use_rag", False) else "agent"

    workflow.add_conditional_edges("router", should_use_rag, {
        "rag": "rag",
        "agent": "agent"
    })

    workflow.add_edge("rag", "agent")
    workflow.add_edge("agent", END)

    return workflow.compile()


def get_response_from_ai_agent(
        llm_id: str,
        query: List[str],
        allow_search: bool,
        system_prompt: str,
        provider: str,
        user_id: str = "default",
        session_id: str = "default",
        similarity_threshold: float = 0.5
):
    """Enhanced function with memory, RAG, and smart routing"""

    try:
        # Initialize LLM
        if provider == "Groq":
            llm = ChatGroq(model=llm_id)
        elif provider == "OpenAI":
            llm = ChatOpenAI(model=llm_id)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        # Initialize tools
        tools = []
        if allow_search:
            try:
                tools = [TavilySearch(max_results=2)]
            except Exception as e:
                print(f"Warning: Tavily search initialization failed: {e}")

        # Get session history
        session_history = memory_manager.get_session_history(session_id)

        # Prepare messages with history
        messages = [SystemMessage(content=system_prompt)]

        # Add previous conversation history (last 10 messages)
        for hist_msg in session_history[-10:]:
            if hist_msg["type"] == "human":
                messages.append(HumanMessage(content=hist_msg["content"]))
            elif hist_msg["type"] == "ai":
                messages.append(AIMessage(content=hist_msg["content"]))

        # Add current query
        messages.extend([HumanMessage(content=q) for q in query])

        # Create enhanced agent
        agent = create_enhanced_agent(llm, tools, allow_search)

        # Prepare state
        state = {
            "messages": messages,
            "user_id": user_id,
            "session_id": session_id,
            "use_rag": False,
            "similarity_threshold": similarity_threshold,
            "retrieved_docs": []
        }

        # Get response
        response = agent.invoke(state)
        final_messages = response.get("messages", [])

        # Extract AI response
        ai_messages = [m.content for m in final_messages if isinstance(m, AIMessage)]
        final_response = ai_messages[-1] if ai_messages else "No response from agent."

        # Save to memory
        for q in query:
            memory_manager.add_to_session(session_id, {
                "type": "human",
                "content": q
            })

        memory_manager.add_to_session(session_id, {
            "type": "ai",
            "content": final_response
        })

        return final_response

    except Exception as e:
        error_msg = f"Error in AI agent: {str(e)}"
        print(error_msg)
        return error_msg


# Utility functions for PDF processing
def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from PDF file"""
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except ImportError:
        try:
            # Fallback to PyPDF
            import PyPDF2
            text = ""
            with open(pdf_file, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error extracting PDF text with PyPDF2: {e}")
            return ""
        except Exception as e:
            print(f"Error extracting PDF text: {e}")
            return ""



def process_uploaded_pdf(user_id: str, pdf_file, filename: str) -> bool:
    """Process uploaded PDF and store in RAG system"""
    try:
        pdf_text = extract_text_from_pdf(pdf_file)
        if pdf_text.strip():
            success = rag_manager.process_pdf_content(user_id, pdf_text, filename)
            return success
        else:
            print("No text extracted from PDF")
            return False
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return False


def get_chat_history(session_id: str) -> List[Dict]:
    """Get chat history for a session"""
    return memory_manager.get_session_history(session_id)


def clear_chat_history(session_id: str):
    """Clear chat history for a session"""
    memory_manager.clear_session(session_id)


def get_user_documents(user_id: str) -> List[str]:
    """Get list of documents uploaded by user"""
    if not rag_manager.cohere_available or user_id not in rag_manager.vector_stores:
        return []

    try:
        # Extract unique document sources
        docs = rag_manager.vector_stores[user_id].similarity_search("", k=1000)
        sources = set()
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown')
            sources.add(source)

        return list(sources)
    except Exception as e:
        print(f"Error getting user documents: {e}")
        return []
