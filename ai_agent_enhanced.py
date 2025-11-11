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
    """Manages document storage and retrieval with advanced semantic search"""

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
                    model="rerank-english-v3.0",  # Specify rerank model
                    top_n=5  # Set top_n for reranking
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
            chunk_size=800,  # Smaller chunks for better precision
            chunk_overlap=200,  # Good overlap to maintain context
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],  # Better semantic boundaries
        )

    def process_pdf_content(self, user_id: str, pdf_pages: List[Dict[str, Any]], filename: str):
        """Process PDF content with page numbers and store in vector database"""
        if not self.cohere_available:
            print("Warning: Cohere not available. Cannot process PDF for RAG.")
            return False

        try:
            all_documents = []
            
            # Process each page separately to maintain page numbers
            for page_info in pdf_pages:
                page_num = page_info['page_number']
                page_text = page_info['text']
                
                # Skip empty pages
                if not page_text or len(page_text.strip()) < 50:
                    continue
                
                # Split page text into chunks
                chunks = self.text_splitter.split_text(page_text)
                
                # Create documents with rich metadata
                for i, chunk in enumerate(chunks):
                    if len(chunk.strip()) < 20:  # Skip very small chunks
                        continue
                        
                    doc = Document(
                        page_content=chunk.strip(),
                        metadata={
                            "source": filename,
                            "page_number": page_num,
                            "user_id": user_id,
                            "chunk_id": i,
                            "total_chunks_in_page": len(chunks),
                            "timestamp": datetime.now().isoformat(),
                            "char_count": len(chunk)
                        }
                    )
                    all_documents.append(doc)

            if not all_documents:
                print("[v0] No valid chunks created from PDF")
                return False

            # Create or update vector store for user
            if user_id in self.vector_stores:
                # Add to existing store
                self.vector_stores[user_id].add_documents(all_documents)
                print(f"[v0] Added {len(all_documents)} chunks to existing vector store")
            else:
                # Create new store
                self.vector_stores[user_id] = FAISS.from_documents(
                    all_documents, self.embeddings
                )
                print(f"[v0] Created new vector store with {len(all_documents)} chunks")

            print(f"[v0] Successfully processed {len(pdf_pages)} pages into {len(all_documents)} chunks")
            return True
        except Exception as e:
            print(f"Error processing PDF content: {e}")
            import traceback
            traceback.print_exc()
            return False

    def retrieve_relevant_docs(self, user_id: str, query: str, k: int = 5) -> List[Document]:
        """Retrieve relevant documents using hybrid semantic search + reranking"""
        if user_id not in self.vector_stores:
            print(f"[v0] No vector store found for user: {user_id}")
            return []

        try:
            # Step 1: Initial retrieval with higher k for better recall
            initial_k = min(k * 4, 20)  # Get more candidates for reranking
            docs_with_scores = self.vector_stores[user_id].similarity_search_with_score(
                query, k=initial_k
            )

            if not docs_with_scores:
                print(f"[v0] No documents found for query")
                return []

            docs = [doc for doc, score in docs_with_scores]
            print(f"[v0] Initial retrieval: {len(docs)} documents")

            # Step 2: Rerank using Cohere for semantic relevance
            if self.cohere_available and self.reranker and len(docs) > 0:
                try:
                    # Prepare documents and query for reranking
                    doc_texts = [doc.page_content for doc in docs]
                    
                    # Use Cohere client directly for better control
                    rerank_response = cohere_client.rerank(
                        model="rerank-english-v3.0",
                        query=query,
                        documents=doc_texts,
                        top_n=k,
                        return_documents=False
                    )

                    # Get reranked documents with scores
                    reranked_docs = []
                    for result in rerank_response.results:
                        idx = result.index
                        relevance_score = result.relevance_score
                        
                        original_doc = docs[idx]
                        # Add relevance score to metadata
                        original_doc.metadata["relevance_score"] = relevance_score
                        reranked_docs.append(original_doc)

                    print(f"[v0] Reranked to top {len(reranked_docs)} documents")
                    
                    # Log relevance scores for debugging
                    for i, doc in enumerate(reranked_docs[:3]):
                        score = doc.metadata.get('relevance_score', 'N/A')
                        page = doc.metadata.get('page_number', 'N/A')
                        print(f"[v0] Rank {i+1}: Page {page}, Score: {score:.4f}")
                    
                    return reranked_docs

                except Exception as e:
                    print(f"Reranking failed: {e}")
                    import traceback
                    traceback.print_exc()
                    # Fallback to similarity search results
                    return docs[:k]
            else:
                # No reranker available, use vector similarity scores
                for doc, score in docs_with_scores[:k]:
                    doc.metadata["relevance_score"] = float(1.0 / (1.0 + score))
                return [doc for doc, _ in docs_with_scores[:k]]
                
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            import traceback
            traceback.print_exc()
            return []

    def calculate_similarity_score(self, user_id: str, query: str) -> float:
        """Calculate similarity score to determine if RAG should be used"""
        if user_id not in self.vector_stores:
            print(f"[v0] No vector store for similarity calculation")
            return 0.0

        try:
            # Get top document with score
            docs_with_scores = self.vector_stores[user_id].similarity_search_with_score(query, k=1)
            
            if docs_with_scores:
                doc, distance = docs_with_scores[0]
                # Convert FAISS L2 distance to similarity score (0-1 range)
                # Lower distance = higher similarity
                similarity = 1.0 / (1.0 + distance)
                print(f"[v0] Similarity score: {similarity:.4f} (distance: {distance:.4f})")
                return similarity

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

    print(f"\n[v0] ====== ROUTER EVALUATION ======")
    print(f"[v0] Query: {user_query[:100]}...")
    
    # Only use RAG if Cohere is available and user has documents
    if rag_manager.cohere_available and user_id in rag_manager.vector_stores:
        # Calculate similarity with stored documents
        similarity_score = rag_manager.calculate_similarity_score(user_id, user_query)
        threshold = state.get("similarity_threshold", 0.3)

        print(f"[v0] Similarity score: {similarity_score:.4f}, Threshold: {threshold}")

        if similarity_score > threshold:
            # Use RAG
            retrieved_docs = rag_manager.retrieve_relevant_docs(user_id, user_query, k=5)
            state["retrieved_docs"] = retrieved_docs
            state["use_rag"] = True
            print(f"[v0] ✓ Router decision: Using RAG with {len(retrieved_docs)} documents")
            print(f"[v0] =================================\n")
            return state
        else:
            print(f"[v0] ✗ Similarity too low for RAG")

    # Use regular LLM/Search
    state["use_rag"] = False
    print(f"[v0] Router decision: Using LLM/Search (no RAG)")
    print(f"[v0] =================================\n")
    return state


def rag_node(state: AgentState) -> AgentState:
    """Handles RAG-based responses with enhanced context and source references"""
    messages = state["messages"]
    retrieved_docs = state.get("retrieved_docs", [])

    if not retrieved_docs:
        print("[v0] RAG node: No documents retrieved")
        return state

    print(f"\n[v0] ====== RAG CONTEXT BUILDING ======")
    print(f"[v0] Building context from {len(retrieved_docs)} documents")

    # Build context with numbered sources
    context_parts = []
    sources_metadata = []
    
    for idx, doc in enumerate(retrieved_docs, 1):
        page_num = doc.metadata.get('page_number', 'Unknown')
        source = doc.metadata.get('source', 'Unknown')
        relevance = doc.metadata.get('relevance_score', 0)
        
        # Format context chunk
        context_parts.append(
            f"[EXCERPT {idx}] (Source: {source}, Page: {page_num}, Relevance: {relevance:.3f})\n{doc.page_content}"
        )
        
        # Store metadata for references section
        sources_metadata.append({
            'excerpt_num': idx,
            'source': source,
            'page': page_num,
            'relevance': relevance,
            'preview': doc.page_content[:100] + "..."
        })
        
        print(f"[v0] Excerpt {idx}: Page {page_num}, Relevance: {relevance:.3f}")
    
    context = "\n\n" + "="*80 + "\n\n".join(context_parts) + "\n\n" + "="*80
    
    user_query = messages[-1].content

    enhanced_prompt = f"""You are a highly accurate document analysis assistant. Your task is to answer questions based EXCLUSIVELY on the provided document excerpts.

DOCUMENT CONTEXT:
{context}

USER QUESTION: 
{user_query}

CRITICAL INSTRUCTIONS:
1. **ONLY use information from the document excerpts above** - Do not use external knowledge
2. **CITE PAGE NUMBERS** - Always reference the specific page number(s) where you found information
3. **BE SPECIFIC** - Quote relevant parts when appropriate
4. **SYNTHESIZE** - If information appears across multiple pages, synthesize it coherently
5. **ADMIT LIMITATIONS** - If the excerpts don't contain enough information, clearly state: "Based on the available excerpts, I cannot find sufficient information about [topic]. The document may not cover this, or it might be in sections not retrieved."

RESPONSE FORMAT:
1. Start with a direct answer to the question
2. Support your answer with specific references to page numbers (e.g., "According to page 5..." or "As stated on page 7...")
3. End with a **References** section listing all pages you cited

Example format:
[Your comprehensive answer here with inline citations like "According to page 3..." or "As mentioned on page 7..."]

**References:**
- Page 3: [Brief description of what was found]
- Page 7: [Brief description of what was found]

Now, please provide your answer:"""

    # Replace the last message with enhanced prompt
    enhanced_messages = messages[:-1] + [HumanMessage(content=enhanced_prompt)]
    state["messages"] = enhanced_messages
    
    print(f"[v0] Context length: {len(context)} characters")
    print(f"[v0] Sources: {set(s['page'] for s in sources_metadata)}")
    print(f"[v0] ==================================\n")

    return state


def create_enhanced_agent(llm, tools=None, use_search=True):
    """Creates an enhanced agent with RAG, memory, and routing capabilities"""

    def agent_node(state: AgentState) -> AgentState:
        messages = state["messages"]

        if state.get("use_rag", False):
            # RAG response
            print("[v0] Agent: Generating RAG-based response...")
            response = llm.invoke(messages)
            print(f"[v0] Agent: Response generated ({len(response.content)} chars)")
            return {"messages": [response]}
        else:
            # Regular LLM response with optional search
            if use_search and tools:
                try:
                    print("[v0] Agent: Using search-enabled agent...")
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
                print("[v0] Agent: Using standard LLM response...")
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
        similarity_threshold: float = 0.3
):
    """Enhanced function with memory, RAG, and smart routing"""

    try:
        # Initialize LLM
        if provider == "Groq":
            llm = ChatGroq(model=llm_id, temperature=0.1)  # Lower temperature for more accurate RAG responses
        elif provider == "OpenAI":
            llm = ChatOpenAI(model=llm_id, temperature=0.1)
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
        print(f"\n[v0] ====== AGENT INVOCATION ======")
        print(f"[v0] User: {user_id}, Session: {session_id}")
        response = agent.invoke(state)
        final_messages = response.get("messages", [])

        # Extract AI response
        ai_messages = [m.content for m in final_messages if isinstance(m, AIMessage)]
        final_response = ai_messages[-1] if ai_messages else "No response from agent."
        
        print(f"[v0] Final response length: {len(final_response)} chars")
        print(f"[v0] ================================\n")

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
        import traceback
        traceback.print_exc()
        return error_msg


def extract_text_from_pdf(pdf_file) -> List[Dict[str, Any]]:
    """Extract text from PDF file with page numbers"""
    try:
        import pdfplumber
        pages_data = []
        
        with pdfplumber.open(pdf_file) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text()
                if page_text and len(page_text.strip()) > 50:  # Skip nearly empty pages
                    pages_data.append({
                        'page_number': page_num,
                        'text': page_text.strip()
                    })
                    
        print(f"[v0] Extracted {len(pages_data)} pages from PDF using pdfplumber")
        return pages_data
        
    except ImportError:
        print("[v0] pdfplumber not available, trying PyPDF2")
        try:
            # Fallback to PyPDF2
            import PyPDF2
            pages_data = []
            
            with open(pdf_file, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages, start=1):
                    page_text = page.extract_text()
                    if page_text and len(page_text.strip()) > 50:
                        pages_data.append({
                            'page_number': page_num,
                            'text': page_text.strip()
                        })
                        
            print(f"[v0] Extracted {len(pages_data)} pages from PDF using PyPDF2")
            return pages_data
            
        except Exception as e:
            print(f"Error extracting PDF text with PyPDF2: {e}")
            return []
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        import traceback
        traceback.print_exc()
        return []


def process_uploaded_pdf(user_id: str, pdf_file, filename: str) -> bool:
    """Process uploaded PDF and store in RAG system"""
    try:
        print(f"\n[v0] ====== PDF PROCESSING ======")
        print(f"[v0] File: {filename}")
        print(f"[v0] User: {user_id}")
        
        pdf_pages = extract_text_from_pdf(pdf_file)
        
        if pdf_pages:
            print(f"[v0] Successfully extracted {len(pdf_pages)} pages")
            total_chars = sum(len(p['text']) for p in pdf_pages)
            print(f"[v0] Total characters: {total_chars}")
            
            success = rag_manager.process_pdf_content(user_id, pdf_pages, filename)
            
            if success:
                print(f"[v0] ✓ PDF processing completed successfully")
            else:
                print(f"[v0] ✗ PDF processing failed")
                
            print(f"[v0] ============================\n")
            return success
        else:
            print("[v0] ✗ No text extracted from PDF")
            print(f"[v0] ============================\n")
            return False
            
    except Exception as e:
        print(f"[v0] ✗ Error processing PDF: {e}")
        import traceback
        traceback.print_exc()
        print(f"[v0] ============================\n")
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
