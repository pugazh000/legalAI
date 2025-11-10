"""
Backend logic for the Legal Document Simplifier AI.

- Uses OpenRouter via langchain_openai.ChatOpenAI
- Uses FAISS for in-memory vector storage (no sqlite)
- Includes date utilities and safe wrappers for robustness
"""

import os
import io
import re
import json
import difflib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from dateutil.parser import parse as parse_date

from dotenv import load_dotenv
load_dotenv()

# -------------------------------------------------------------------------
# Imports with fallback compatibility (LangChain 0.2 → 0.3+)
# -------------------------------------------------------------------------
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
except Exception:
    raise ImportError("Please install langchain-community.")

try:
    from langchain_core.documents import Document
    from langchain_core.prompts import PromptTemplate
except Exception:
    from langchain.docstore.document import Document
    from langchain.prompts import PromptTemplate

try:
    from langchain.chains.llm import LLMChain
except Exception:
    try:
        from langchain.chains.base import LLMChain
    except Exception:
        from langchain_core.runnables import RunnableSequence as LLMChain

try:
    from langchain_openai import ChatOpenAI
except Exception:
    raise ImportError("Please install langchain-openai.")

# PDF / DOCX parsing
import pdfplumber
import docx

# -------------------------------------------------------------------------
# Environment configuration
# -------------------------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("GEMINI_API_KEY")
MODEL_ID = os.getenv("MODEL_ID", "nvidia/nemotron-nano-9b-v2:free")

print("=== BACKEND STARTUP ===")
print(f"Model ID: {MODEL_ID}")
print(f"API Key loaded: {'YES' if OPENROUTER_API_KEY else 'NO'}")
print("=======================")

# -------------------------------------------------------------------------
# LLM factory (OpenRouter endpoint)
# -------------------------------------------------------------------------
def _get_llm(model_name: str = None, temperature: float = 0.0):
    """Return a ChatOpenAI client configured for OpenRouter."""
    if model_name is None:
        model_name = MODEL_ID

    print(f"🔧 Using model: {model_name}")
    return ChatOpenAI(
        model=model_name,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=temperature,
    )

# -------------------------------------------------------------------------
# File parsing
# -------------------------------------------------------------------------
def parse_file(uploaded_file) -> str:
    """Extract text from PDF or DOCX file."""
    fname = uploaded_file.name.lower()
    content = uploaded_file.read()

    if fname.endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            text = "\n\n".join([page.extract_text() or "" for page in pdf.pages])
    elif fname.endswith(".docx"):
        doc = docx.Document(io.BytesIO(content))
        text = "\n\n".join([p.text for p in doc.paragraphs])
    else:
        text = content.decode("utf-8", errors="ignore")
    return text

# -------------------------------------------------------------------------
# Embedding & FAISS vector store
# -------------------------------------------------------------------------
def _get_embedding_model(name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Return a HuggingFace embedding model (local)."""
    return HuggingFaceEmbeddings(model_name=name)

VECTOR_STORES: Dict[str, FAISS] = {}

def chunk_and_store(
    text: str,
    collection_name: str = "legal_docs",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    """Split text into chunks and store embeddings in FAISS."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)
    docs = [Document(page_content=c, metadata={"chunk_id": str(i)}) for i, c in enumerate(chunks)]

    embedder = _get_embedding_model(embedding_model_name)
    vectordb = FAISS.from_documents(docs, embedder)
    VECTOR_STORES[collection_name] = vectordb
    print(f"📚 Stored {len(chunks)} chunks into FAISS collection '{collection_name}'")
    return vectordb

# -------------------------------------------------------------------------
# Core RAG & LLM logic
# -------------------------------------------------------------------------
def semantic_search(query: str, collection_name: str = "legal_docs", top_k: int = 5) -> str:
    if collection_name not in VECTOR_STORES:
        return "⚠️ No vector store found. Please upload and process a document first."

    vectordb = VECTOR_STORES[collection_name]
    docs = vectordb.similarity_search(query, k=top_k)
    context = "\n\n".join([d.page_content[:1500] for d in docs])

    llm = _get_llm()
    prompt = PromptTemplate(
        template="""You are a legal assistant. Using this context, answer clearly and simply.

Context:
{context}

Question:
{question}""",
        input_variables=["context", "question"],
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.invoke({"context": context, "question": query})["text"]

def summarize_text(text: str) -> str:
    """Summarize a legal document in plain English."""
    llm = _get_llm()
    prompt = PromptTemplate(
        template="""Summarize this legal document in plain English.
Highlight its purpose, obligations, key dates, and important actions.

Document:
{doc}""",
        input_variables=["doc"],
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.invoke({"doc": text})["text"]

# -------------------------------------------------------------------------
# Risk & Obligation Analysis
# -------------------------------------------------------------------------
def analyze_document_for_risks(text: str) -> Tuple[Dict[str, List[str]], List[str]]:
    llm = _get_llm()
    prompt = PromptTemplate(
        template="""Analyze the following document. Return JSON with keys "High", "Medium", "Low", and "Obligations".

Document:
{doc}""",
        input_variables=["doc"],
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    raw = chain.invoke({"doc": text[:8000]})["text"]
    try:
        cleaned = raw.strip().strip("```json").strip("```")
        parsed = json.loads(re.search(r"\{.*\}", cleaned, re.DOTALL).group(0))
    except Exception as e:
        print("⚠️ JSON parsing failed:", e)
        parsed = {"High": [], "Medium": [], "Low": [], "Obligations": []}
    return {k: parsed.get(k, []) for k in ["High", "Medium", "Low"]}, parsed.get("Obligations", [])

def safe_analyze_document_for_risks(text: str):
    try:
        return analyze_document_for_risks(text)
    except Exception as e:
        print("⚠️ safe fallback:", e)
        return {"High": [], "Medium": [], "Low": []}, ["Manual review recommended"]

# -------------------------------------------------------------------------
# Date Utilities
# -------------------------------------------------------------------------
def simple_date_validator(text: str) -> Dict[str, Any]:
    dates = re.findall(r"\b\d{4}[-/]\d{2}[-/]\d{2}\b", text)
    result = {"status": "UNKNOWN", "message": "No clear dates found"}
    if not dates:
        return result
    try:
        expiry = parse_date(dates[-1])
        now = datetime.now()
        if expiry < now:
            result = {"status": "EXPIRED", "message": f"Expired on {expiry.date()}"}
        elif expiry < now + timedelta(days=30):
            result = {"status": "EXPIRING_SOON", "message": f"Expires soon on {expiry.date()}"}
        else:
            result = {"status": "VALID", "message": f"Valid until {expiry.date()}"}
    except Exception as e:
        result["message"] = f"Date parsing failed: {e}"
    return result

def intelligent_date_extraction_and_validation(text: str) -> Dict[str, Any]:
    """Extract multiple dates and classify them."""
    matches = re.findall(r"\b\d{4}[-/]\d{2}[-/]\d{2}\b", text)
    now = datetime.now()
    extracted = []
    for d in matches:
        try:
            dt = parse_date(d)
            if dt < now:
                status = "Past"
            elif dt < now + timedelta(days=30):
                status = "Expiring Soon"
            else:
                status = "Future"
            extracted.append({"date": str(dt.date()), "status": status})
        except Exception:
            continue
    return {"dates": extracted, "count": len(extracted)}

# -------------------------------------------------------------------------
# ✅ FIXED: Missing Function
# -------------------------------------------------------------------------
def semantic_search_with_intelligent_validation(
    query: str, collection_name: str = "legal_docs", top_k: int = 5
) -> str:
    """Semantic search + intelligent date validation."""
    base = semantic_search(query, collection_name, top_k)
    if collection_name not in VECTOR_STORES:
        return base
    vectordb = VECTOR_STORES[collection_name]
    docs = vectordb.similarity_search(query, k=top_k)
    combined = " ".join(d.page_content for d in docs)
    extracted = intelligent_date_extraction_and_validation(combined)
    if not extracted["dates"]:
        return base + "\n\nℹ️ No dates found."
    summary = [f"- {d['date']} → {d['status']}" for d in extracted["dates"]]
    return base + "\n\n📅 Date Analysis:\n" + "\n".join(summary)

# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------
def compare_documents(text1: str, text2: str) -> str:
    a_lines, b_lines = text1.splitlines(), text2.splitlines()
    diff = difflib.unified_diff(a_lines, b_lines, fromfile="Document A", tofile="Document B", lineterm="")
    return "### Document Comparison\n```diff\n" + "\n".join(diff) + "\n```"

# -------------------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("✅ Backend module loaded successfully (FAISS + OpenRouter).")
    print("Model ID:", MODEL_ID)

