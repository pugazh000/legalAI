"""
Backend logic for the Legal Document Simplifier AI.

✅ Compatible with both new and old LangChain versions.
✅ Uses OpenRouter via langchain_openai.ChatOpenAI.
✅ Uses FAISS for in-memory vector storage (no SQLite dependency).
"""

import json
import os
import difflib
import re
import io
from typing import Tuple, Dict, List, Any
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date

from dotenv import load_dotenv
load_dotenv()

# -------------------------------------------------------------------------
# LangChain / Embeddings / Vector DB / LLM
# -------------------------------------------------------------------------
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

# Handle new/old LangChain LLMChain import paths
try:
    from langchain.chains import LLMChain
except ImportError:
    try:
        from langchain.chains.llm import LLMChain
    except ImportError:
        try:
            from langchain.chains.base import LLMChain
        except ImportError:
            from langchain_core.runnables import RunnableSequence as LLMChain

from langchain_openai import ChatOpenAI

# File parsing
import pdfplumber
import docx

# -------------------------------------------------------------------------
# Config / Env
# -------------------------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_ID = os.getenv("MODEL_ID", "nvidia/nemotron-nano-9b-v2:free")

print("=== BACKEND STARTUP ===")
print(f"Model ID from env: {MODEL_ID}")
print(f"OPENROUTER_API_KEY present: {'YES' if OPENROUTER_API_KEY else 'NO'}")
print("=======================")

# -------------------------------------------------------------------------
# Helper: Safe LLMChain runner (cross-version)
# -------------------------------------------------------------------------
def _safe_chain_run(llm, prompt, inputs: dict):
    """Run chain safely across LangChain versions."""
    try:
        # Newer versions may use from_llm / invoke
        if hasattr(LLMChain, "from_llm"):
            chain = LLMChain.from_llm(llm, prompt)
        else:
            chain = LLMChain(llm=llm, prompt=prompt)

        if hasattr(chain, "invoke"):
            result = chain.invoke(inputs)
            if isinstance(result, dict) and "text" in result:
                return result["text"]
            return str(result)
        else:
            return chain.run(inputs)
    except TypeError as e:
        print("⚠️ TypeError during chain run (retrying fallback):", e)
        try:
            chain = LLMChain(llm=llm, prompt=prompt)
            return chain.run(inputs)
        except Exception as ex:
            return f"⚠️ LLMChain failed: {ex}"
    except Exception as e:
        return f"⚠️ Chain execution failed: {e}"

# -------------------------------------------------------------------------
# LLM client factory
# -------------------------------------------------------------------------
def _get_llm(model_name: str = None, temperature: float = 0.0):
    if model_name is None:
        model_name = MODEL_ID

    print(f"🔧 Using LLM model: {model_name}")
    print("🔧 Base: https://openrouter.ai/api/v1")

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
    """Parse uploaded pdf/docx/txt and return extracted text."""
    fname = uploaded_file.name.lower()
    content = uploaded_file.read()
    if fname.endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            return "\n\n".join([page.extract_text() or "" for page in pdf.pages])
    elif fname.endswith(".docx"):
        doc = docx.Document(io.BytesIO(content))
        return "\n\n".join([p.text for p in doc.paragraphs])
    else:
        return content.decode("utf-8", errors="ignore")

# -------------------------------------------------------------------------
# Vector store (FAISS)
# -------------------------------------------------------------------------
def _get_embedding_model(embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    return HuggingFaceEmbeddings(model_name=embedding_model_name)

VECTOR_STORES: Dict[str, FAISS] = {}

def chunk_and_store(
    text: str,
    collection_name: str = "legal_docs",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    docs = [Document(page_content=c, metadata={"chunk_id": str(i)}) for i, c in enumerate(chunks)]
    vectordb = FAISS.from_documents(docs, _get_embedding_model())
    VECTOR_STORES[collection_name] = vectordb
    print(f"📚 Stored {len(chunks)} chunks in FAISS collection '{collection_name}'")
    return vectordb

# -------------------------------------------------------------------------
# Core LLM functions
# -------------------------------------------------------------------------
def summarize_text(text: str, model_name: str = None) -> str:
    llm = _get_llm(model_name)
    prompt = PromptTemplate(
        template="""Summarize this legal document in simple English.
Include key purpose, obligations, and important dates.

DOCUMENT:
{doc}""",
        input_variables=["doc"],
    )
    return _safe_chain_run(llm, prompt, {"doc": text})

def semantic_search(query: str, collection_name="legal_docs", top_k=5, model_name=None) -> str:
    if collection_name not in VECTOR_STORES:
        return "⚠️ No document found. Upload and process one first."
    vectordb = VECTOR_STORES[collection_name]
    docs = vectordb.similarity_search(query, k=top_k)
    context_text = "\n\n".join([d.page_content for d in docs])
    llm = _get_llm(model_name)
    prompt = PromptTemplate(
        template="""You are a legal assistant. Use the retrieved context to answer clearly.

CONTEXT:
{context}

QUESTION:
{question}

Answer in plain English for a non-lawyer.""",
        input_variables=["context", "question"],
    )
    return _safe_chain_run(llm, prompt, {"context": context_text, "question": query})

def analyze_document_for_risks(text: str, model_name: str = None) -> Tuple[Dict[str, List[str]], List[str]]:
    llm = _get_llm(model_name)
    prompt = PromptTemplate(
        template="""Analyze this document for risks and obligations.
Return valid JSON only.

DOCUMENT:
{doc}

JSON FORMAT:
{
  "High": ["..."],
  "Medium": ["..."],
  "Low": ["..."],
  "Obligations": ["..."]
}""",
        input_variables=["doc"],
    )
    raw = _safe_chain_run(llm, prompt, {"doc": text[:8000]})
    try:
        data = json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group(0))
    except Exception:
        data = {"High": [], "Medium": [], "Low": [], "Obligations": []}
    return {k: data.get(k, []) for k in ["High", "Medium", "Low"]}, data.get("Obligations", [])

def safe_analyze_document_for_risks(text: str, model_name: str = None):
    try:
        return analyze_document_for_risks(text, model_name)
    except Exception as e:
        print("⚠️ Safe fallback triggered:", e)
        return {"High": [], "Medium": [], "Low": []}, ["Manual review recommended."]

# -------------------------------------------------------------------------
# Date Extraction & Validation
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
            return {"status": "EXPIRED", "message": f"Expired on {expiry.date()}"}
        elif expiry < now + timedelta(days=30):
            return {"status": "EXPIRING_SOON", "message": f"Expires soon on {expiry.date()}"}
        else:
            return {"status": "VALID", "message": f"Valid until {expiry.date()}"}
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}

def intelligent_date_extraction_and_validation(text: str) -> Dict[str, Any]:
    matches = re.findall(r"\b\d{4}[-/]\d{2}[-/]\d{2}\b", text)
    now = datetime.now()
    out = []
    for d in matches:
        try:
            dt = parse_date(d)
            if dt < now:
                s = "Past"
            elif dt < now + timedelta(days=30):
                s = "Expiring Soon"
            else:
                s = "Future"
            out.append({"date": str(dt.date()), "status": s})
        except Exception:
            continue
    return {"count": len(out), "dates": out}

# -------------------------------------------------------------------------
# Document Comparison
# -------------------------------------------------------------------------
def compare_documents(text1: str, text2: str) -> str:
    a, b = text1.splitlines(), text2.splitlines()
    diff = difflib.unified_diff(a, b, fromfile="Document A", tofile="Document B", lineterm="")
    return "### Document Comparison\n```diff\n" + "\n".join(diff) + "\n```"

# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("✅ Backend ready (OpenRouter + FAISS)")
    print("Model:", MODEL_ID)
