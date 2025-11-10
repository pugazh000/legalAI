"""
Backend logic for ⚖️ The Legal Simplifier AI

- Uses OpenRouter via langchain_openai.ChatOpenAI
- Uses FAISS for in-memory vector storage
- Compatible with LangChain v0.2 and v0.3+
"""

import os
import io
import re
import json
import difflib
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date
from typing import Dict, List, Tuple, Any

import pdfplumber
import docx
from dotenv import load_dotenv

# Load environment
load_dotenv()

# -------------------------------------------------------------------------
# LangChain imports (safe across versions)
# -------------------------------------------------------------------------
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
except ImportError:
    from langchain.embeddings.huggingface import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS

try:
    from langchain_core.documents import Document
    from langchain_core.prompts import PromptTemplate
except ImportError:
    from langchain.docstore.document import Document
    from langchain.prompts import PromptTemplate

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    raise ImportError("Please install 'langchain-openai' to use OpenRouter LLMs.")

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("GEMINI_API_KEY")
MODEL_ID = os.getenv("MODEL_ID", "nvidia/nemotron-nano-9b-v2:free")

print("=== BACKEND STARTUP ===")
print(f"Model ID: {MODEL_ID}")
print(f"OpenRouter API key present: {'YES' if OPENROUTER_API_KEY else 'NO'}")
print("========================")

# -------------------------------------------------------------------------
# LLM client factory (OpenRouter)
# -------------------------------------------------------------------------
def _get_llm(model_name: str = None, temperature: float = 0.0):
    """Return a ChatOpenAI client configured for OpenRouter."""
    if model_name is None:
        model_name = MODEL_ID

    try:
        key_present = bool(OPENROUTER_API_KEY and len(OPENROUTER_API_KEY) > 8)
    except Exception:
        key_present = False

    print(f"🔧 _get_llm() -> model: {model_name}")
    print(f"🔧 OpenRouter key loaded? {'YES' if key_present else 'NO'}")

    return ChatOpenAI(
        model=model_name,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=temperature,
    )

# -------------------------------------------------------------------------
# ✅ Universal LLM chain runner (works for both old and new LangChain)
# -------------------------------------------------------------------------
def run_llm_chain(llm, prompt: "PromptTemplate", inputs: dict) -> str:
    """Run LLM chain compatible with both LangChain v0.2 and v0.3+."""
    try:
        from langchain.chains import LLMChain
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain.run(inputs)
    except Exception:
        runnable = prompt | llm
        result = runnable.invoke(inputs)
        if isinstance(result, dict) and "text" in result:
            return result["text"]
        return str(result)

# -------------------------------------------------------------------------
# File parsing
# -------------------------------------------------------------------------
def parse_file(uploaded_file) -> str:
    """Parse uploaded pdf/docx file-like object and return extracted text."""
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
# Embeddings & Vector store (FAISS, in-memory)
# -------------------------------------------------------------------------
def _get_embedding_model(embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Return HuggingFace embedding model (no API key required)."""
    return HuggingFaceEmbeddings(model_name=embedding_model_name)

VECTOR_STORES: Dict[str, FAISS] = {}

def chunk_and_store(
    text: str,
    collection_name: str = "legal_docs",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
):
    """Split text into chunks, embed, and store in FAISS memory."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(text)
    docs = [Document(page_content=c, metadata={"chunk_id": str(i)}) for i, c in enumerate(chunks)]
    embedder = _get_embedding_model()
    vectordb = FAISS.from_documents(docs, embedder)
    VECTOR_STORES[collection_name] = vectordb
    print(f"📚 Stored {len(chunks)} chunks into FAISS collection '{collection_name}'")
    return vectordb

# -------------------------------------------------------------------------
# Core LLM Functions
# -------------------------------------------------------------------------
def summarize_text(text: str, model_name: str = None) -> str:
    llm = _get_llm(model_name=model_name)
    prompt_template = """Summarize the document in plain language.
Highlight purpose, key obligations, important dates, and immediate actions.

DOCUMENT:
{doc}"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["doc"])
    return run_llm_chain(llm, prompt, {"doc": text})

def semantic_search(query: str, collection_name: str = "legal_docs", top_k: int = 5, model_name: str = None) -> str:
    if collection_name not in VECTOR_STORES:
        return "⚠️ No vector store found. Please upload and process a document first."
    vectordb = VECTOR_STORES[collection_name]
    docs = vectordb.similarity_search(query, k=top_k)
    context_text = "\n\n".join([d.page_content[:2000] for d in docs])
    llm = _get_llm(model_name=model_name)
    prompt_template = """You are a legal assistant. Use these retrieved chunks to answer:

RETRIEVED CHUNKS:
{context}

QUESTION:
{question}

Answer clearly in plain English for a non-lawyer.
"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return run_llm_chain(llm, prompt, {"context": context_text, "question": query})

def analyze_document_for_risks(text: str, model_name: str = None) -> Tuple[Dict[str, List[str]], List[str]]:
    llm = _get_llm(model_name=model_name)
    prompt_template = """Analyze this document and identify risks and obligations.

DOCUMENT:
{doc}

Respond in JSON:
{
  "High": ["..."],
  "Medium": ["..."],
  "Low": ["..."],
  "Obligations": ["..."]
}"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["doc"])
    raw = run_llm_chain(llm, prompt, {"doc": text[:8000]})
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.splitlines()[1:-1])
        parsed = json.loads(re.search(r"\{.*\}", cleaned, re.DOTALL).group(0))
    except Exception as e:
        print("⚠️ Risk parsing failed:", e)
        parsed = {"High": [], "Medium": [], "Low": [], "Obligations": []}
    return {k: parsed.get(k, []) for k in ["High", "Medium", "Low"]}, parsed.get("Obligations", [])

def safe_analyze_document_for_risks(text: str, model_name: str = None):
    try:
        return analyze_document_for_risks(text, model_name)
    except Exception as e:
        print("⚠️ safe_analyze_document_for_risks fallback triggered:", e)
        return {"High": [], "Medium": [], "Low": []}, ["Manual review recommended due to analysis limitations"]

# -------------------------------------------------------------------------
# Date Utilities
# -------------------------------------------------------------------------
def simple_date_validator(text: str) -> Dict[str, Any]:
    date_matches = re.findall(r"\b\d{4}[-/]\d{2}[-/]\d{2}\b", text)
    results = {"status": "UNKNOWN", "message": "No clear dates found", "action_needed": []}
    if date_matches:
        try:
            expiry = parse_date(date_matches[-1])
            now = datetime.now()
            if expiry < now:
                results["status"] = "EXPIRED"
                results["message"] = f"Expired on {expiry.date()}"
                results["action_needed"].append("Renew or renegotiate the contract")
            elif expiry < now + timedelta(days=30):
                results["status"] = "EXPIRING_SOON"
                results["message"] = f"Expires soon on {expiry.date()}"
                results["action_needed"].append("Plan renewal or extension")
            else:
                results["status"] = "VALID"
                results["message"] = f"Valid until {expiry.date()}"
        except Exception as e:
            results["message"] = f"Date parsing failed: {e}"
    return results

def semantic_search_with_dates(query: str, collection_name: str = "legal_docs", top_k: int = 5) -> str:
    base_answer = semantic_search(query=query, collection_name=collection_name, top_k=top_k)
    if collection_name not in VECTOR_STORES:
        return base_answer
    vectordb = VECTOR_STORES[collection_name]
    docs = vectordb.similarity_search(query, k=top_k)
    validation = simple_date_validator(" ".join(d.page_content for d in docs))
    if validation["status"] == "EXPIRED":
        return f"🔴 NO - {validation['message']}"
    elif validation["status"] == "EXPIRING_SOON":
        return f"🟡 ALMOST EXPIRED - {validation['message']}"
    elif validation["status"] == "VALID":
        return f"🟢 YES - {validation['message']}"
    else:
        return base_answer

# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------
def compare_documents(text1: str, text2: str) -> str:
    a_lines, b_lines = text1.splitlines(), text2.splitlines()
    diff = difflib.unified_diff(a_lines, b_lines, fromfile="Document A", tofile="Document B", lineterm="")
    md_lines = ["### Document Comparison\n", "```diff"]
    md_lines.extend(list(diff))
    md_lines.append("```")
    return "\n".join(md_lines)

# -------------------------------------------------------------------------
# Advanced Date Extraction + Intelligent Search
# -------------------------------------------------------------------------
def intelligent_date_extraction_and_validation(text: str) -> Dict[str, Any]:
    """Find all dates in text and label them as Past / Expiring Soon / Future."""
    date_matches = re.findall(r"\b\d{4}[-/]\d{2}[-/]\d{2}\b", text)
    extracted_dates = []
    now = datetime.now()
    for d in date_matches:
        try:
            dt = parse_date(d)
            if dt < now:
                status = "Past"
            elif dt < now + timedelta(days=30):
                status = "Expiring Soon"
            else:
                status = "Future"
            extracted_dates.append({"date": str(dt.date()), "status": status})
        except Exception:
            continue
    return {"dates": extracted_dates, "count": len(extracted_dates)}

def semantic_search_with_intelligent_validation(
    query: str,
    collection_name: str = "legal_docs",
    top_k: int = 5
) -> str:
    """Run semantic search and append date analysis results."""
    base_answer = semantic_search(query=query, collection_name=collection_name, top_k=top_k)
    if collection_name not in VECTOR_STORES:
        return base_answer

    vectordb = VECTOR_STORES[collection_name]
    docs = vectordb.similarity_search(query, k=top_k)
    combined_text = " ".join(d.page_content for d in docs)
    extracted = intelligent_date_extraction_and_validation(combined_text)

    if not extracted["dates"]:
        return base_answer + "\n\nℹ️ No clear dates found."
    else:
        summary_lines = [f"- {d['date']} → {d['status']}" for d in extracted["dates"]]
        return base_answer + "\n\n📅 Date Analysis:\n" + "\n".join(summary_lines)


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("✅ Backend module loaded successfully (FAISS + OpenRouter ready).")
    print("Model ID:", MODEL_ID)

