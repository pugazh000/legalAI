"""
Backend logic for the Legal Document Simplifier AI.

- Uses OpenRouter via langchain_openai.ChatOpenAI
- Uses FAISS for in-memory vector storage (no sqlite)
- Includes date utilities and safe wrappers for robustness
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
# Defensive imports for LangChain universe (works across v0.2 / v0.3+)
# -------------------------------------------------------------------------
# text splitter
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except Exception:
        raise ImportError(
            "Cannot import RecursiveCharacterTextSplitter. "
            "Install 'langchain-text-splitters' or a compatible langchain version."
        )

# embeddings + vectorstores (community package)
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
except Exception:
    raise ImportError(
        "Cannot import langchain_community components. Install 'langchain-community'."
    )

# Document & PromptTemplate
try:
    from langchain_core.documents import Document
    from langchain_core.prompts import PromptTemplate
except Exception:
    try:
        from langchain.docstore.document import Document
        from langchain.prompts import PromptTemplate
    except Exception:
        raise ImportError(
            "Cannot import Document/PromptTemplate. Install 'langchain-core' or compatible langchain."
        )

# LLMChain: multi-version safe import
try:
    from langchain.chains.llm import LLMChain
except Exception:
    try:
        from langchain.chains.base import LLMChain
    except Exception:
        try:
            from langchain_core.runnables import RunnableSequence as LLMChain  # fallback
        except Exception:
            raise ImportError("Cannot import LLMChain or RunnableSequence. Update LangChain.")

# Chat client (OpenRouter)
try:
    from langchain_openai import ChatOpenAI
except Exception:
    raise ImportError("Install 'langchain-openai' to use ChatOpenAI (OpenRouter support).")

# File parsing
import pdfplumber
import docx

# -------------------------------------------------------------------------
# Config / Env
# -------------------------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("GEMINI_API_KEY")
MODEL_ID = os.getenv("MODEL_ID", "nvidia/nemotron-nano-9b-v2:free")

print("=== BACKEND STARTUP ===")
print(f"Model ID from env: {MODEL_ID}")
print(f"OPENROUTER_API_KEY present: {'YES' if OPENROUTER_API_KEY else 'NO'}")
print("=======================")

# -------------------------------------------------------------------------
# Helper - build LLMChain or RunnableSequence safely
# -------------------------------------------------------------------------
def _make_llm_chain(llm, prompt):
    """Return a runnable or chain object, safe across LangChain versions."""
    try:
        return LLMChain(llm=llm, prompt=prompt)
    except TypeError:
        # new runnable versions
        from langchain_core.runnables import RunnableSequence
        return RunnableSequence(prompt | llm)

# -------------------------------------------------------------------------
# LLM client factory
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
    print(f"🔧 OPENROUTER key loaded? {'YES' if key_present else 'NO'}")
    print("🔧 openai_api_base: https://openrouter.ai/api/v1")

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
    """Parse uploaded pdf/docx file-like object and return extracted text."""
    fname = uploaded_file.name.lower()
    content = uploaded_file.read()
    text = ""
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
# Embeddings & Vector store
# -------------------------------------------------------------------------
def _get_embedding_model(embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Return a HuggingFaceEmbeddings instance."""
    return HuggingFaceEmbeddings(model_name=embedding_model_name)

VECTOR_STORES: Dict[str, FAISS] = {}

def chunk_and_store(
    text: str,
    collection_name: str = "legal_docs",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(text)
    docs = [
        Document(page_content=c, metadata={"chunk_id": str(i), "source": collection_name})
        for i, c in enumerate(chunks)
    ]
    embedder = _get_embedding_model(embedding_model_name)
    vectordb = FAISS.from_documents(docs, embedder)
    VECTOR_STORES[collection_name] = vectordb
    print(f"📚 Stored {len(chunks)} chunks into FAISS collection '{collection_name}'")
    return vectordb

# -------------------------------------------------------------------------
# Core RAG functions
# -------------------------------------------------------------------------
def semantic_search(
    query: str,
    collection_name: str = "legal_docs",
    top_k: int = 5,
    model_name: str = None,
) -> str:
    if collection_name not in VECTOR_STORES:
        return "⚠️ No vector store found. Please upload and process a document first."

    vectordb = VECTOR_STORES[collection_name]
    docs = vectordb.similarity_search(query, k=top_k)

    ctxs = [f"--- chunk {i} ---\n{d.page_content.strip()[:2000]}" for i, d in enumerate(docs)]
    context_text = "\n\n".join(ctxs)

    llm = _get_llm(model_name=model_name)
    prompt = PromptTemplate(
        template="""You are a legal assistant. Use these retrieved chunks to answer:

RETRIEVED CHUNKS:
{context}

QUESTION:
{question}

Answer clearly in plain English for a non-lawyer.""",
        input_variables=["context", "question"],
    )
    chain = _make_llm_chain(llm, prompt)

    try:
        if hasattr(chain, "invoke"):
            result = chain.invoke({"context": context_text, "question": query})
            return result.get("text", str(result))
        else:
            return chain.run({"context": context_text, "question": query})
    except Exception as e:
        print("⚠️ semantic_search failed:", e)
        return "Semantic search failed due to model error."

# ✅ FIXED FUNCTION (prevents TypeError)
def summarize_text(text: str, model_name: str = None) -> str:
    """Summarize a document in plain language."""
    llm = _get_llm(model_name=model_name)
    prompt = PromptTemplate(
        template="""Summarize the document in plain language.
Highlight purpose, key obligations, important dates, and immediate actions.

DOCUMENT:
{doc}""",
        input_variables=["doc"],
    )
    chain = _make_llm_chain(llm, prompt)

    try:
        if hasattr(chain, "invoke"):
            result = chain.invoke({"doc": text})
            if isinstance(result, dict) and "text" in result:
                return result["text"]
            elif isinstance(result, str):
                return result
            else:
                return str(result)
        else:
            return chain.run({"doc": text})
    except Exception as e:
        print("⚠️ summarize_text failed:", e)
        return "Summary unavailable due to model response error."

# -------------------------------------------------------------------------
# Risk & Obligations
# -------------------------------------------------------------------------
def analyze_document_for_risks(text: str, model_name: str = None) -> Tuple[Dict[str, List[str]], List[str]]:
    llm = _get_llm(model_name=model_name)
    prompt = PromptTemplate(
        template="""Analyze this document and identify risks and obligations.

DOCUMENT:
{doc}

Respond in JSON:
{
  "High": ["..."],
  "Medium": ["..."],
  "Low": ["..."],
  "Obligations": ["..."]
}""",
        input_variables=["doc"],
    )
    chain = _make_llm_chain(llm, prompt)
    try:
        if hasattr(chain, "invoke"):
            raw = chain.invoke({"doc": text[:8000]})
            raw = raw.get("text", str(raw))
        else:
            raw = chain.run({"doc": text[:8000]})
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
        return {"High": [], "Medium": [], "Low": []}, [
            "Manual review recommended due to analysis limitations"
        ]

# -------------------------------------------------------------------------
# Date utilities, compare, etc.
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

def compare_documents(text1: str, text2: str) -> str:
    a_lines, b_lines = text1.splitlines(), text2.splitlines()
    diff = difflib.unified_diff(a_lines, b_lines, fromfile="Document A", tofile="Document B", lineterm="")
    md_lines = ["### Document Comparison\n", "```diff"]
    md_lines.extend(list(diff))
    md_lines.append("```")
    return "\n".join(md_lines)

# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("Backend module loaded (FAISS, OpenRouter).")
    print("Model ID:", MODEL_ID)
