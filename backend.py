"""
Backend logic for the Legal Document Simplifier AI.

This version uses OpenRouter API (via langchain_openai.ChatOpenAI)
instead of Google Gemini.
"""
import json
import os
import difflib
import re
import io
from typing import Tuple, Dict, List, Any
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date
from dateutil.relativedelta import relativedelta

from dotenv import load_dotenv
load_dotenv()

VECTORSTORE_PERSIST_DIR = os.getenv("VECTORSTORE_PERSIST_DIR", "./chroma_db")

# -------------------------------------------------------------------------
# LangChain / Embeddings / Vector DB / LLM
# -------------------------------------------------------------------------
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings   # ✅ updated
from langchain_community.vectorstores import Chroma                # ✅ updated
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI

# File parsing
import pdfplumber
import docx
import numpy as np

# -------------------------------------------------------------------------
# LLM configuration (OpenRouter)
# -------------------------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("GEMINI_API_KEY")  # OpenRouter key
MODEL_ID = os.getenv("MODEL_ID", "x-ai/grok-4-fast:free")

if not OPENROUTER_API_KEY:
    print("Warning: OpenRouter API key not set. LLM calls will fail.")

def _get_llm(model_name: str = None, temperature: float = 0.0):
    """Return a ChatOpenAI client configured for OpenRouter."""
    if model_name is None:
        model_name = MODEL_ID
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
# Vector store
# -------------------------------------------------------------------------
def _get_embedding_model(embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    return HuggingFaceEmbeddings(model_name=embedding_model_name)

def chunk_and_store(
    text: str,
    collection_name: str = "legal_docs",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    persist_directory: str = VECTORSTORE_PERSIST_DIR,
):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(text)
    docs = [Document(page_content=c, metadata={"chunk_id": str(i), "source": collection_name}) for i, c in enumerate(chunks)]
    embedder = _get_embedding_model(embedding_model_name)
    chroma_db = Chroma.from_documents(
        documents=docs,
        embedding=embedder,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    chroma_db.persist()
    return chroma_db

# -------------------------------------------------------------------------
# Core RAG functions
# -------------------------------------------------------------------------
def semantic_search(
    query: str,
    collection_name: str = "legal_docs",
    top_k: int = 5,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    persist_directory: str = VECTORSTORE_PERSIST_DIR,
    model_name: str = None,
) -> str:
    embedder = _get_embedding_model(embedding_model_name)
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedder, collection_name=collection_name)
    docs = vectordb.similarity_search(query, k=top_k)
    ctxs = [f"--- chunk {i} from {d.metadata.get('source','')} ---\n{d.page_content.strip()[:2000]}" for i, d in enumerate(docs)]
    context_text = "\n\n".join(ctxs)
    llm = _get_llm(model_name=model_name, temperature=0.0)
    prompt_template = """You are a legal assistant. Use these retrieved chunks to answer:

RETRIEVED CHUNKS:
{context}

QUESTION:
{question}

Answer clearly in plain English for a non-lawyer.
"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({"context": context_text, "question": query})

def summarize_text(text: str, model_name: str = None) -> str:
    llm = _get_llm(model_name=model_name, temperature=0.0)
    prompt_template = """Summarize the document in plain language.
Highlight purpose, key obligations, important dates, and immediate actions.

DOCUMENT:
{doc}"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["doc"])
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({"doc": text})

def analyze_document_for_risks(text: str, model_name: str = None) -> Tuple[Dict[str, List[str]], List[str]]:
    llm = _get_llm(model_name=model_name, temperature=0.0)
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
    chain = LLMChain(llm=llm, prompt=prompt)
    raw = chain.run({"doc": text[:8000]})
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.splitlines()[1:-1])
        parsed = json.loads(re.search(r"\{.*\}", cleaned, re.DOTALL).group(0))
    except Exception as e:
        print("Risk parsing failed:", e)
        parsed = {"High": [], "Medium": [], "Low": [], "Obligations": []}
    return {k: parsed.get(k, []) for k in ["High", "Medium", "Low"]}, parsed.get("Obligations", [])

# -------------------------------------------------------------------------
# Safe wrapper (prevents frontend import errors)
# -------------------------------------------------------------------------
def safe_analyze_document_for_risks(text: str, model_name: str = None) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Wrapper with fallback if risk analysis fails.
    """
    try:
        return analyze_document_for_risks(text, model_name)
    except Exception as e:
        print(f"Analysis failed with error: {e}")
        risks = {"High": [], "Medium": [], "Low": []}
        obligations = []
        text_lower = text.lower()
        if 'expir' in text_lower:
            risks["High"].append("Document may contain expired items - manual review required")
        if any(word in text_lower for word in ['breach', 'default', 'violation']):
            risks["High"].append("Breach or default language detected")
        if any(word in text_lower for word in ['penalty', 'fine']):
            risks["High"].append("Penalty clauses identified")
        if any(word in text_lower for word in ['payment', 'fee', 'cost']):
            risks["Medium"].append("Financial obligations present")
        if any(word in text_lower for word in ['comply', 'regulation', 'requirement']):
            risks["Medium"].append("Compliance requirements indicated")
        if any(word in text_lower for word in ['report', 'submission']):
            risks["Medium"].append("Reporting obligations may apply")
        if any(word in text_lower for word in ['shall', 'must', 'required']):
            obligations.append("Document contains mandatory obligations - review recommended")
        obligations.append("Manual review recommended due to analysis limitations")
        return risks, obligations

# -------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------
def compare_documents(text1: str, text2: str) -> str:
    a_lines, b_lines = text1.splitlines(), text2.splitlines()
    diff = difflib.unified_diff(a_lines, b_lines, fromfile="Document A", tofile="Document B", lineterm="")
    md_lines = ["### Document Comparison\n", "```diff"]
    md_lines.extend(list(diff))
    md_lines.append("```")
    return "\n".join(md_lines)

def embed_texts(texts: List[str], embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    embedder = _get_embedding_model(embedding_model_name)
    return embedder.embed_documents(texts)

# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("Backend module loaded. VECTORSTORE persist dir:", VECTORSTORE_PERSIST_DIR)
    print("Using model:", MODEL_ID)
