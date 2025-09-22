"""
Streamlit frontend for "⚖ The Legal Simplifier AI".

Run:
    streamlit run frontend.py

Requirements:
    pip install -r requirements.txt
"""

import streamlit as st

from backend import (
    parse_file,
    chunk_and_store,
    summarize_text,
    safe_analyze_document_for_risks,
    semantic_search,
    intelligent_date_extraction_and_validation,
    semantic_search_with_intelligent_validation,
    compare_documents,
    simple_date_validator,
    semantic_search_with_dates,
)

st.set_page_config(page_title="⚖ The Legal Simplifier AI", layout="wide")

st.title("⚖ The Legal Simplifier AI")
st.write("An AI-powered tool to make legal documents understandable and actionable.")

# Ensure session_state defaults
if "doc1_text" not in st.session_state:
    st.session_state.doc1_text = ""
if "doc2_text" not in st.session_state:
    st.session_state.doc2_text = ""
if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False
if "collection_name" not in st.session_state:
    st.session_state.collection_name = "legal_docs"

# --- Section 1: Upload and Analyze Documents
st.header("📤 Upload & Analyze Documents")
uploaded_file = st.file_uploader("Upload primary document (.pdf or .docx)", type=["pdf", "docx"], key="u1")
if uploaded_file:
    with st.spinner("Parsing file..."):
        text = parse_file(uploaded_file)
        st.session_state.doc1_text = text
    st.success("File parsed and stored in memory.")
    st.text_area("Parsed text (primary)", value=st.session_state.doc1_text[:4000], height=200)

    # Auto-ingest into Chroma right after parsing
    with st.spinner("Chunking, embedding, and storing into vector store..."):
        chunk_and_store(
            text=st.session_state.doc1_text,
            collection_name=st.session_state.collection_name,
            chunk_size=500,
            chunk_overlap=50,
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        )
        st.session_state.vectorstore_ready = True
    st.success("Document ingested into vector store. Ready for semantic queries.")

# --- Section 2: Plain Language Summary
st.header("📑 Plain Language Summary")
if st.button("Generate Summary"):
    if not st.session_state.doc1_text:
        st.error("Please upload and parse a primary document first.")
    else:
        with st.spinner("Generating simplified summary..."):
            summary = summarize_text(st.session_state.doc1_text)
        st.info(summary)

# --- Section 3: Risk & Obligation Analysis
st.header("⚠️ Risk & Obligation Analysis")
if st.button("Analyze Risks & Obligations"):
    if not st.session_state.doc1_text:
        st.error("Please upload and parse a primary document first.")
    else:
        with st.spinner("Analyzing risks and obligations..."):
            risks, obligations = safe_analyze_document_for_risks(st.session_state.doc1_text)

        st.subheader("Risks")
        for level, items in risks.items():
            if items:
                if level == "High":
                    st.error(f"**High Risks:**\n- " + "\n- ".join(items))
                elif level == "Medium":
                    st.warning(f"**Medium Risks:**\n- " + "\n- ".join(items))
                else:
                    st.info(f"**Low Risks:**\n- " + "\n- ".join(items))

        st.subheader("Obligations")
        if obligations:
            st.success("**Obligations:**\n- " + "\n- ".join(obligations))
        else:
            st.info("No clear obligations detected.")

# --- Section: Enhanced Search with Date Validation
st.header("🔍 Smart Validity Check")
st.caption("Ask about document validity - now with intelligent date analysis!")

validity_query = st.text_input("Ask about validity:", value="Is my insurance valid?", key="validity_search")

col1, col2 = st.columns(2)
with col1:
    if st.button("🚀 Check Validity", type="primary"):
        if not st.session_state.vectorstore_ready:
            st.error("Please upload and process a document first.")
        else:
            with st.spinner("Analyzing document with date intelligence..."):
                try:
                    answer = semantic_search_with_dates(
                        query=validity_query,
                        collection_name="legal_docs",
                        top_k=5
                    )
                    if "🔴 NO" in answer or "NOT VALID" in answer:
                        st.error(answer)
                    elif "🟢 YES" in answer or "VALID" in answer:
                        st.success(answer)
                    elif "🟡" in answer or "EXPIRING" in answer:
                        st.warning(answer)
                    else:
                        st.info(answer)
                except Exception as e:
                    st.error(f"Enhanced search failed: {str(e)}")
                    st.info("Try the regular semantic search above.")

with col2:
    if st.button("📅 Quick Date Check"):
        if not st.session_state.doc1_text:
            st.error("Please upload a document first.")
        else:
            with st.spinner("Extracting dates..."):
                try:
                    results = simple_date_validator(st.session_state.doc1_text)
                    if results["status"] == "EXPIRED":
                        st.error(f"❌ **EXPIRED**: {results['message']}")
                    elif results["status"] == "VALID":
                        st.success(f"✅ **VALID**: {results['message']}")
                    elif results["status"] == "EXPIRING_SOON":
                        st.warning(f"⚠️ **EXPIRING SOON**: {results['message']}")
                    else:
                        st.info(f"ℹ️ {results['message']}")
                    if results["action_needed"]:
                        st.write("**Actions needed:**")
                        for action in results["action_needed"]:
                            st.write(f"• {action}")
                except Exception as e:
                    st.error(f"Date check failed: {str(e)}")

# --- Section 4: Semantic Search
st.header("🔍 Semantic Search")
query_input = st.text_input("Enter your question", value="What are my obligations?")
if st.button("Search"):
    if not st.session_state.vectorstore_ready:
        st.error("Please ingest the primary document (chunk & store) before semantic searching.")
    else:
        with st.spinner("Running semantic search and generating final answer..."):
            answer = semantic_search(
                query=query_input,
                collection_name="legal_docs",
                top_k=5
            )
        st.info(answer)

# --- Section 5: Compare Documents
st.header("📑 Compare Two Documents")
uploaded_file2 = st.file_uploader("Upload secondary document (.pdf or .docx)", type=["pdf", "docx"], key="u2")
if uploaded_file2:
    with st.spinner("Parsing second file..."):
        text2 = parse_file(uploaded_file2)
        st.session_state.doc2_text = text2
    st.success("Second file parsed and stored in memory.")
    st.text_area("Parsed text (secondary)", value=st.session_state.doc2_text[:4000], height=200)

if st.button("Compare Documents"):
    if not st.session_state.doc1_text or not st.session_state.doc2_text:
        st.error("Please upload both documents first.")
    else:
        diff_report = compare_documents(st.session_state.doc1_text, st.session_state.doc2_text)
        st.markdown(diff_report)

# --- Section 6: Download Placeholder
st.header("⬇️ Download")
st.write("Download functionality scaffolded (not wired to actual file outputs in this demo).")
