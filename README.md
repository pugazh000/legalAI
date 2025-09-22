# ⚖️ Legal Simplifier AI

An AI-powered tool to make legal documents clear, actionable, and easy to understand.

## ✨ Features

* **📤 Document Upload**: Upload contracts in PDF or DOCX format.
* **📝 Plain-Language Summaries**: Simplify complex legal text into clear English.
* **🔍 Semantic Q\&A**: Ask natural language questions about your contract and get context-aware answers.
* **⚠️ Risk & Obligation Analysis**: Automatically detect risks, obligations, and key action items.
* **📅 Smart Date Extraction & Validation**: Highlight important contract dates (expiry, deadlines, renewals).
* **📄 Document Comparison**: Compare two contracts with a markdown-style redline of changes.
* **🌐 User-Friendly Interface**: Built with Streamlit for ease of use — no coding or legal background required.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/pugazh000/legalAI.git
cd legalAI
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Create a `.env` file in the project root with your API credentials:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
MODEL_ID=x-ai/grok-4-fast:free
```

### 4. Run the App

```bash
streamlit run frontend.py
```

### 5. Deploy on Render (Optional)

You can also deploy using Render. Once deployed, access it at:

```
https://your-app-name.onrender.com
```

---

## 📂 Sample File

A sample contract (`Contract document.pdf`) is included.

To test:

1. Launch the app.
2. Upload the sample contract.
3. Explore summaries, risks, obligations, and Q\&A.

---

## 🛠️ Usage Workflow

1. **Upload your contract** (or use the sample provided).
2. **Review the summary** for purpose, obligations, and deadlines.
3. **Ask natural questions** about the contract for clarity.
4. **Compare two versions** of a contract to spot differences.

---

## 🧰 Tech Stack

* **Python**, **Streamlit**
* **LangChain**, **Hugging Face Transformers**, **FAISS** (vector DB)
* **OpenRouter API** (via `langchain-openai`)
* **pdfplumber**, **python-docx**

---

## ⚠️ Notes

* Requires **Python 3.10+**.
* Best used for legal education, clarity, and drafting assistance.
* ⚠️ Not a substitute for professional legal advice.
* Avoid uploading sensitive or confidential contracts in production.

---

## 📜 License

This project is licensed under the **MIT License**.

---

🙌 Feel free to fork, modify, and contribute!
