# 📊 Finance RAG Assistant

An advanced **Production-Style Retrieval-Augmented Generation (RAG) System** for financial and business document analysis using:

* Hybrid Retrieval (BM25 + Vector Search)
* Cross-Encoder Re-ranking
* Gemini AI
* ChromaDB
* Conversational Memory
* RAG Evaluation (RAGAS)
* Streamlit Web UI

---

# 🚀 Features

## ✅ Advanced RAG Pipeline

* PDF document ingestion
* Intelligent chunking
* Embedding generation
* ChromaDB vector storage

## ✅ Hybrid Retrieval

Combines:

* **BM25 keyword retrieval**
* **Semantic vector search**

for improved accuracy and recall.

## ✅ Cross-Encoder Re-ranking

Uses:

* `cross-encoder/ms-marco-MiniLM-L-6-v2`

to improve retrieval precision by re-scoring retrieved chunks.

## ✅ Gemini AI Integration

Uses:

* `gemini-2.5-flash`

for fast and accurate answer generation.

## ✅ Conversational Memory

Supports:

* Follow-up questions
* Context-aware conversations
* Query rewriting

## ✅ Citation Enforcement

Every generated answer includes:

* Source document
* Page number citations

to reduce hallucinations and improve transparency.

## ✅ Evaluation Pipeline

Uses:

* RAGAS

to evaluate:

* Faithfulness
* Answer Relevancy
* Context Precision

---

# 🧠 Tech Stack

| Component  | Technology                          |
| ---------- | ----------------------------------- |
| LLM        | Gemini 2.5 Flash                    |
| Embeddings | BAAI/bge-base-en-v1.5               |
| Vector DB  | ChromaDB                            |
| Retrieval  | BM25 + Vector Search                |
| Re-ranking | Sentence Transformers Cross-Encoder |
| Framework  | LangChain                           |
| Evaluation | RAGAS                               |
| UI         | Streamlit                           |

---

# 📂 Project Structure

```bash
RAG PROJECT/
│
├── app.py
├── requirements.txt
├── .env
│
├── docs/
├── db/
│
├── rag_app/
│   ├── ingestion.py
│   ├── retrieval.py
│   ├── generation.py
│   ├── evaluation.py
│   ├── runtime.py
│   ├── schemas.py
│   ├── settings.py
│   ├── api.py
│   └── __init__.py
│
├── evaluation_dataset.json
├── evaluation_generated_rows.json
└── evaluation_results_scores.json
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/finance-rag-assistant.git

cd finance-rag-assistant
```

---

## 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file:

```env
GOOGLE_API_KEY=your_gemini_api_key
```

---

# 📥 Add Financial PDFs

Place your PDFs inside:

```bash
docs/
```

Example:

* Amazon Annual Reports
* Microsoft Annual Reports
* Tesla Reports
* Nvidia Reports
* Apple Reports

---

# 🏗️ Run Document Ingestion

```bash
python rag_app/ingestion.py
```

This will:

* Load PDFs
* Split chunks
* Generate embeddings
* Store vectors in ChromaDB

---

# 🔍 Run Retrieval Pipeline

```bash
python rag_app/retrieval.py
```

---

# 🤖 Run Answer Generation

```bash
python rag_app/generation.py
```

---

# 📊 Run Evaluation

```bash
python rag_app/evaluation.py
```

---

# 🌐 Launch Streamlit UI

```bash
streamlit run app.py
```

---

# 📈 Example Questions

* What future applications does Amazon envision for its same-day delivery infrastructure?
* How much did Microsoft pay to acquire GitHub?
* What is Prime Air?
* Who is ROBERT K. BURGESS?
* What was Microsoft's first hardware product release?

---

# 🎯 Key Highlights

✅ Production-style RAG architecture
✅ Hybrid retrieval pipeline
✅ Re-ranking for better precision
✅ Enterprise-style evaluation pipeline
✅ Hallucination reduction
✅ Citation-based answers
✅ Streamlit frontend

---

# 🔮 Future Improvements

 📈Multi-user Scalability
 📈Streaming Responses
 📈Authentication/Login

# 📜 License

This project is for educational and portfolio purposes.

---

# 👨‍💻 Author

Nirav Mistry

AI/ML Engineer | RAG Systems | LLM Applications | Generative AI
