# 🔍 Financial RAG — Semantic Search Q&A System with ML Re-ranking

A production-ready **Two-Stage Retrieval-Augmented Generation (RAG)** pipeline designed for intelligent question answering over complex financial documents (SEC 10-K filings). 

This project combines state-of-the-art Natural Language Processing (**Sentence Transformers**), scalable vector storage (**ChromaDB**), and a custom **Logistic Regression Re-ranker** to deliver highly accurate context retrieval for financial queries.

---

## 🏗️ Architecture: Two-Stage Retrieval Pipeline

```mermaid
graph TD
    A[SEC 10-K Reports .txt] --> B[Text Cleaning & Chunking]
    B -->|1000 chars, 100 overlap| C[Sentence Transformers]
    C -->|all-MiniLM-L6-v2| D[(ChromaDB Vector Store)]
    
    Q[User Query] --> E[Query Embedding]
    E --> D
    
    D -->|Stage 1: Top-K Semantic Search| F[Candidate Chunks]
    F -->|Feature Extraction| G[Logistic Regression Re-ranker]
    G -->|Stage 2: Top-N Re-ranked| H[Final Context for LLM]
1. Document Processing & IngestionSource Data: SEC 10-K Filings (Apple, Microsoft, Tesla, etc.)Chunking Strategy: 1000-character windows with 100-character overlap to preserve semantic context boundaries.Embeddings: all-MiniLM-L6-v2 generating 384-dimensional dense vectors.Storage: Persistent ChromaDB collection using Cosine Similarity index.2. Two-Stage Retrieval (The Core)To solve the problem of generic semantic search failing on highly specific financial metrics, this project implements a two-stage retrieval:Dense Retrieval (Recall): Fetches the top 20 candidate chunks using cosine similarity from ChromaDB.ML Re-ranking (Precision): A trained Logistic Regression classifier evaluates the candidates. It uses engineered features (e.g., financial_score to detect dollar amounts and percentages, alongside raw semantic distance) to re-rank the chunks, ensuring mathematically heavy paragraphs surface to the top for financial questions.📊 Performance & Market AlignmentBased on recent ML / AI Engineer market analysis, this project demonstrates highly demanded skills:NLP & LLM ecosystems (Embeddings, RAG architectures).Vector Databases (ChromaDB integration).Classic Machine Learning applied to text (scikit-learn, Logistic Regression feature importance).MetricsMetricValueDocument typeSEC 10-K FilingsChunk size1000 chars (100 overlap)Embedding modelall-MiniLM-L6-v2Re-rankerLogistic Regression (scikit-learn)Vector DBChromaDB🛠️ Tech StackComponentTechnologyLanguagePython 3.10+Vector StoreChromaDBEmbeddingssentence-transformersMachine Learningscikit-learn, numpy, pandasVisualizationmatplotlib, seaborn (for Feature Importance)🚀 Quick Start1. Clone the repositoryBashgit clone https://github.com/yaroslavfedchenko11-rgb/financial-rag.git
cd financial-rag
2. Install dependenciesBashpip install -r requirements.txt
(Requires: chromadb, sentence-transformers, scikit-learn, pandas, jupyterlab)3. Run the pipelineLaunch Jupyter to explore the end-to-end pipeline:Bashjupyter notebook notebooks/financial_rag_semantic_final.ipynb
📁 Project StructurePlaintextfinancial-rag/
├── data/
│   └── financial_docs/       # Place your SEC 10-K .txt files here
├── notebooks/
│   └── financial_rag_semantic_final.ipynb  # Main pipeline & experiments
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies
💡 Key HighlightsSmart Data Cleaning: Custom RegEx filters to remove SEC boilerplate (TOC, page numbers).Feature Engineering: Visualized feature importance (coefficients) to prove why the Re-ranker prioritizes certain chunks over others.Metadata Filtering: Supports querying specific companies directly in ChromaDB (where={"ticker": "AAPL"}).