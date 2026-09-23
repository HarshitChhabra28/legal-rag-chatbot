# LexRAG — Corporate Law RAG Chatbot

> A citation-aware Retrieval-Augmented Generation (RAG) chatbot for answering questions from a curated corporate-law knowledge base.

LexRAG is a legal question-answering system built using **Hybrid Retrieval + Retrieval-Augmented Generation (RAG)**.

The system combines:

- **BGE embeddings** for semantic search
- **Milvus Lite** for vector retrieval
- **BM25** for keyword-based retrieval
- **Reciprocal Rank Fusion (RRF)** for hybrid ranking
- **Google Gemini** for grounded answer generation
- **FastAPI** for the backend API
- **HTML/CSS/JavaScript** for the frontend

The chatbot is designed to provide **source-grounded answers with citations** and avoid unsupported answers when the available context is insufficient.

---

## Demo

The chatbot can be run locally using FastAPI and a custom frontend.

### Local Demo

```text
Frontend
   ↓
FastAPI Backend
   ↓
Hybrid Retrieval
   ↓
Milvus + BM25
   ↓
RRF
   ↓
Google Gemini
   ↓
Answer + Citations
```

---

## Features

- Legal question answering using RAG
- Hybrid semantic + keyword retrieval
- BGE-based semantic embeddings
- Milvus vector search
- BM25 keyword search
- Reciprocal Rank Fusion (RRF)
- Citation-aware answer generation
- Source mapping for retrieved documents
- Context-grounded responses
- Insufficient-context safeguard
- Custom chatbot interface
- FastAPI REST API
- Gemini-powered generation
- Local execution without paid hosting

---

# System Architecture

```text
                         ┌─────────────────────┐
                         │      User Query     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    FastAPI API      │
                         │       /chat         │
                         └──────────┬──────────┘
                                    │
                   ┌────────────────┴────────────────┐
                   │                                 │
                   ▼                                 ▼
        ┌─────────────────────┐          ┌─────────────────────┐
        │   Semantic Search   │          │    BM25 Search      │
        │                     │          │                     │
        │ BGE Embeddings      │          │ Keyword Retrieval   │
        │ + Milvus            │          │                     │
        └──────────┬──────────┘          └──────────┬──────────┘
                   │                                 │
                   └────────────────┬────────────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Reciprocal Rank     │
                         │ Fusion (RRF)        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Top Retrieved       │
                         │ Legal Chunks        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Google Gemini     │
                         │ Grounded Generation │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Answer + Citations  │
                         └─────────────────────┘
```

---

# How It Works

## 1. Data Cleaning

The original legal datasets are cleaned and converted into a unified schema.

The normalized structure contains:

```text
id
doc_type
title
main_text
date
metadata
```

The dataset contains four document categories:

- Cases
- Articles
- Notifications
- Queries / FAQs

---

## 2. Document Chunking

Legal documents are divided into retrieval-friendly chunks.

Cases are split into smaller chunks, while articles, notifications and queries are generally kept as single chunks.

Final dataset:

| Document Type | Records | Chunks |
|---|---:|---:|
| Cases | 191 | 469 |
| Articles | 52 | 52 |
| Notifications | 213 | 213 |
| Queries / FAQs | 37 | 37 |
| **Total** | **493** | **771** |

Out of 191 case records:

```text
122 / 191
```

cases were split into more than one chunk.

---

## 3. Citation Formatting

Each chunk receives a document-specific citation.

### Case

```text
Party v. Party — Citation, Date [Bench: judges]
```

### Article

```text
"Title" by Author (Year)
```

### Notification

```text
Title — Notification No. X, dated Y
```

### Query / FAQ

```text
FAQ: "Title"
```

Citation metadata was validated before indexing.

```text
Missing / suspect metadata fields:
0 / 771
```

---

# Embedding Model

The project uses:

```text
BAAI/bge-base-en-v1.5
```

The embedding dimension is:

```text
768
```

The model is used to convert legal chunks and user queries into vector representations for semantic retrieval.

---

# Vector Database

The project uses:

```text
Milvus Lite
```

Collection:

```text
legal_chunks
```

The vector search uses:

```text
Cosine Similarity
```

The Milvus collection stores information such as:

- Chunk ID
- Document type
- Title
- Text
- Citation
- Embedding vector

---

# BM25 Retrieval

In addition to semantic retrieval, the system uses:

```text
rank-bm25
```

BM25 helps retrieve documents containing important exact legal terminology.

This is particularly useful for queries containing:

- Section numbers
- Case names
- Legal terms
- Notification numbers
- Specific phrases

---

# Hybrid Retrieval

LexRAG combines semantic retrieval and keyword retrieval.

```text
                 User Query
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
    BGE + Milvus             BM25
   Semantic Search       Keyword Search
          │                     │
          └──────────┬──────────┘
                     │
                     ▼
                    RRF
                     │
                     ▼
              Top 6 Chunks
```

Current configuration:

| Parameter | Value |
|---|---:|
| Semantic Retrieval | Top 10 |
| BM25 Retrieval | Top 10 |
| Final Context | Top 6 |
| RRF `k` | 60 |

---

# Reciprocal Rank Fusion

The semantic and BM25 ranked lists are combined using **Reciprocal Rank Fusion (RRF)**.

Conceptually:

```text
Semantic Ranking
       +
BM25 Ranking
       ↓
      RRF
       ↓
Unified Ranking
       ↓
Top 6 Chunks
```

This allows the system to combine two different retrieval signals without requiring their scores to be directly comparable.

---

# LLM Generation

The final retrieved context is passed to:

```text
Google Gemini
```

The generation stage receives the user's question together with the retrieved legal chunks.

The prompt instructs the model to:

- Answer only from the supplied context
- Cite substantive claims
- Use numbered citations
- Avoid unsupported legal claims
- Avoid inventing procedural details
- State when the context is insufficient
- Synthesize relevant information instead of simply listing documents

---

# Citation Mapping

The model produces numbered references such as:

```text
[1]
[2]
```

The application then maps these labels to the actual citation metadata.

For example:

```text
[1] "Consequences of Compounding of Offences"
    by T V Narayanaswamy (2025)
```

This approach keeps citation generation controlled by the application rather than requiring the LLM to generate citation metadata itself.

---

# Grounding and Hallucination Control

A major design goal of LexRAG is to prevent unsupported legal conclusions.

For example, if the retrieved context only states that Section 441 of the Companies Act, 2013 deals with compounding of offences but does not provide a detailed definition, the chatbot should not invent additional legal information.

Instead, it can respond that the provided context is insufficient.

Example:

```text
Based on the provided context, section 441 of the Companies Act,
2013 deals with the compounding of offences and its consequences.
However, the provided context does not contain a specific definition
or detailed explanation of what compounding of offences entails.
```

This makes the system **context-grounded rather than purely generative**.

---

# Dataset

The final unified dataset contains:

| Document Type | Records |
|---|---:|
| Cases | 191 |
| Articles | 52 |
| Notifications | 213 |
| Queries / FAQs | 37 |
| **Total** | **493** |

After chunking:

| Document Type | Chunks |
|---|---:|
| Cases | 469 |
| Articles | 52 |
| Notifications | 213 |
| Queries / FAQs | 37 |
| **Total** | **771** |

---

# Retrieval Evaluation

A dedicated evaluation script is included:

```text
09_evaluate.py
```

The evaluation contains:

```text
37 FAQ questions
```

For this evaluation setup, the corresponding source chunk was retrieved within the top 6 results for:

```text
37 / 37
```

### Retrieval Hit Rate

```text
100%
```

### Evaluation Limitation

The FAQ questions used for evaluation are themselves present in the indexed dataset.

Therefore, this result primarily demonstrates the correctness of the retrieval and indexing pipeline for the evaluation set.

It should **not** be interpreted as a general benchmark of legal question-answering performance on unseen questions.

---

# Project Structure

```text
legal-rag-chatbot/
│
├── database/
│   ├── clean_data.py
│   ├── chunk_data.py
│   ├── citation_formatter.py
│   ├── 04_embed_and_index.py
│   ├── 05_bm25_index.py
│   ├── 06_hybrid_retrieve_and_generate.py
│   └── 07_api.py
│
├── data/
│   └── raw datasets
│
├── frontend.html
│
├── requirements.txt
│
├── 08_test_questions.py
├── 09_evaluate.py
│
├── chunks.jsonl
├── chunks_with_citations.jsonl
├── unified_records.jsonl
│
├── bm25_index.pkl
├── bm25_chunks.pkl
│
├── law_project.db
│
└── README.md
```

---

# Tech Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Backend | FastAPI |
| Frontend | HTML, CSS, JavaScript |
| Embedding Model | BAAI/bge-base-en-v1.5 |
| Embedding Dimension | 768 |
| Vector Database | Milvus Lite |
| Keyword Retrieval | BM25 |
| Hybrid Ranking | Reciprocal Rank Fusion |
| LLM | Google Gemini |
| API Server | Uvicorn |
| Data Processing | Pandas |
| Version Control | Git + GitHub |

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/HarshitChhabra28/legal-rag-chatbot.git
```

Move into the project directory:

```bash
cd legal-rag-chatbot
```

---

## 2. Create a Virtual Environment

For Windows:

```bash
python -m venv venv
```

Activate the environment:

```bash
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Gemini API Configuration

The project uses Google Gemini for answer generation.

Set the API key as an environment variable.

### Windows CMD

```cmd
set GEMINI_API_KEY=YOUR_API_KEY
```

### Verify the Environment Variable

```cmd
python -c "import os; print(bool(os.getenv('GEMINI_API_KEY')))"
```

Expected:

```text
True
```

> Never commit your API key to GitHub or place it directly inside the Python source code.

---

# Running the Application

Start the FastAPI server:

```bash
uvicorn database.07_api:app --reload --port 8000
```

Open the chatbot in your browser:

```text
http://localhost:8000
```

Health check:

```text
http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "loaded": false
}
```

The retrieval resources are loaded when the first `/chat` request is received.

---

# API

## POST `/chat`

### Request

```json
{
  "query": "What is compounding of offences under Section 441 of the Companies Act?"
}
```

### Response

```json
{
  "answer": "Based on the provided context...",
  "sources": [
    {
      "label": 1,
      "citation": "\"Consequences of Compounding of Offences\" by T V Narayanaswamy (2025)"
    }
  ]
}
```

---

# Data Processing Pipeline

The complete data pipeline is:

```text
Raw Excel Files
       │
       ▼
Data Cleaning
       │
       ▼
Unified Records
       │
       ▼
Document Chunking
       │
       ▼
Citation Formatting
       │
       ▼
BGE Embeddings
       │
       ▼
Milvus Vector Index
       │
       ├──────────────────┐
       │                  │
       ▼                  ▼
   Vector Search       BM25 Index
       │                  │
       └────────┬─────────┘
                │
                ▼
               RRF
                │
                ▼
          Top 6 Chunks
                │
                ▼
         Gemini Generation
                │
                ▼
        Answer + Citations
```

---

# Retrieval Flow

For every user question:

```text
1. User enters a legal question
              ↓
2. Query is converted into a BGE embedding
              ↓
3. Milvus performs semantic search
              ↓
4. Query is searched using BM25
              ↓
5. Both ranked lists are combined using RRF
              ↓
6. Top 6 chunks are selected
              ↓
7. Retrieved context is provided to Gemini
              ↓
8. Gemini generates a grounded answer
              ↓
9. Citation labels are mapped to source metadata
              ↓
10. Answer + sources are returned to the frontend
```

---

# API Architecture

```text
Browser
   │
   │ POST /chat
   ▼
FastAPI
   │
   ▼
step6.ask()
   │
   ├── BGE Embedding
   │
   ├── Milvus Search
   │
   ├── BM25 Search
   │
   ├── RRF
   │
   └── Gemini
          │
          ▼
     Answer + Sources
```

---

# Why Hybrid Retrieval?

Legal information contains many exact terms that can be important for retrieval.

Examples include:

- `Section 441`
- Case names
- Notification numbers
- Statutory terminology
- Specific legal phrases

Semantic retrieval is useful for understanding meaning and related concepts.

BM25 is useful for exact keyword matching.

Therefore:

```text
Semantic Search
      +
Keyword Search
      ↓
Hybrid Retrieval
      ↓
Better Retrieval Coverage
```

---

# Why RAG?

A general-purpose LLM can generate answers using its pretrained knowledge.

However, a legal knowledge system may need to answer specifically from a controlled dataset.

RAG changes the generation process:

```text
Question
   ↓
Retrieve Relevant Documents
   ↓
Provide Retrieved Context
   ↓
Generate Answer From Context
```

This allows the system to:

- Ground responses in the available documents
- Provide source citations
- Reduce unsupported claims
- Explicitly report insufficient context

---

# Key Engineering Decisions

## BGE Base Embeddings

The project uses:

```text
BAAI/bge-base-en-v1.5
```

because it provides a practical balance between embedding quality and local resource requirements.

---

## Milvus

Milvus is used for vector similarity search over the indexed legal chunks.

---

## BM25

BM25 complements vector retrieval by providing lexical matching.

This is particularly useful in legal documents where exact terminology can be important.

---

## Reciprocal Rank Fusion

RRF combines the independently ranked semantic and BM25 results.

This avoids directly comparing incompatible score scales.

---

## Gemini

Gemini is used as the generation layer after retrieval.

The model receives retrieved context rather than being asked to answer purely from general knowledge.

---

# Backend

The backend is implemented using:

```text
FastAPI
```

Main API endpoint:

```text
POST /chat
```

Health endpoint:

```text
GET /health
```

Frontend endpoint:

```text
GET /
```

The frontend is served directly by FastAPI.

---

# Frontend

The frontend is a custom HTML/CSS/JavaScript interface.

The interface includes:

- Chat input
- User messages
- Assistant responses
- Typing/loading state
- Citation chips
- Source information
- Suggested questions
- Responsive dark UI

The frontend communicates with the FastAPI backend through:

```text
/chat
```

---

# Deployment

The project was tested for deployment using Render.

The application successfully completed the build stage, but the Render Free instance has a **512 MB RAM limit**.

The complete pipeline:

```text
BGE Embedding Model
        +
Milvus Lite
        +
BM25
        +
FastAPI
        +
Gemini Client
```

exceeded the available memory during deployment.

Therefore, the current portfolio demonstration is intended to run locally.

No paid hosting is required for the project demonstration.

---

# Limitations

- The chatbot is limited to the indexed legal dataset.
- It should not be treated as a substitute for professional legal advice.
- If relevant information is absent from the retrieved context, the system may report insufficient context.
- The current evaluation dataset contains questions represented in the indexed corpus.
- The current Milvus Lite setup is intended primarily for development/demo usage.
- The complete retrieval pipeline requires more memory than the Render Free 512 MB instance provides.
- The system has not been evaluated against a large unseen legal benchmark.

---

# Future Improvements

Potential future improvements include:

- Query rewriting
- Cross-encoder reranking
- Better metadata filtering
- More diverse evaluation datasets
- Evaluation on unseen questions
- Conversation memory
- Document-level retrieval
- Streaming responses
- Authentication
- API rate limiting
- Production-grade vector database
- Improved source inspection
- Automated retrieval metrics
- Better frontend source visualization

---

# Learning Outcomes

This project provided hands-on experience with:

- Retrieval-Augmented Generation
- Large Language Model integration
- Vector embeddings
- Vector databases
- Semantic search
- BM25 retrieval
- Hybrid retrieval
- Reciprocal Rank Fusion
- Prompt engineering
- Citation-aware generation
- Data cleaning
- Document chunking
- Metadata management
- FastAPI
- REST APIs
- Frontend-backend integration
- Git
- GitHub
- Local deployment

---

# Project Workflow

```text
                    DATA PIPELINE

Excel Dataset
     ↓
Data Cleaning
     ↓
Unified Schema
     ↓
Chunking
     ↓
Citation Formatting
     ↓
Embedding
     ↓
Milvus
     +
BM25
```

```text
                    QUERY PIPELINE

User Question
     ↓
BGE Embedding
     ↓
Milvus Search
     +
BM25 Search
     ↓
RRF
     ↓
Top 6 Context Chunks
     ↓
Gemini
     ↓
Grounded Answer
     ↓
Citation Mapping
     ↓
Frontend
```

---

# Example Query

### User

```text
What is compounding of offences under Section 441 of the Companies Act?
```

### System Behavior

The system retrieves relevant legal context and checks whether the retrieved information actually contains the required answer.

If the context only establishes that Section 441 deals with compounding of offences but does not provide a detailed definition, the system does not invent additional information.

Example:

```text
Based on the provided context, section 441 of the Companies Act,
2013 deals with the compounding of offences and its consequences.
However, the provided context does not contain a specific definition
or detailed explanation of what compounding of offences entails.
```

### Source

```text
"Consequences of Compounding of Offences"
by T V Narayanaswamy (2025)
```

---

# Repository

GitHub:

https://github.com/HarshitChhabra28/legal-rag-chatbot

---

# Author

## Harshit Chhabra

B.Tech Computer Science

This project was developed as a portfolio project to explore practical applications of:

- Generative AI
- Retrieval-Augmented Generation
- Legal document retrieval
- Vector databases
- Hybrid search
- LLM-based question answering

---

# Disclaimer

This project is developed for educational and portfolio purposes.

The chatbot provides answers based on the indexed dataset and should not be considered professional legal advice.

The system may return an insufficient-context response when the available dataset does not contain enough information to answer a question reliably.