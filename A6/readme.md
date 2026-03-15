# NLP Assignment A6 — Retrieval Augmented Generation Techniques

**Student ID:** st125989  
**Course:** Natural Language Processing  
**Assignment:** A6 – RAG Techniques  
**Chapter:** Chapter 9  

---

# Project Overview

This project implements and evaluates **Retrieval Augmented Generation (RAG)** techniques using a textbook chapter as the knowledge source.

The assignment focuses on:

1. Preparing a QA dataset from the chapter
2. Implementing a **Naive RAG pipeline**
3. Implementing **Contextual Retrieval**
4. Comparing both approaches using **ROUGE evaluation**
5. Building a **web application** that allows users to ask questions about the chapter.

The system retrieves relevant information from the chapter and generates answers using a **local Large Language Model (LLM)**.

---

# Project Structure
A6
│
├── app
│ ├── app.py # Dash web application
│ └── utils.py # Retrieval and generation utilities
│
├── data
│ └── chapter_9.pdf # Source chapter
│
├── outputs
│ ├── task1
│ │ ├── chapter_9_clean.txt
│ │ ├── chapter_9_paragraphs.json
│ │ ├── chapter_9_qa_pairs.json
│ │ └── chapter_9_ground_truth_qa.json
│ │
│ └── task2
│ ├── naive_rag_outputs.json
│ ├── contextual_chunks.json
│ ├── contextual_rag_outputs.json
│ └── response-st125989-chapter-9.json
│
├── a6.ipynb # Main notebook implementation
└── README.md

---

# Task 1 — Dataset Preparation

The first task creates a **Question Answer dataset** from Chapter 9.

### Steps

1. Extract raw text from the chapter PDF
2. Clean and normalize the text
3. Split the chapter into paragraphs
4. Automatically generate question-answer pairs
5. Create the **ground truth QA dataset**

### Output Files

```
chapter_9_clean.txt
chapter_9_paragraphs.json
chapter_9_qa_pairs.json
chapter_9_ground_truth_qa.json
```

This dataset is used to evaluate the RAG systems in Task 2.

---

# Task 2 — RAG Pipeline Implementation

Two retrieval pipelines were implemented.

## 1. Naive RAG

Naive RAG uses basic chunking and semantic retrieval.

### Pipeline

```
Question
   ↓
Sentence Transformer Embedding
   ↓
FAISS Vector Search
   ↓
Top-K Chunk Retrieval
   ↓
Prompt Construction
   ↓
LLM Answer Generation
```

### Model Components

Embedding Model

```
sentence-transformers/all-MiniLM-L6-v2
```

Language Model

```
Qwen/Qwen2.5-1.5B-Instruct
```

Vector Database

```
FAISS
```

---

## 2. Contextual Retrieval

Contextual Retrieval improves RAG by adding **additional context to each chunk before embedding**.

Instead of embedding raw chunks only, the system generates **context prefixes** describing the chunk.

Example

```
Context Prefix:
This section explains how instruction tuning improves the ability of LLMs to follow user instructions.

Chunk:
Instruction tuning fine-tunes a pretrained model on instruction-response datasets.
```

This improves retrieval accuracy because the embedding better captures the semantic meaning of the text.

---

# Evaluation

Both pipelines were evaluated using **ROUGE metrics**.

Metrics used

- ROUGE-1
- ROUGE-2
- ROUGE-L

The generated answers were compared with the **ground truth answers**.

---

# Results

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |
|------|------|------|------|
| Naive RAG | 0.2154 | 0.1024 | 0.1915 |
| Contextual Retrieval | **0.2324** | **0.1445** | **0.2197** |

### Analysis

Contextual Retrieval achieved higher scores across all metrics.

Reasons:

- Context prefixes improve semantic chunk representations
- Retrieval quality increases
- Better context leads to more accurate answer generation

This demonstrates that **contextualized embeddings improve RAG performance**.

---

# Task 3 — Web Application

A web interface was implemented using **Dash**.

The application allows users to:

- Ask questions about Chapter 9
- Retrieve relevant contextual chunks
- Generate answers using the RAG pipeline
- View the source chunks used for the answer

---

# Web App Features

- Interactive question input
- Adjustable Top-K retrieval slider
- LLM generated answers
- Source chunk transparency
- Modern chatbot style interface
- Collapsible chunk explanations

---

# Example Questions

Examples that work well with the system:

```
What is the goal of instruction tuning?
What is preference alignment?
What is the key insight of Direct Preference Optimization?
What aspects are used to evaluate system outputs?
```

---

# Running the Web Application

Install dependencies:

```bash
pip install dash sentence-transformers faiss-cpu transformers accelerate torch numpy
```

Run the app:

```bash
python app/app.py
```

Open the browser:

```
http://127.0.0.1:8050
```

---

# Technologies Used

| Component | Tool |
|--------|--------|
| Language | Python |
| LLM | Qwen2.5 |
| Embeddings | Sentence Transformers |
| Vector Search | FAISS |
| Evaluation | ROUGE |
| Web App | Dash |
| Data Processing | Python / JSON |

---

# Key Learnings

This assignment demonstrates:

- How Retrieval Augmented Generation works
- The importance of retrieval quality in RAG systems
- How contextual embeddings improve semantic search
- Evaluation of generative QA systems
- Integration of LLM pipelines into a web application

---

# Conclusion

The project successfully implemented two RAG systems and demonstrated that **Contextual Retrieval improves answer generation quality**.

The final system provides an interactive interface where users can query textbook knowledge using modern RAG techniques.

---

