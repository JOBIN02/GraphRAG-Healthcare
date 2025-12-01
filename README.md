# GraphRAG Healthcare – Diabetes Awareness, Treatment & Management

AI-powered medical Q&A system using **GraphRAG**, **Knowledge Graphs**, and **LLM agents** for clinically-safe, explainable answers.

---

## 🔍 1. System Architecture Overview

The system combines vector retrieval + knowledge graph reasoning + multi-agent workflow:

* **Document Preprocessing** → chunking, cleaning, entity extraction
* **Embedding Generation** → FAISS vector index
* **Knowledge Graph (KG)** → medical entities + relationships
* **GraphRAG Retrieval** → multi-hop graph search + semantic retrieval
* **LLM Agent Pipeline** → intent → retrieval → summarization → graph-reasoning → answer
* **REST API** → exposes the `/query` endpoint
* **UI (optional)** → Streamlit/React frontend

---

## 🌐 2. Domain & Dataset Source

### **Domain Chosen**

Healthcare — **Diabetes Awareness, Treatment & Management**

### **Dataset Sources Used**

* WHO — Diabetes Fact Sheets
* NIH — Treatment Guidelines
* ADA — Standards of Care
* MedlinePlus — Patient Education
* Additional public health articles, clinical notes, medical descriptions

All downloaded material was cleaned, converted to text, and stored under:

```
data/raw/
```

---

## 🧠 3. Knowledge Graph Structure

The KG contains entities such as:

* **Symptoms**
* **Causes**
* **Medications** (e.g., Metformin)
* **Lifestyle Factors**
* **Complications**
* **Diagnosis Methods**

Relationships include:

* *treats*, *causes*, *associated_with*, *reduces_risk*, *belongs_to*, etc.

Graph stored under:

```
data/graphs/
```

---

## 🔎 4. RAG / GraphRAG Design

### **Standard RAG (Vector-based)**

* Uses **FAISS**
* Retrieves top-k similar chunks
* Good for general Q&A but limited for reasoning

### **GraphRAG (Enhanced)**

Enhances retrieval using graph reasoning:

* Entity extraction from queries
* Multi-hop graph traversal
* Neighbor expansion
* Structural + semantic context combined

This improves **accuracy**, **explainability**, and **clinical safety**.

---

## 🤖 5. Agent Workflow

The AI agent follows a multi-module pipeline:

### **1. Intent Classification**

Detects type of query:

* Symptoms
* Diagnosis
* Treatment
* Medication
* Lifestyle
* Complications
* General medical info

### **2. GraphRAG Retrieval**

Pulls:

* Vector semantic hits
* Graph neighbors
* Entity relationships
* Multi-hop connections

### **3. Summarizer (LLM)**

Compresses retrieved text into a medically correct summary.

### **4. Graph Reasoning Module**

Analyzes KG structure to enhance reasoning.

### **5. Final Answer Generator**

Combines:

* Query
* Summaries
* Graph context
* Medical reasoning

Produces a **clinically-safe final answer**.

---

## 🛠️ 6. API Endpoints

### **POST /query**

Retrieves GraphRAG output + final answer.

#### **Request**

```json
{
  "query": "How does metformin work?",
  "k": 5,
  "hops": 2
}
```

#### **Response**

```json
{
  "intent": "treatment",
  "retrieval": {
    "vector_hits": [...],
    "graph_nodes": {...}
  },
  "summary": "Metformin reduces hepatic glucose production...",
  "final": "Metformin works by..."
}
```

---

## ▶️ 7. Setup & Run Instructions

### **Step 1 — Initialize Git**

```
git init
git remote add origin https://github.com/JOBIN02/GraphRAG-Healthcare.git
```

### **Step 2 — Create `.gitignore`**

```
# Python
__pycache__/
*.pyc

# Virtual env
.venv/

# Environment variables
.env

# Data
data/raw/*
data/embeddings/*
data/graphs/*
data/processed/*

# Streamlit
.streamlit/

# VSCode
.vscode/
```

### **Step 3 — Add Files**

```
git add .
```

### **Step 4 — Commit**

```
git commit -m "Initial commit: GraphRAG Healthcare system"
```

### **Step 5 — Push to GitHub**

```
git branch -M main
git push -u origin main
```

---

## 💬 8. Sample Queries & Answers

### **Query: “What are the symptoms of diabetes?”**

**Answer:**

* Excessive thirst
* Frequent urination
* Fatigue
* Blurred vision
  *Graph: Symptoms cluster*

---

### **Query: “How does metformin control blood sugar?”**

**Answer:**
Metformin:

* Reduces liver glucose production
* Improves insulin sensitivity
* Enhances glucose uptake
  *Graph: Metformin → Glucose → Insulin Resistance*

---

### **Query: “What lifestyle changes prevent diabetes?”**

**Answer:**

* Weight management
* Daily physical activity
* Balanced meals
* Reduced sugar consumption
  *Graph: Obesity → Diabetes Risk → Lifestyle*

---

## ⚠️ 9. Challenges & Learnings

### **Challenges**

* Gemini API key misconfigurations
* spaCy model download failures
* Circular imports across modules
* NetworkX version issues
* Vector/KG node mismatches
* LLM hallucination control
* API 500 errors from JSON formatting

### **Learnings**

* GraphRAG greatly improves accuracy
* Multi-hop reasoning is crucial for healthcare
* Clean chunking boosts retrieval performance
* Structured outputs reduce hallucinations
* Modular architecture simplifies debugging

---

## 🌍 10. Public Repository

**GitHub Link:**
[https://github.com/JOBIN02/GraphRAG-Healthcare](https://github.com/JOBIN02/GraphRAG-Healthcare)


