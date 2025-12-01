# agent.py
import os
import json
import numpy as np
from typing import List, Dict, Any
# from pathlib import Path
from graphrag_core import embedder, init_faiss_index, CHUNKS_OUT, FAISS_INDEX_PATH, build_knowledge_graph, faiss_search_topk
# import faiss
# import networkx as nx
import google.generativeai as genai

# Load chunks metadata
with open(CHUNKS_OUT, "r", encoding="utf-8") as f:
    CHUNKS = json.load(f)

# load faiss
FAISS_INDEX = init_faiss_index(index_path=FAISS_INDEX_PATH)

# Build or load graph from chunks:
chunk_texts = [c["text"] for c in CHUNKS]
chunk_ids = [c["chunk_id"] for c in CHUNKS]
KG = build_knowledge_graph(chunk_texts, chunk_ids)

# ---------------------------
# Simple intent classifier
# ---------------------------
def classify_intent(query: str) -> str:
    """
    Very simple intent classifier:
      - looks for 'how', 'what' -> 'definition' or 'explain'
      - if 'through' or 'via' or 'affect' -> 'multi-hop'
      - 'compare' or 'vs' -> 'comparison'
      - 'how to', 'steps' -> 'procedure'
    This is a rule-based classifier; you can replace this with an LLM call for more accuracy.
    """
    q = query.lower()
    if any(w in q for w in ["how does", "how do", "how to", "how would", "affect", "through", "via"]):
        if "how does" in q and "through" in q:
            return "multi-hop"
        return "explain"
    if any(w in q for w in ["vs", "compare", "difference", "difference between"]):
        return "comparison"
    if any(w in q for w in ["step", "how to", "procedure", "steps"]):
        return "procedure"
    # fallback
    return "general"

# ---------------------------
# Graph-aware retrieval
# ---------------------------
def retrieve_with_graph(query: str, top_k=5, graph_expand_hops=1) -> Dict[str, Any]:
    """
    1) Embed query
    2) Use FAISS to find top_k relevant chunks
    3) Extract entities from query and find nearest nodes in KG (by string similarity)
    4) Graph-walk neighbors up to graph_expand_hops and add their doc chunks to context
    5) Return unique list of chunks & their meta
    """
    q_emb = embedder.encode([query], convert_to_numpy=True)[0].astype("float32")
    # normalize and search
    qnorm = q_emb / (np.linalg.norm(q_emb) + 1e-9)
    I, D = FAISS_INDEX.search(np.asarray([qnorm]), top_k)
    top_indices = I[0].tolist()
    scores = D[0].tolist()

    retrieved = []
    for idx, score in zip(top_indices, scores):

        # Convert index to int (FAISS sometimes returns floats)
        idx = int(idx)

        if 0 <= idx < len(CHUNKS):
            retrieved.append({
                "chunk_meta": CHUNKS[idx],
                "score": float(score)
            })
    # Graph expansion: find candidate nodes (naive string match)
    # Extract tokens/entities from query
    from graphrag_core import extract_entities_from_chunk
    q_entities = extract_entities_from_chunk(query)
    # find nearest nodes by substring / lowercase match
    nodes_to_add = set()
    for ent in q_entities:
        for node in KG.nodes():
            if ent.lower() in node.lower() or node.lower() in ent.lower() or ent.lower().split()[0] in node.lower():
                nodes_to_add.add(node)
    # expand neighbors
    expanded_doc_ids = set()
    for node in nodes_to_add:
        expanded_doc_ids.update(KG.nodes[node].get("doc_ids", []))
        # neighbors
        neighbors = list(KG.neighbors(node)) if graph_expand_hops >= 1 else []
        for nb in neighbors:
            expanded_doc_ids.update(KG.nodes[nb].get("doc_ids", []))
    # attach expanded chunks
    for cid in expanded_doc_ids:
        # find chunk meta with matching chunk_id
        for cm in CHUNKS:
            if cm["chunk_id"] == cid:
                retrieved.append({"chunk_meta": cm, "score": None})
                break
    # dedupe by chunk_id preserving order
    seen = set()
    uniq_retrieved = []
    for r in retrieved:
        cid = r["chunk_meta"]["chunk_id"]
        if cid not in seen:
            uniq_retrieved.append(r)
            seen.add(cid)
    # rank by score (None treated as 0)
    uniq_retrieved = sorted(uniq_retrieved, key=lambda x: x["score"] if x["score"] is not None else 0, reverse=True)
    return {
        "query": query,
        "intent": classify_intent(query),
        "retrieved": uniq_retrieved
    }

# ---------------------------
# RAG summarization & final synthesis
# ---------------------------
# The Gemini wrapper - you must supply your own implementation/keys.
# Below is a stub for how to call Gemini via HTTP or Vertex AI SDK.
# Replace with working code for your environment.

genai.configure(api_key="AIzaSyA5Qs9ca5a7mYGBtHMM6yejsrftgz4l_z8")
MODEL_NAME = "gemini-2.5-flash"   # or "gemini-2.0-flash", "gemini-pro", etc.

def call_gemini(prompt: str, max_tokens: int = 512) -> str:
    """
    Calls Gemini using your API key.
    Returns generated text ONLY (clean).
    """
    model = genai.GenerativeModel(MODEL_NAME)

    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.2
            # "top_p": 0.9,
            # "top_k": 40
        }
    )
    # print(response)
    return response.text

def summarize_retrieved(retrieved_items: List[Dict], query: str, max_context_chars=4000) -> str:
    """
    Build a summarization prompt from retrieved chunks and call Gemini (or your LLM).
    Concatenate top N chunks up to max_context_chars.
    """
    chunks = []
    for r in retrieved_items:
        txt = r["chunk_meta"]["text"]
        chunks.append(txt)
        if sum(len(c) for c in chunks) > max_context_chars:
            break
    context = "\n\n---\n\n".join(chunks)
    prompt = f"""
        You are a helpful assistant. Use ONLY the provided CONTEXT to answer the query. Do not invent facts.
        Query: {query}

        CONTEXT:
        {context}

        INSTRUCTIONS:
        1. Provide a concise but thorough answer grounded in context.
        2. If context is insufficient, say "I don't know — not enough information in the documents."
        3. Provide references to the chunk ids you used for each claim.

        Answer:
        """
    answer = call_gemini(prompt)
    return answer

# ---------------------------
# Full pipeline wrapper
# ---------------------------
def run_pipeline(query: str, top_k:int = 5, graph_expand_hops:int = 1):
    ret = retrieve_with_graph(query, top_k=top_k, graph_expand_hops=graph_expand_hops)
    summary = summarize_retrieved(ret["retrieved"], query)
    # optionally post-process: produce final synthesis
    final_prompt = f"""
        You are an expert summarizer. Given the following short answer (DELIVERABLE) and the ORIGINAL QUERY, refine and produce a final polished answer with bullets and citations to chunk IDs.

        QUERY: {query}

        DELIVERABLE:
        {summary}
        """
    final_answer = call_gemini(final_prompt)
    return {
        "intent": ret["intent"],
        "retrieved": ret["retrieved"],
        "summary": summary,
        "final_answer": final_answer
    }
