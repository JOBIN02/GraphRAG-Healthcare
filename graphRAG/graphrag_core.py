# graphrag_core.py
import json
from typing import List, Dict, Tuple, Any
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
# from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import spacy
import re

nlp = spacy.load("en_core_web_sm")

# ---------------------------
# Configuration (edit paths)
# ---------------------------
DATA_DIR = Path("data/")          # where your .txt docs live
META_OUT = Path("data/metadata.json")  # metadata store
CHUNKS_OUT = Path("data/chunks.json")  # chunk -> meta
FAISS_DIR = Path("data/faiss_index")   # directory for faiss + meta
FAISS_DIR.mkdir(parents=True, exist_ok=True)
FAISS_INDEX_PATH = FAISS_DIR / "faiss.index"
EMB_DIM = 768   # dimension of chosen sentence-transformer (mpnet)
EMBED_MODEL_NAME = "all-mpnet-base-v2"
DUP_THRESHOLD = 0.97  # cosine similarity threshold to remove duplicates
CHUNK_TARGET_TOKENS = 500  # target tokens per chunk (range 300-800)
MIN_CHUNK_TOKENS = 150

# Initialize model
print("Loading SentenceTransformer:", EMBED_MODEL_NAME)
embedder = SentenceTransformer(EMBED_MODEL_NAME)


# ---------------------------
# 1) Metadata extraction
# ---------------------------
def extract_metadata_from_txt(file_path: Path) -> dict:
    """
    Heuristic metadata extraction from txt files:
      - title: first non-empty line
      - author: line starting with 'Author:' or 'By ' within first 10 lines
      - date: looks for YYYY or common date patterns within first 20 lines
      - URL: if present anywhere as http(s)
      - headers: lines that look like section headers (all-caps or ending with ':')
    """
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    lines = [ln.strip() for ln in text.splitlines()]
    # title
    title = next((ln for ln in lines if ln), "")
    # author heuristics
    author = ""
    for ln in lines[:12]:
        if ln.lower().startswith("author:"):
            author = ln.split(":", 1)[1].strip()
            break
        if ln.lower().startswith("by "):
            author = ln[3:].strip()
            break
    # date heuristics
    date = ""
    date_pattern = re.compile(r"\b(19|20)\d{2}\b")
    for ln in lines[:20]:
        m = date_pattern.search(ln)
        if m:
            date = m.group(0)
            break
    # url
    url_match = re.search(r"https?://\S+", text)
    url = url_match.group(0) if url_match else ""
    # headers: simple heuristics
    headers = [ln for ln in lines if (ln.isupper() and len(ln.split()) < 8) or ln.endswith(":")]
    return {
        "file": str(file_path),
        "title": title,
        "author": author,
        "date": date,
        "url": url,
        "headers": headers,
        "category": "",  # placeholder: you can label categories later
        "full_text": text
    }


# ---------------------------
# 2) Chunking (approx tokens)
# ---------------------------
def _approx_tokens_to_words(n_tokens: int) -> int:
    # Approx: 1 token ≈ 0.75 words (varies). We'll approximate tokens->words
    return int(n_tokens * 0.75)

def chunk_text(text: str, target_tokens: int = CHUNK_TARGET_TOKENS, min_tokens: int = MIN_CHUNK_TOKENS) -> List[str]:
    """
    Split text into chunks approximately target_tokens (approx token->words).
    This uses sentence boundaries to avoid chopping sentences.
    """
    target_words = _approx_tokens_to_words(target_tokens)
    words = text.split()
    if len(words) <= target_words:
        return [text.strip()]

    # naive sentence splitting by punctuation
    sentences = re.split(r'(?<=[\.\?\!])\s+', text)
    chunks = []
    current = []
    current_len = 0
    for sent in sentences:
        wl = len(sent.split())
        if current_len + wl <= target_words or current_len == 0:
            current.append(sent)
            current_len += wl
        else:
            chunk = " ".join(current).strip()
            if len(chunk.split()) >= min_tokens:
                chunks.append(chunk)
            else:
                # if too small, merge
                if chunks:
                    chunks[-1] += " " + chunk
                else:
                    chunks.append(chunk)
            current = [sent]
            current_len = wl
    if current:
        chunks.append(" ".join(current).strip())
    # final cleaning: remove empty
    chunks = [c for c in chunks if len(c.split()) >= 10]
    return chunks


# ---------------------------
# 3) Embedding generation & FAISS
# ---------------------------
def generate_embeddings(texts: List[str]) -> np.ndarray:
    """
    Use sentence-transformers embedder to produce embeddings matrix (n x d)
    """
    embs = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embs.astype("float32")


def init_faiss_index(dim: int = EMB_DIM, index_path: Path = FAISS_INDEX_PATH) -> faiss.IndexFlatIP:
    """
    Initialize or load FAISS index (Inner Product with normalized vectors)
    We'll use cosine by normalizing vectors and using inner product.
    """
    if index_path.exists():
        idx = faiss.read_index(str(index_path))
        print("Loaded existing FAISS index:", index_path)
        return idx
    idx = faiss.IndexFlatIP(dim)
    print("Created new FAISS IndexFlatIP (dim=%d)" % dim)
    return idx


def save_faiss_index(idx: faiss.IndexFlatIP, path: Path):
    faiss.write_index(idx, str(path))
    print("FAISS index saved to", path)


# ---------------------------
# 4) Deduplication
# ---------------------------
def deduplicate_embeddings(embs: np.ndarray, texts: List[str], threshold: float = DUP_THRESHOLD) -> Tuple[np.ndarray, List[str]]:
    """
    Remove near-duplicate chunks using cosine similarity.
    Strategy:
      - Normalize vectors, compute similarity matrix in blocks
      - If two items > threshold, drop the latter
    Note: O(n^2) worst-case. For 20-1000 chunks this is fine.
    """
    if len(embs) == 0:
        return embs, texts
    # normalize rows
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    embs_norm = embs / norms
    keep_mask = np.ones(len(embs), dtype=bool)
    for i in range(len(embs)):
        if not keep_mask[i]:
            continue
        # compute similarity to remaining
        sims = embs_norm[i+1:] @ embs_norm[i]
        dup_indices = np.where(sims >= threshold)[0]
        if dup_indices.size > 0:
            # set mask to False for duplicates (offset by i+1)
            keep_mask[dup_indices + i + 1] = False
    kept_embs = embs[keep_mask]
    kept_texts = [t for k, t in zip(keep_mask, texts) if k]
    print(f"Deduplication: kept {len(kept_texts)}/{len(texts)} chunks (threshold={threshold})")
    return kept_embs, kept_texts


# ---------------------------
# 5) Entity extraction & graph building
# ---------------------------
def extract_entities_from_chunk(text: str) -> List[str]:
    """
    Use spaCy to extract entities and noun chunks as candidate nodes.
    Return a list of strings (unique, normalized)
    """
    doc = nlp(text)
    ents = set()
    for ent in doc.ents:
        ents.add(ent.text.strip())
    # Include noun chunks to surface technical concepts
    for nc in doc.noun_chunks:
        txt = nc.text.strip()
        if len(txt.split()) <= 6 and len(txt) >= 3:
            ents.add(txt)
    # simple cleaning
    cleaned = []
    for e in ents:
        e2 = re.sub(r'\s+', ' ', e)
        e2 = e2.strip(" .,:;\"'()[]")
        if len(e2) > 2:
            cleaned.append(e2)
    return list(cleaned)


def build_knowledge_graph(chunks: List[str], chunk_ids: List[str]) -> nx.Graph:
    """
    Build a coarse KG using co-occurrence of entities within the same chunk.
    nodes: unique entities
    edges: co-occurrence counts (converted to confidence later)
    """
    G = nx.Graph()
    chunk_entity_map = {}
    for cid, chunk in zip(chunk_ids, chunks):
        entities = extract_entities_from_chunk(chunk)
        chunk_entity_map[cid] = entities
        for e in entities:
            if not G.has_node(e):
                G.add_node(e, doc_ids=set([cid]))
            else:
                G.nodes[e]["doc_ids"].add(cid)
        # add edges for co-occurrence pairs
        for i in range(len(entities)):
            for j in range(i+1, len(entities)):
                a, b = entities[i], entities[j]
                if G.has_edge(a, b):
                    G[a][b]["weight"] += 1
                else:
                    G.add_edge(a, b, weight=1)

    # convert sets to list and compute confidence scores (normalize weights)
    # normalize edge weights to [0,1]
    all_w = [d["weight"] for _, _, d in G.edges(data=True)]
    if all_w:
        maxw = max(all_w)
    else:
        maxw = 1
    for u, v, d in G.edges(data=True):
        d["confidence"] = round(d["weight"] / maxw, 4)
    # flatten doc_ids
    for n, d in G.nodes(data=True):
        d["doc_ids"] = list(d["doc_ids"])
    print(f"Built graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# ---------------------------
# 6) Retrieval utilities (graph + faiss)
# ---------------------------
def faiss_search_topk(idx: faiss.IndexFlatIP, query_emb: np.ndarray, k: int = 5) -> Tuple[List[int], List[float]]:
    """
    Query FAISS index (we assume vectors already normalized).
    """
    # normalize query
    q = query_emb.astype("float32")
    qnorm = q / (np.linalg.norm(q) + 1e-9)
    D, I = idx.search(np.asarray([qnorm]), k)
    # D is inner product (cosine) scores
    return I[0].tolist(), D[0].tolist()


# ---------------------------
# 7) High-level pipeline
# ---------------------------
def build_index_from_txts(data_dir: Path = DATA_DIR, save_meta=True) -> dict:
    """
    Walk the data_dir, read txt files, extract metadata, chunk, embed, dedupe, save faiss and chunk metadata.
    Returns a dict containing mappings and the faiss index.
    """

    metadata = []
    all_chunks = []
    chunk_meta = []

    # --------------------------------------------------
    # STEP 1: Scan for .txt files
    # --------------------------------------------------
    src_files = list(data_dir.glob("*.txt"))
    print("Found", len(src_files), "txt files")

    if len(src_files) == 0:
        print("No source .txt files found. Exiting early.")
        return {
            "faiss_index": None,
            "chunks": [],
            "embeddings": np.array([]),
        }

    # --------------------------------------------------
    # STEP 2: Extract metadata + chunks
    # --------------------------------------------------
    for f in tqdm(src_files):
        meta = extract_metadata_from_txt(f)
        metadata.append(meta)

        chunks = chunk_text(meta["full_text"])

        for ci, c in enumerate(chunks):
            chunk_id = f"{f.stem}__chunk{ci}"
            all_chunks.append(c)
            chunk_meta.append({
                "chunk_id": chunk_id,
                "source_file": str(f),
                "title": meta["title"],
                "author": meta["author"],
                "date": meta["date"],
                "url": meta["url"],
            })

    # If no chunks
    if len(all_chunks) == 0:
        print("Text files found but no chunks generated. Exiting early.")
        return {
            "faiss_index": None,
            "chunks": [],
            "embeddings": np.array([]),
        }

    # --------------------------------------------------
    # STEP 3: Generate embeddings
    # --------------------------------------------------
    print("Generating embeddings for", len(all_chunks), "chunks...")
    embs = generate_embeddings(all_chunks)

    if not isinstance(embs, np.ndarray) or embs.size == 0:
        print("No embeddings generated. Exiting early.")
        return {
            "faiss_index": None,
            "chunks": [],
            "embeddings": np.array([]),
        }

    # Normalize embeddings (cosine similarity via inner product)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    embs_norm = embs / norms

    # --------------------------------------------------
    # STEP 4: Deduplicate embeddings
    # --------------------------------------------------
    embs_kept, chunks_kept = deduplicate_embeddings(
        embs_norm, all_chunks, threshold=DUP_THRESHOLD
    )

    if len(chunks_kept) == 0:
        print("All chunks removed during deduplication. Exiting.")
        return {
            "faiss_index": None,
            "chunks": [],
            "embeddings": np.array([]),
        }

    # --------------------------------------------------
    # STEP 5: Filter metadata
    # --------------------------------------------------
    kept_meta = []
    for i, cm in enumerate(chunk_meta):
        if chunk_meta[i]["chunk_id"] and all_chunks[i] in chunks_kept:
            kept_meta.append({**cm, "text": all_chunks[i]})

    # If mismatch fallback
    if len(kept_meta) != len(chunks_kept):
        kept_meta = []
        for idx, text in enumerate(chunks_kept):
            kept_meta.append({
                "chunk_id": f"chunk_{idx}",
                "source_file": "unknown",
                "title": "",
                "author": "",
                "date": "",
                "url": "",
                "text": text,
            })

    # --------------------------------------------------
    # STEP 6: Build FAISS index
    # --------------------------------------------------
    dim = embs_kept.shape[1]
    index = init_faiss_index(dim=dim, index_path=FAISS_INDEX_PATH)

    if index.ntotal > 0:
        print("Clearing existing FAISS index")
        index = init_faiss_index(dim=dim, index_path=FAISS_INDEX_PATH)

    # Normalize again before adding to FAISS
    norms = np.linalg.norm(embs_kept, axis=1, keepdims=True)
    embs_kept = embs_kept / (norms + 1e-9)

    index.add(embs_kept.astype("float32"))
    save_faiss_index(index, FAISS_INDEX_PATH)

    # --------------------------------------------------
    # STEP 7: Save metadata
    # --------------------------------------------------
    if save_meta:
        CHUNKS_OUT.parent.mkdir(parents=True, exist_ok=True)

        with open(CHUNKS_OUT, "w", encoding="utf-8") as f:
            json.dump(kept_meta, f, indent=2)

        with open(META_OUT, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print("Saved chunk metadata & document metadata")

    # --------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------
    return {
        "faiss_index": index,
        "chunks": kept_meta,
        "embeddings": embs_kept,
    }
