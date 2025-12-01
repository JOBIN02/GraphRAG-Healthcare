# app.py
import streamlit as st
from agent import run_pipeline, CHUNKS, KG
# from pyvis.network import Network
# import networkx as nx
# import json
# from pathlib import Path
# import streamlit.components.v1 as components

st.set_page_config(layout="wide", page_title="GraphRAG Chat")

st.title("GraphRAG — Chat with your documents (Streamlit)")

# col1, col2 = st.columns([1.5, 1])

# with col1:
query = st.text_input("Ask a question about your documents", value="What is Type 2 diabetes and how is it usually treated?")
top_k = st.slider("Top-k retrieval (FAISS)", 1, 10, 5)
hops = st.slider("Graph expansion hops", 0, 2, 1)
if st.button("Ask") and query.strip():
    with st.spinner("Running GraphRAG pipeline..."):
        result = run_pipeline(query, top_k=top_k, graph_expand_hops=hops)
    st.subheader("Final Answer")
    st.write(result["final_answer"])
    st.subheader("RAG Summary (raw)")
    st.write(result["summary"])
    st.subheader("Retrieved Chunks (top results)")
    for r in result["retrieved"][:top_k]:
        cm = r["chunk_meta"]
        st.markdown(f"**Chunk ID:** `{cm['chunk_id']}`   — score: {r.get('score')}")
        st.text(cm["text"][:1000] + ("..." if len(cm["text"])>1000 else ""))

# with col2:
#     st.subheader("Knowledge Graph Visualization")
#     # We'll render a pyvis graph for the KG nodes+edges (limited to top N nodes for clarity)
#     G = KG
#     # pick top nodes by degree
#     deg = sorted(G.degree, key=lambda x: x[1], reverse=True)
#     top_nodes = [n for n,_ in deg[:150]]  # cap to 150 nodes
#     H = G.subgraph(top_nodes).copy()
#     net = Network(height="300px", width="100%", notebook=False, directed=False)
#     for n, d in H.nodes(data=True):
#         title = f"{n}<br/>docs: {len(d.get('doc_ids',[]))}"
#         net.add_node(n, label=n if len(n) < 30 else n[:27]+"...", title=title, size=10+len(d.get('doc_ids',[])))
#     for u, v, d in H.edges(data=True):
#         net.add_edge(u, v, value=float(d.get("confidence", 0.1)))
#     net.repulsion(node_distance=200, central_gravity=0.1)
#     # save html and embed
#     tmpfile = "tmp_kg.html"
#     net.save_graph(tmpfile)
#     HtmlFile = open(tmpfile, 'r', encoding='utf-8')
#     components.html(HtmlFile.read(), height=300, scrolling=True)
