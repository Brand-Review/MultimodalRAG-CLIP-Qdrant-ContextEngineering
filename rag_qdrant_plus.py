# rag_qdrant_plus.py
# ------------------------------------------------------------
# Full RAG demo with Hugging Face (embeddings + generation),
# Qdrant vector DB, cross-encoder rerank, MMR diversity,
# near-duplicate filtering, context compression, citations.
#
# Quick start:
#   python rag_qdrant_plus.py
#
# Optional (use server Qdrant):
#   python rag_qdrant_plus.py --use_server_qdrant --qdrant_host localhost --qdrant_port 6333
#
# Requirements:
#   pip install qdrant-client sentence-transformers transformers accelerate torch numpy tqdm
# ------------------------------------------------------------


# Use these queries and observe what changes when you toggle stages:

# Rerank (Cross-Encoder) shines

# Query: “Where is the Eiffel Tower and which museums nearby should I visit?”

# Without rerank, ANN may pull eiffel_programming, eiffel_bridge_spain, or replicas (lexical “Eiffel”).

# With rerank, et_base_1 / et_base_2 / paris_nearby_museums / orsay_focus rise.

# MMR diversity

# Query: “How do I visit the Eiffel Tower (transport, viewpoints, nearby museums)?”

# ANN+r erkank returns many near-identical base snippets.

# With MMR (λ≈0.7) you’ll get paris_transport + paris_nearby_museums + et_hours_tickets_long instead of 4 copies of the base description.

# Near-duplicate filter

# Query: “Where is the Eiffel Tower located?”

# et_base_1 and et_neardup have almost identical wording.

# With dedup (cosine ≥ 0.95), one is dropped, freeing context for something complementary (e.g., paris_admin).

# Context compression

# Query: “How can I avoid long lines at the Eiffel Tower?”

# et_hours_tickets_long, fluff_paris_1, fluff_paris_2 are long with only one key sentence (“buy timed-entry/advance tickets”).

# With compression to 1–2 sentences, prompts get lean and answers become sharper.

# Rerank vs replicas/bridge/programming

# Query: “How tall is the Eiffel Tower?”

# Distractors: et_replica_lv (165 m), et_replica_texas (20 m), eiffel_bridge_spain.

# Good rerank keeps et_conflict_height and et_base_1/2 on top.

# Freshness / disambiguation

# Query: “What reopened in Paris in 2024 near the river?” → notredame_reopen should surface; others are older or irrelevant.

# Query: “What is Python used for in machine learning?” → python_language must beat python_snake.

import os
import re
import uuid
import argparse
from typing import List, Dict, Any, Tuple
import numpy as np
from tqdm import tqdm

# Embeddings & Reranker
from sentence_transformers import SentenceTransformer, CrossEncoder

# Generation
from transformers import pipeline

# Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# -----------------------------
# Defaults / Config
# -----------------------------
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5")     # strong small embedder
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
GEN_MODEL_NAME = os.getenv("GEN_MODEL_NAME", "google/flan-t5-base")            # CPU-friendly generator

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "demo_rag_docs")

# Retrieval & post-retrieval
TOP_K_INITIAL = 16          # ANN retrieve more than needed
TOP_K_FINAL = 5             # Keep fewer after reranking/diversification
RERANK_MIN_SCORE = 0.35     # If top rerank < threshold => "I don't know"
MMR_LAMBDA = 0.7            # Higher favors relevance, lower favors diversity
NEAR_DUP_THRESH = 0.95      # Cosine similarity to drop near-duplicates
SENTS_PER_CHUNK = 2         # Sentence-level compression per chunk

# Chunking (word-based for simplicity)
CHUNK_SIZE = 300
CHUNK_OVERLAP = 60

# Qdrant connection defaults (we use in-memory unless user opts into server)
USE_IN_MEMORY_QDRANT_DEFAULT = True
QDRANT_HOST_DEFAULT = "localhost"
QDRANT_PORT_DEFAULT = 6333

# -----------------------------
# Example documents (replace with your data)
# -----------------------------
# DOCUMENTS: List[Tuple[str, str]] = [
#     ("guide_001", """The Eiffel Tower is located in Paris, France. It was constructed from 1887 to 1889 and serves as a global cultural icon of France."""),
#     ("guide_002", """The Statue of Liberty stands on Liberty Island in New York Harbor. It was a gift from the people of France to the United States in 1886."""),
#     ("kb_everest", """Mount Everest is Earth's highest mountain above sea level, located in the Himalayas on the China–Nepal border. Its elevation is about 8,849 meters."""),
#     ("kb_python", """Python is a high-level programming language widely used for machine learning, data analysis, and web development. Popular libraries include NumPy and PyTorch."""),
#     ("sports_2022", """The FIFA World Cup 2022 took place in Qatar. Argentina won the tournament by defeating France on penalties in the final."""),
#     ("travel_paris", """Top sights in Paris include the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral. Paris is renowned for art, fashion, gastronomy, and culture."""),
# ]

DOCUMENTS = [
    # --- Eiffel Tower (true facts, paraphrases, near-duplicates, conflicts) ---
    ("et_base_1", """The Eiffel Tower stands in the 7th arrondissement of Paris, on the Champ de Mars near the Seine.
It is 324 meters tall with antennas and about 300 meters to the roof. Nearby museums include the Louvre and Musée d'Orsay."""),
    ("et_base_2", """Paris's Eiffel Tower rises above the Champ de Mars. Height: 324 m including antennas.
Visitors often combine the tower with the Louvre Museum and the Musée d'Orsay on the Left Bank."""),
    ("et_hist_1", """Constructed for the 1889 Exposition Universelle, the Eiffel Tower was engineered by Gustave Eiffel's company between 1887 and 1889.
Its role shifted from temporary exhibit to permanent landmark and radio mast."""),
    ("et_hours_tickets_long", """Planning tips: Crowds peak at sunset; online timed-entry tickets can cut queueing time.
There are restaurants on the first and second levels; weather or strikes sometimes change hours.
Security screening occurs before elevators. The most important point: buy advance tickets online from the official site."""),
    ("et_conflict_height", """Some guides list the Eiffel Tower height as 300 m. This refers to the structure without antennas before extensions.
Modern measurements state 324 m including antennas; maintenance may alter centimeters."""),
    # near-duplicate of et_base_1 (should be dropped by dedup at high cosine)
    ("et_neardup", """The Eiffel Tower is in Paris's 7th arrondissement on the Champ de Mars beside the Seine.
With antennas it reaches 324 m (about 300 m to the roof). The Louvre and the Musée d'Orsay are close."""),
    # --- High lexical overlap but wrong target (forces strong reranking) ---
    ("eiffel_programming", """Eiffel is also an object-oriented programming language designed by Bertrand Meyer,
emphasizing Design by Contract. It has nothing to do with visiting the tower in Paris."""),
    ("eiffel_bridge_spain", """The Pont de les Peixateries Velles (Eiffel Bridge) in Girona, Spain, was built by Gustave Eiffel’s firm in 1877.
Despite the 'Eiffel' name, it is a different structure from the Paris tower."""),
    # --- Replicas (semantic trap) ---
    ("et_replica_lv", """A half-scale replica of the Eiffel Tower (about 165 meters) stands at Paris Las Vegas on the Strip in Nevada, USA.
It includes an observation deck and a restaurant called 'Eiffel Tower Restaurant'."""),
    ("et_replica_texas", """Paris, Texas features a small Eiffel Tower replica topped with a red cowboy hat, roughly 20 meters tall."""),
    # --- Nearby sights & practical info (makes MMR pick varied aspects) ---
    ("paris_nearby_museums", """The Musée d'Orsay sits on the Left Bank facing the Tuileries; it is a short RER or walk from the Eiffel Tower.
The Louvre Museum lies across the river near the Tuileries Garden; many visitors pair these with the tower in one day."""),
    ("paris_transport", """For the Eiffel Tower, common transport is Métro line 6 (Bir-Hakeim) or RER C (Champ de Mars–Tour Eiffel).
Trocadéro offers a classic photo viewpoint across the river."""),
    ("paris_admin", """Paris is divided into 20 arrondissements; the 7th includes the Champ de Mars and the Eiffel Tower.
Embassies and museums cluster here, and the Seine riverfront paths are popular at night."""),
    # --- Statue of Liberty cluster (lexically similar 'Liberty', 'gift from France') ---
    ("sol_base_1", """The Statue of Liberty stands on Liberty Island in New York Harbor. It was a gift from France in 1886.
Ferries depart from Battery Park and Liberty State Park."""),
    ("sol_neardup", """Located on Liberty Island, New York Harbor, the Statue of Liberty was gifted by France (dedicated 1886).
Access is by ferry; pedestal/crown tickets are timed."""),
    # --- Everest cluster (confusable 'height' questions) ---
    ("everest_height", """Mount Everest's elevation is about 8,849 meters (2020 Nepal–China survey) on the China–Nepal border in the Himalayas."""),
    ("k2_height", """K2, in the Karakoram, rises to about 8,611 meters—lower than Everest but with a higher fatality rate among eight-thousanders."""),
    # --- Python (word-sense disambiguation) ---
    ("python_language", """Python is a high-level programming language used in machine learning, data analysis, and web development.
Popular libraries include NumPy, pandas, and PyTorch."""),
    ("python_snake", """Pythons are nonvenomous constrictor snakes found in Africa, Asia, and Australia. They are unrelated to the Python programming language."""),
    # --- Long/fluffy tourism paragraphs (to show compression benefit) ---
    ("fluff_paris_1", """Visitors adore Paris for cafés, river cruises, and fashion week. While many activities are seasonal,
the crucial advice for the tower is to secure timed-entry tickets online; almost every other tip is secondary to that practice.
Street performers appear near Trocadéro in the evenings."""),
    ("fluff_paris_2", """Paris itineraries vary widely. Some travelers choose sunrise photos, others prefer midnight illuminations.
Amid all options, one practical rule stands out for the Eiffel Tower: buy tickets in advance on the official website to avoid long lines."""),
    # --- Louvre specifics (useful for multi-aspect answers) ---
    ("louvre_hours", """The Louvre Museum is generally closed on Tuesdays and open the other days; entrance is near the glass pyramid at the Cour Napoléon."""),
    ("orsay_focus", """Musée d'Orsay hosts Impressionist and Post-Impressionist masterpieces; it is about a 25–30 minute riverside walk from the Eiffel Tower."""),
    # --- Notre-Dame (freshness/nearby but different topic) ---
    ("notredame_reopen", """Notre-Dame de Paris, on the Île de la Cité, reopened to the public in December 2024 after restoration following the 2019 fire."""),
    # --- Intentionally misleading yet overlapping vocabulary ---
    ("tower_misc", """The 'Eiffel Tower palm' is a nickname gardeners sometimes use for a palm with a skinny trunk and tufted crown;
despite the nickname, it has no connection to Paris landmarks."""),
]


# -----------------------------
# Utilities
# -----------------------------
STOPWORDS = set("""
a an and are as at be by for from has he in is it its of on or that the to was were will with
""".split())

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Simple word-based chunker with overlap."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = max(0, end - overlap)
    return chunks

def sent_tokenize(text: str) -> List[str]:
    # Lightweight sentence split
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sents if s]

def score_sentence(query_tokens: set, sent: str) -> int:
    tokens = [t.lower() for t in re.findall(r"\w+", sent)]
    tokens = [t for t in tokens if t not in STOPWORDS]
    return len(set(tokens) & query_tokens)

# -----------------------------
# Embeddings / Reranker / Generator
# -----------------------------
def build_embedder(model_name: str) -> SentenceTransformer:
    model = SentenceTransformer(model_name)
    return model

def embed_texts(embedder: SentenceTransformer, texts: List[str]) -> np.ndarray:
    # normalize for cosine similarity (bge models recommend normalize_embeddings=True)
    vecs = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vecs.astype(np.float32)

def build_reranker(model_name: str) -> CrossEncoder:
    return CrossEncoder(model_name)

def rerank_results(reranker: CrossEncoder, query: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not hits:
        return []
    pairs = [(query, h["text"]) for h in hits]
    scores = reranker.predict(pairs)  # higher is better
    for h, s in zip(hits, scores):
        h["rerank_score"] = float(s)
    return sorted(hits, key=lambda x: x["rerank_score"], reverse=True)

def build_generator(model_name: str):
    return pipeline("text2text-generation", model=model_name)

# -----------------------------
# Qdrant helpers
# -----------------------------
def get_qdrant_client(use_server: bool, host: str, port: int) -> QdrantClient:
    if not use_server:
        return QdrantClient(":memory:")
    return QdrantClient(host=host, port=port)

def recreate_collection(client: QdrantClient, name: str, vector_size: int):
    exist = [c.name for c in client.get_collections().collections]
    if name in exist:
        client.delete_collection(name)
    client.recreate_collection(
        collection_name=name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )

def upsert_points(client: QdrantClient, name: str, vectors: np.ndarray, payloads: List[Dict[str, Any]]):
    points = [PointStruct(id=str(uuid.uuid4()), vector=v, payload=p) for v, p in zip(vectors, payloads)]
    client.upsert(collection_name=name, points=points)

# -----------------------------
# Ingestion
# -----------------------------
def ingest_documents(embedder: SentenceTransformer, client: QdrantClient, docs: List[Tuple[str, str]]):
    all_chunks: List[str] = []
    payloads: List[Dict[str, Any]] = []

    for doc_id, text in docs:
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        for i, ch in enumerate(chunks):
            all_chunks.append(ch)
            payloads.append({"doc_id": doc_id, "chunk_id": f"{doc_id}_chunk_{i}", "text": ch})

    print(f"Embedding {len(all_chunks)} chunks...")
    vectors = embed_texts(embedder, all_chunks)

    recreate_collection(client, COLLECTION_NAME, vectors.shape[1])
    print("Upserting into Qdrant...")
    upsert_points(client, COLLECTION_NAME, vectors, payloads)
    print(f"Ingested {len(all_chunks)} chunks into collection '{COLLECTION_NAME}'.")

# -----------------------------
# MMR diversity + duplicate filtering + compression
# -----------------------------
def mmr_diversify(embedder: SentenceTransformer, query: str, hits: List[Dict[str, Any]],
                  k: int, lambda_mult: float) -> List[Dict[str, Any]]:
    """Maximal Marginal Relevance on normalized embeddings."""
    if not hits:
        return []
    q_vec = embed_texts(embedder, [f"query: {query}"])[0]  # (d,)
    cand_texts = [h["text"] for h in hits]
    cand_vecs = embed_texts(embedder, cand_texts)          # (n, d) normalized

    n = len(hits)
    candidates = list(range(n))
    selected_indices: List[int] = []

    # Precompute sim(query, cand)
    sim_q = cand_vecs @ q_vec  # (n,)

    while candidates and len(selected_indices) < k:
        if not selected_indices:
            # pick the most relevant to query
            best_local_idx = int(np.argmax(sim_q[candidates]))
            chosen = candidates[best_local_idx]
            selected_indices.append(chosen)
            candidates.remove(chosen)
            continue

        # Compute redundancy (max sim to any already selected)
        sel_vecs = cand_vecs[selected_indices]              # (m, d)
        # cosine sims to selected (m)
        max_redundancy = np.max(cand_vecs[candidates] @ sel_vecs.T, axis=1)  # (len(candidates),)
        # MMR score
        mmr_scores = lambda_mult * sim_q[candidates] - (1.0 - lambda_mult) * max_redundancy
        best_local_idx = int(np.argmax(mmr_scores))
        chosen = candidates[best_local_idx]
        selected_indices.append(chosen)
        candidates.remove(chosen)

    return [hits[i] for i in selected_indices]

def drop_near_duplicates(embedder: SentenceTransformer, hits: List[Dict[str, Any]], sim_thresh: float) -> List[Dict[str, Any]]:
    """Remove near-duplicate chunks using cosine similarity on normalized embeddings."""
    if not hits:
        return []
    kept: List[Dict[str, Any]] = []
    kept_vecs: List[np.ndarray] = []
    vecs = embed_texts(embedder, [h["text"] for h in hits])  # (n,d) normalized

    for i, h in enumerate(hits):
        v = vecs[i]
        is_dup = False
        for kv in kept_vecs:
            if float(v @ kv) >= sim_thresh:
                is_dup = True
                break
        if not is_dup:
            kept.append(h)
            kept_vecs.append(v)
    return kept

def compress_context(query: str, hits: List[Dict[str, Any]], sents_per_chunk: int) -> List[Dict[str, Any]]:
    """Keep most query-relevant sentences per chunk (very light extractive compression)."""
    q_tokens = set(t for t in re.findall(r"\w+", query.lower()) if t not in STOPWORDS)
    compressed = []
    for h in hits:
        sents = sent_tokenize(h["text"])
        if not sents:
            compressed.append(h)
            continue
        sents_sorted = sorted(sents, key=lambda s: score_sentence(q_tokens, s), reverse=True)
        pick = " ".join(sents_sorted[:sents_per_chunk])
        compressed.append({**h, "text": pick})
    return compressed

# -----------------------------
# Retrieval (ANN -> Rerank -> MMR -> Dedup -> Compress)
# -----------------------------
def retrieve(client: QdrantClient,
             embedder: SentenceTransformer,
             reranker: CrossEncoder,
             query: str,
             top_k_initial: int = TOP_K_INITIAL,
             top_k_final: int = TOP_K_FINAL) -> List[Dict[str, Any]]:

    # 1) ANN (dense) retrieve
    q_text = f"query: {query}"
    q_vec = embed_texts(embedder, [q_text])[0]
    ann = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=q_vec.tolist(),
        limit=top_k_initial,
        with_payload=True,
        score_threshold=None,
    )
    hits = [{"score": r.score, "text": r.payload.get("text", ""), "doc_id": r.payload.get("doc_id")} for r in ann]

    print("without reranking:")
    print(hits)

    # 2) Rerank (cross-encoder)
    hits = rerank_results(reranker, query, hits)
    if not hits or hits[0].get("rerank_score", 0.0) < RERANK_MIN_SCORE:
        return []  # "I don't know" path


    # 3) MMR diversity
    hits = mmr_diversify(embedder, query, hits, k=top_k_final, lambda_mult=MMR_LAMBDA)

    # 4) Near-duplicate filtering
    hits = drop_near_duplicates(embedder, hits, sim_thresh=NEAR_DUP_THRESH)

    # # 5) Context compression
    # hits = compress_context(query, hits, SENTS_PER_CHUNK)

    return hits

# -----------------------------
# Prompt & Generation with citations
# -----------------------------
def make_prompt(user_query: str, contexts: List[Dict[str, Any]]) -> str:
    numbered = []
    for i, c in enumerate(contexts, 1):
        did = c.get("doc_id", "unknown")
        numbered.append(f"[{i}] (source: {did}) {c['text']}")
    context_block = "\n".join(numbered)
    prompt = (
        "You are a helpful assistant. Answer ONLY using the facts in the context. "
        "If the answer is not present, say \"I don't know\". Cite sources with [number].\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {user_query}\n"
        "Answer (with citations):"
    )
    return prompt

def generate_answer(generator, user_query: str, retrieved: List[Dict[str, Any]]) -> str:
    if not retrieved:
        return "I don't know."
    prompt = make_prompt(user_query, retrieved)
    out = generator(prompt, max_new_tokens=220)[0]["generated_text"]
    return out

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="RAG with Qdrant + Rerank + Post-Retrieval")
    parser.add_argument("--use_server_qdrant", action="store_true", help="Use a running Qdrant server instead of in-memory")
    parser.add_argument("--qdrant_host", type=str, default=QDRANT_HOST_DEFAULT)
    parser.add_argument("--qdrant_port", type=int, default=QDRANT_PORT_DEFAULT)
    parser.add_argument("--embed_model", type=str, default=EMBED_MODEL_NAME)
    parser.add_argument("--rerank_model", type=str, default=RERANK_MODEL_NAME)
    parser.add_argument("--gen_model", type=str, default=GEN_MODEL_NAME)
    parser.add_argument("--top_k_initial", type=int, default=TOP_K_INITIAL)
    parser.add_argument("--top_k_final", type=int, default=TOP_K_FINAL)
    args = parser.parse_args()

    use_server = args.use_server_qdrant if args.use_server_qdrant else USE_IN_MEMORY_QDRANT_DEFAULT is False
    # If user did not pass flag, default to in-memory:
    if not args.use_server_qdrant:
        use_server = False

    print("Loading embedding model:", args.embed_model)
    embedder = build_embedder(args.embed_model)

    print("Connecting to Qdrant:", "server" if use_server else "in-memory")
    client = get_qdrant_client(use_server, args.qdrant_host, args.qdrant_port)

    print("Ingesting documents...")
    ingest_documents(embedder, client, DOCUMENTS)

    print("Loading reranker:", args.rerank_model)
    reranker = build_reranker(args.rerank_model)

    print("Loading generator:", args.gen_model)
    generator = build_generator(args.gen_model)

    queries = [
        "Where is the Eiffel Tower located?",
        "Who won FIFA World Cup 2022?",
        "What is the height of Mount Everest?",
        "Which programming language is popular for machine learning?",
        "Where is the Eiffel Tower and which museums nearby should I visit?",
        "How do I visit the Eiffel Tower (transport, viewpoints, nearby museums)?",
        "How can I avoid long lines at the Eiffel Tower?"
    ]

    for q in queries:
        print("\n" + "="*90)
        print("Query:", q)
        hits = retrieve(
            client=client,
            embedder=embedder,
            reranker=reranker,
            query=q,
            top_k_initial=args.top_k_initial,
            top_k_final=args.top_k_final
        )
        if not hits:
            print("[!] Low-confidence retrieval/rerank — returning 'I don't know.'")
        else:
            for i, h in enumerate(hits, 1):
                rrs = h.get("rerank_score", 0.0)
                print(f"[{i}] doc={h.get('doc_id')} rerank={rrs:.3f} text={h['text'][:100]}...")

        answer = generate_answer(generator, q, hits)
        print("\nAnswer:\n", answer)

if __name__ == "__main__":
    main()
