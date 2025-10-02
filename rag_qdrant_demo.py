import os
from typing import List, Dict, Any
import uuid
import numpy as np
from tqdm import tqdm

# -------- Embeddings (Hugging Face / SentenceTransformers) --------
from sentence_transformers import SentenceTransformer

# -------- Qdrant client --------
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# -------- Text generation (Hugging Face) --------
from transformers import pipeline


# -----------------------------
# 0) Config
# -----------------------------
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"      # small, fast, strong baseline
GEN_MODEL_NAME   = "google/flan-t5-base"         # lightweight, CPU-friendly
COLLECTION_NAME  = "demo_rag_docs"
TOP_K            = 2
CHUNK_SIZE       = 300        # tokens-ish (we’ll split by words for simplicity)
CHUNK_OVERLAP    = 60

# Switch between Docker Qdrant and in-memory:
USE_IN_MEMORY_QDRANT = False
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333


# -----------------------------
# 1) Toy documents (replace with your own)
# -----------------------------
DOCUMENTS = [
    ("guide_001", """The Eiffel Tower is located in Paris, France. It was constructed from 1887 to 1889 and serves as a global cultural icon of France."""),
    ("guide_002", """The Statue of Liberty stands on Liberty Island in New York Harbor. It was a gift from the people of France to the United States in 1886."""),
    ("kb_everest", """Mount Everest is Earth's highest mountain above sea level, located in the Himalayas on the China–Nepal border. Its elevation is about 8,849 meters."""),
    ("kb_python", """Python is a high-level programming language widely used for machine learning, data analysis, and web development. Popular libraries include NumPy and PyTorch."""),
    ("sports_2022", """The FIFA World Cup 2022 took place in Qatar. Argentina won the tournament by defeating France on penalties in the final."""),
    ("travel_paris", """Top sights in Paris include the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral. Paris is renowned for art, fashion, gastronomy, and culture."""),
]


# -----------------------------
# 2) Simple word-based chunking
# -----------------------------
def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
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


# -----------------------------
# 3) Build embeddings
# -----------------------------
def build_embedder(model_name: str):
    model = SentenceTransformer(model_name)
    # bge models recommend instruction tuning for queries: "query: {text}"
    return model

def embed_texts(embedder, texts: List[str]) -> np.ndarray:
    # Normalize for cosine similarity (recommended for bge)
    vecs = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vecs.astype(np.float32)


# -----------------------------
# 4) Qdrant helpers
# -----------------------------
def get_qdrant_client() -> QdrantClient:
    if USE_IN_MEMORY_QDRANT:
        return QdrantClient(":memory:")
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def recreate_collection(client: QdrantClient, name: str, vector_size: int):
    if name in [c.name for c in client.get_collections().collections]:
        client.delete_collection(name)
    client.recreate_collection(
        collection_name=name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )

def upsert_points(client: QdrantClient, name: str, vectors: np.ndarray, payloads: List[Dict[str, Any]]):
    points = []
    for vec, payload in zip(vectors, payloads):
        points.append(PointStruct(id=str(uuid.uuid4()), vector=vec, payload=payload))
    client.upsert(collection_name=name, points=points)


# -----------------------------
# 5) Ingestion pipeline
# -----------------------------
def ingest_documents(embedder, client: QdrantClient):
    # 5.1 chunk
    all_chunks = []
    payloads = []
    for doc_id, text in DOCUMENTS:
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        for i, ch in enumerate(chunks):
            all_chunks.append(ch)
            payloads.append({
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_chunk_{i}",
                "text": ch,
            })

    # 5.2 embed
    print(f"Embedding {len(all_chunks)} chunks...")
    vectors = embed_texts(embedder, all_chunks)

    # 5.3 create collection & upsert
    recreate_collection(client, COLLECTION_NAME, vectors.shape[1])
    print("Upserting into Qdrant...")
    upsert_points(client, COLLECTION_NAME, vectors, payloads)

    print(f"Ingested {len(all_chunks)} chunks into collection '{COLLECTION_NAME}'.")


# -----------------------------
# 6) Retrieval pipeline
# -----------------------------
def retrieve(client: QdrantClient, embedder, query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    # bge query instruction: prepend "query: " (improves retrieval)
    q_text = f"query: {query}"
    q_vec = embed_texts(embedder, [q_text])[0]  # shape (d,)
    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=q_vec.tolist(),
        limit=top_k,
        with_payload=True,
        score_threshold=None,
    )
    # Convert to simpler dicts
    out = []
    for r in search_result:
        out.append({"score": r.score, "text": r.payload.get("text", ""), "doc_id": r.payload.get("doc_id")})
    return out


# -----------------------------
# 7) Generation pipeline
# -----------------------------
def build_generator(model_name: str):
    # flan-t5 works with "task-style" prompts; keeps it lightweight for CPU
    return pipeline("text2text-generation", model=model_name)

def make_prompt(user_query: str, contexts: List[str]) -> str:
    # A compact, grounding prompt
    context_block = "\n\n".join(f"- {c}" for c in contexts)
    prompt = (
        "You are a helpful assistant. Answer the user question ONLY using the context below. "
        "If the answer is not present, say you don't know.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {user_query}\n"
        "Answer:"
    )

    print("Prompt:")
    print(prompt)
    return prompt

def generate_answer(generator, user_query: str, retrieved: List[Dict[str, Any]]) -> str:
    contexts = [r["text"] for r in retrieved]
    prompt = make_prompt(user_query, contexts)
    out = generator(prompt, max_new_tokens=200)[0]["generated_text"]
    return out


# -----------------------------
# 8) Demo run
# -----------------------------
def main():
    print("Loading embedding model...")
    embedder = build_embedder(EMBED_MODEL_NAME)

    print("Connecting to Qdrant...")
    client = get_qdrant_client()

    print("Ingesting documents...")
    ingest_documents(embedder, client)

    print("Loading generator...")
    generator = build_generator(GEN_MODEL_NAME)

    # Try a few queries
    queries = [
        "Where is the Eiffel Tower located?",
        "Who won FIFA World Cup 2022?",
        "What is the height of Mount Everest?",
        "Which programming language is popular for machine learning?"
    ]

    for q in queries:
        print("\n" + "="*80)
        print("Query:", q)
        hits = retrieve(client, embedder, q, top_k=TOP_K)
        for i, h in enumerate(hits, 1):
            print(f"[{i}] score={h['score']:.3f} doc={h['doc_id']}  text={h['text'][:80]}...")

        answer = generate_answer(generator, q, hits)
        print("\nAnswer:\n", answer)

if __name__ == "__main__":
    main()
