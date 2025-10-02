###
#Author: Omer Sayem
#Date: 2025-09-27
#Description: RAG pipeline for caption to description generation
###

import os, json, uuid, argparse, re
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm

# Embeddings + Reranker
from sentence_transformers import SentenceTransformer, CrossEncoder

# Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# Generation (Transformers)
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

# -------------------------
# Embeddings + Reranker
# -------------------------
EMBED_MODEL = "intfloat/multilingual-e5-base"           # fast & solid
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # light reranker

COLLECTION = "social_posts_rag"
TOP_K_INITIAL = 24
TOP_K_FINAL = 6
ENGAGEMENT_WEIGHT = 0.10  

# Qdrant server
USE_QDRANT_SERVER = True
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

# -------------------------
# Utilities
# -------------------------
def engagement_score(e: Dict[str, int]) -> float:
    if not e: return 0.0
    # simple weighted sum: shares > comments > reactions
    return float(e.get("reactions",0) + 2*e.get("comments",0) + 3*e.get("shares",0))

def normalize(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec, axis=1, keepdims=True)
    n[n==0] = 1.0
    return vec / n

# -------------------------
# Ingestion
# -------------------------
def load_posts_jsonl(path: str):
    import json, re

    # Read raw text and strip BOM
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    if txt.startswith("\ufeff"):
        txt = txt.lstrip("\ufeff")

    stripped = txt.lstrip()
    if stripped.startswith("["):
        data = json.loads(stripped)
        if not isinstance(data, list):
            raise ValueError("Top-level JSON must be a list (array).")

        out = []
        for obj in data:
            if not isinstance(obj, dict):
                continue
            out.append(obj)
        return out


    posts = []
    for ln, line in enumerate(txt.splitlines(), start=1):
        s = line.strip()
        if not s or s.startswith("//") or s.startswith("#"):
            continue
        
        if s.endswith(","):
            s = s[:-1]
        try:
            posts.append(json.loads(s))
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON parse error on line {ln}: {e.msg}. "
                             f"If your file is a pretty-printed JSON array, keep it in [...] "
                             f"or convert to JSONL (one object per line).") from e
    return posts

def build_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)

def get_qdrant() -> QdrantClient:
    if USE_QDRANT_SERVER:
        return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return QdrantClient(":memory:")

def recreate_collection(client: QdrantClient, dim: int):
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION in existing:
        client.delete_collection(COLLECTION)
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
    )

def ingest_posts(posts_path: str):
    posts = load_posts_jsonl(posts_path)
    if not posts:
        raise ValueError("No posts found in JSONL.")

    embedder = build_embedder()
    client = get_qdrant()

    # text to embed = caption + description (style memory)
    texts = []
    payloads = []
    for p in posts:
        cap = p.get("caption","").strip()
        desc = p.get("description","").strip()
        e = engagement_score(p.get("engagement", {}))
        combined = f"caption: {cap}\n\ndescription: {desc}"
        texts.append(combined)
        payloads.append({
            "caption": cap,
            "description": desc,
            "date": p.get("date",""),
            "engagement_score": e
        })

    print(f"Embedding {len(texts)} items...")
    vecs = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

    print("Creating collection & upserting...")
    recreate_collection(client, vecs.shape[1])
    points = [PointStruct(id=str(uuid.uuid4()), vector=v, payload=pl) for v,pl in zip(vecs, payloads)]
    client.upsert(collection_name=COLLECTION, points=points)
    print(f"Ingested {len(points)} posts into Qdrant.")

# -------------------------
# Retrieval + Rerank
# -------------------------
def build_reranker() -> CrossEncoder:
    return CrossEncoder(RERANK_MODEL)

def retrieve_examples(caption_query: str, top_k_initial=TOP_K_INITIAL, top_k_final=TOP_K_FINAL):
    client = get_qdrant()
    embedder = build_embedder()

    q = f"query: {caption_query}"
    q_vec = embedder.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0].astype(np.float32)

    ann = client.search(collection_name=COLLECTION, query_vector=q_vec.tolist(),
                        limit=top_k_initial, with_payload=True)

    # pack
    hits = []
    for r in ann:
        pl = r.payload or {}
        hits.append({
            "text": f"caption: {pl.get('caption','')}\n\ndescription: {pl.get('description','')}",
            "caption": pl.get("caption",""),
            "description": pl.get("description",""),
            "engagement": float(pl.get("engagement_score",0.0)),
            "score_ann": float(r.score)
        })

    # Cross-encoder rerank (query, description) works well; you can also use (query, caption+desc)
    reranker = build_reranker()
    pairs = [(caption_query, h["text"]) for h in hits]
    rerank_scores = reranker.predict(pairs)
    for h, s in zip(hits, rerank_scores):
        h["score_rerank"] = float(s)

    # Blend small engagement boost (min-max normalize engagement)
    if hits:
        es = np.array([h["engagement"] for h in hits], dtype=float)
        if es.max() > es.min():
            es_norm = (es - es.min()) / (es.max() - es.min())
        else:
            es_norm = np.zeros_like(es)
        for h, en in zip(hits, es_norm):
            h["score_final"] = (1.0 - ENGAGEMENT_WEIGHT) * h["score_rerank"] + ENGAGEMENT_WEIGHT * float(en)
    else:
        for h in hits:
            h["score_final"] = h["score_rerank"]

    hits = sorted(hits, key=lambda x: x["score_final"], reverse=True)[:top_k_final]
    return hits

# -------------------------
# Prompt building
# -------------------------
# STYLE_SYSTEM = (
#     "You are a social media copywriter. Given a new caption and a few past examples, "
#     "write ONE descriptive sentence (15–30 words) that matches the style and tone of the examples. "
#     "Be concrete and visual, avoid hashtags and emojis, do not copy text verbatim, and do not invent facts not implied by the caption."
# )

STYLE_SYSTEM = (
    "You are a social media copywriter. Given a new caption and a few past examples, "
    "write ONE humorous descriptive sentence (15–30 words) that matches the style and tone of the examples. "
)

# STYLE_SYSTEM = (
#     "You are a social media copywriter.\n"
#     "Goal: write ONE vivid sentence (15–30 words) that matches the brand style.\n"
#     "Tone: lightly sarcastic, witty, deadpan; playful irony allowed.\n"
#     "Guidelines: be concrete and visual; avoid hashtags/emojis; do not invent facts beyond the caption.\n"
#     "If the caption is Bengali, reply in Bengali."
# )


def make_fewshot_prompt(caption_query: str, examples: List[Dict[str,Any]]) -> str:
    lines = [STYLE_SYSTEM, "\nEXAMPLES:"]
    for i,e in enumerate(examples,1):
        lines.append(f"[{i}] CAPTION: {e['caption']}")
        # Truncate long descriptions to keep token budget
        desc = e["description"].strip()
        if len(desc) > 240: desc = desc[:240] + "..."
        lines.append(f"    DESCRIPTION: {desc}")
    lines.append(f"\nNEW CAPTION: {caption_query}\nDESCRIPTION:")
    return "\n".join(lines)

# -------------------------
# Generation options
#   (1) Transformers local
#   (2) vLLM (OpenAI-compatible) — shown in comments
# -------------------------
def generate_with_transformers(prompt: str) -> str:
    # Use text-only Qwen (fits AutoModelForCausalLM)
    model_id = os.getenv("GEN_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")  # or "Qwen/Qwen2.5-7B-Instruct" if you have RAM
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",   # GPU/MPS if available, else CPU
    )

    messages = [
        {"role": "system", "content": STYLE_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    out = model.generate(
        input_ids,
        max_new_tokens=96,
        temperature=0.3,
        top_p=0.5,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()


"""
# If you prefer vLLM (recommended for 7B on CPU-limited machines):
from openai import OpenAI
def generate_with_vllm(prompt: str) -> str:
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
    messages = [
        {"role":"system","content":STYLE_SYSTEM},
        {"role":"user","content":prompt}
    ]
    resp = client.chat.completions.create(
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        messages=messages,
        temperature=0.7,
        top_p=0.9,
        max_tokens=96
    )
    return resp.choices[0].message.content.strip()
"""

# -------------------------
# CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posts", type=str, default="posts_data.json", help="JSONL file of past posts")
    ap.add_argument("--ingest", action="store_true", help="(Re)ingest the dataset into Qdrant")
    ap.add_argument("--caption", type=str, help="New caption to generate a description for")
    args = ap.parse_args()

    if args.ingest:
        ingest_posts(args.posts)

    if args.caption:
        examples = retrieve_examples(args.caption)
        prompt = make_fewshot_prompt(args.caption, examples)
        print("\n---- FEW-SHOT PROMPT ----\n", prompt, "\n--------------------------\n")

        desc = generate_with_transformers(prompt)  # or: generate_with_vllm(prompt)
        print("Generated description:\n", desc)

if __name__ == "__main__":
    main()
