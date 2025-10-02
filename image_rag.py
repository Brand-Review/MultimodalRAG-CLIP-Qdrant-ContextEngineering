# mmrag_images_qdrant.py
import os, uuid, argparse, json, math
from typing import List, Dict, Any
import numpy as np
from PIL import Image
from tqdm import tqdm

# Embedders
from sentence_transformers import SentenceTransformer

# Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# Generation (choose one path below)
from transformers import AutoProcessor
from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

# ------------------ Config ------------------
# Image encoder → CLIP (good cross-modal)
IMG_EMB = "sentence-transformers/clip-ViT-L-14"
# Text encoder → multilingual (Bangla-friendly)
TXT_EMB = "intfloat/multilingual-e5-base"

COLLECTION = "mm_posts"
TOPK_INIT = 24
TOPK_FINAL = 6
MMR_LAMBDA = 0.7
DEDUP_THR = 0.96

USE_SERVER = False
QHOST, QPORT = "localhost", 6333


def load_json_any(path:str)->List[Dict[str,Any]]:
    txt = open(path,"r",encoding="utf-8").read().strip()
    if txt.startswith("["): return json.loads(txt)
    return [json.loads(line) for line in txt.splitlines() if line.strip()]

def norm(v:np.ndarray)->np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n==0]=1
    return v/n

# ------------------ Embedders ----------------
IMG = None; TXT = None
def get_img_encoder():
    global IMG
    if IMG is None: IMG = SentenceTransformer(IMG_EMB)
    return IMG

def get_txt_encoder():
    global TXT
    if TXT is None: TXT = SentenceTransformer(TXT_EMB)
    return TXT

def emb_image(paths:List[str])->np.ndarray:
    enc = get_img_encoder()
    imgs = [Image.open(p).convert("RGB") for p in paths]
    return enc.encode(imgs, convert_to_numpy=True, normalize_embeddings=True).astype("float32")

def emb_text(texts:List[str], mode:str)->np.ndarray:
    # E5 wants "query:" vs "passage:"
    prefix = "query: " if mode=="query" else "passage: "
    enc = get_txt_encoder()
    return enc.encode([prefix+t for t in texts], convert_to_numpy=True, normalize_embeddings=True).astype("float32")

# ------------------ Qdrant -------------------
_Q = None
def qdrant()->QdrantClient:
    global _Q
    if _Q is None:
        _Q = QdrantClient(host=QHOST, port=QPORT) if USE_SERVER else QdrantClient(":memory:")
    return _Q

def recreate_collection(dim_img:int, dim_txt:int):
    qc = qdrant()
    names = [c.name for c in qc.get_collections().collections]
    if COLLECTION in names: qc.delete_collection(COLLECTION)
    qc.recreate_collection(
        collection_name=COLLECTION,
        vectors_config={
            "image": VectorParams(size=dim_img, distance=Distance.COSINE),
            "text":  VectorParams(size=dim_txt, distance=Distance.COSINE),
        }
    )

def upsert(points:List[PointStruct]):
    qdrant().upsert(collection_name=COLLECTION, points=points)


def ingest(posts_json:str, images_dir:str):
    posts = load_json_any(posts_json)
    
    paths = [os.path.join(images_dir, p["image"]) for p in posts]
    caps  = [p.get("caption","") for p in posts]
    descs = [p.get("description","") for p in posts]

    print(f"Embedding {len(paths)} images...")
    v_img = emb_image(paths)
    print(f"Embedding {len(caps)} texts (caption+description)...")
    texts = [f"caption: {c}\n\ndescription: {d}" for c,d in zip(caps,descs)]
    v_txt = emb_text(texts, mode="passage")

    recreate_collection(v_img.shape[1], v_txt.shape[1])

    pts=[]
    for vi, vt, p, c, d in zip(v_img, v_txt, posts, caps, descs):
        payload = {
            "image": p["image"], "caption": c, "description": d,

            ## wtf is this?
            **{k:v for k,v in p.items() if k not in ["image","caption","description"]}
        }
        pts.append(PointStruct(id=str(uuid.uuid4()), vector={"image":vi, "text":vt}, payload=payload))

    upsert(pts)
    print(f"Ingested {len(pts)} items into '{COLLECTION}'")


def search_by_text(query:str, topk:int)->List[Dict[str,Any]]:
    qv = emb_text([query], mode="query")[0]
    res = qdrant().search(collection_name=COLLECTION, query_vector=("text", qv.tolist()),
                          limit=topk, with_payload=True)
    return [{"score":r.score, "payload":r.payload} for r in res]

def search_by_image(img_path:str, topk:int)->List[Dict[str,Any]]:
    qv = emb_image([img_path])[0]
    res = qdrant().search(collection_name=COLLECTION, query_vector=("image", qv.tolist()),
                          limit=topk, with_payload=True)
    return [{"score":r.score, "payload":r.payload} for r in res]

def cosine(a,b): return float(np.dot(a,b))

def mmr_diversify(vecs:np.ndarray, base_scores:np.ndarray, k:int, lam:float)->List[int]:
    """vecs: (n,d) normalized; base_scores: relevance (n,)"""
    chosen=[]; cand=list(range(len(vecs)))
    while cand and len(chosen)<k:
        if not chosen:
            i = int(np.argmax(base_scores[cand]))
            chosen.append(cand.pop(cand.index(i)))
            continue
        sel = np.stack([vecs[i] for i in chosen],0)   # (m,d)
        red = vecs[cand]@sel.T                        # (c,m)
        mmr = lam*base_scores[cand] - (1-lam)*np.max(red,axis=1)
        i = cand[int(np.argmax(mmr))]
        cand.remove(i); chosen.append(i)
    return chosen

def dedup_keep(vecs:np.ndarray, thr:float)->List[int]:
    keep=[]
    for i,v in enumerate(vecs):
        if all(float(v@vecs[j]) < thr for j in keep):
            keep.append(i)
    return keep


def post_process_for_text_query(query:str, hits:List[Dict[str,Any]])->List[Dict[str,Any]]:
    """We’ll rerank by CLIP text↔image similarity + apply MMR + dedup on image vectors."""
    if not hits: return []
    # Build image vectors again for MMR/dedup
    img_paths = [h["payload"]["image"] for h in hits]
    img_vecs = emb_image([os.path.join(args.images_dir,p) for p in img_paths])
    # Relevance: use text encoder to get query; use TEXT vectors we stored (already used)
    # For a tighter rerank, re-score with CLIP cross-modal: text (CLIP) vs image (CLIP).
    clip_txt = SentenceTransformer(IMG_EMB)  # CLIP text tower
    q_clip = clip_txt.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")[0]
    rel = img_vecs @ q_clip  # cosine since normalized

    # MMR
    idx = mmr_diversify(img_vecs, rel, k=min(TOPK_FINAL,len(hits)), lam=MMR_LAMBDA)
    # Dedup
    vec_sel = img_vecs[idx]
    keep_rel_idx = dedup_keep(vec_sel, DEDUP_THR)
    idx = [idx[i] for i in keep_rel_idx]
    return [hits[i] for i in idx]


STYLE = (
  "You are a social content writer. Write ONE vivid sentence (15–30 words), "
  "brand-safe, lightly witty, grounded in the new image/caption and inspired by prior examples. "
  "No hashtags/emojis; do not invent facts beyond what’s visible or stated."
)

def gen_with_qwen_vl(new_image_path:str, new_caption:str, examples:List[Dict[str,Any]])->str:
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype="auto", device_map="auto"
    )

    ex_lines = []
    for i,e in enumerate(examples,1):
        ex_lines.append(f"[{i}] CAPTION: {e['payload'].get('caption','')}")
        ex_lines.append(f"    DESCRIPTION: {e['payload'].get('description','')[:200]}")

    messages = [
        {"role":"system","content":STYLE},
        {"role":"user","content":[
            {"type":"text","text":"EXAMPLES:\n"+ "\n".join(ex_lines) + f"\n\nNEW CAPTION: {new_caption}\nWrite the description for this image:"},
            {"type":"image","image":Image.open(new_image_path).convert("RGB")}
        ]}
    ]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=96, temperature=0.7, top_p=0.9, do_sample=True)
    text = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
    return text

    
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--posts", required=True, help="JSON/JSONL with fields: image, caption, description (and more)")
    ap.add_argument("--images_dir", required=True, help="Directory containing image files")
    ap.add_argument("--ingest", action="store_true")
    ap.add_argument("--query_text", type=str, help="Text query (caption) to retrieve similar images/examples")
    ap.add_argument("--query_image", type=str, help="Path to an image to find similar examples")
    ap.add_argument("--generate_for_image", type=str, help="Path to a NEW image for generation (optional)")
    ap.add_argument("--generate_caption", type=str, default="", help="Optional new caption for generation")
    args = ap.parse_args()

    if args.ingest:
        ingest(args.posts, args.images_dir)

    hits = []
    if args.query_text:
        # initial ANN by text vector
        hits = search_by_text(args.query_text, TOPK_INIT)
        hits = post_process_for_text_query(args.query_text, hits)
    elif args.query_image:
        hits = search_by_image(args.query_image, TOPK_INIT)
        # Optional: apply MMR/dedup on image-to-image as well (similar to above)

    if hits:
        print("\nTop examples:")
        for i,h in enumerate(hits,1):
            print(f"[{i}] img={h['payload'].get('image')}  cap={h['payload'].get('caption')[:60]}  desc={h['payload'].get('description')[:60]}")

    if args.generate_for_image:
        desc = gen_with_qwen_vl(args.generate_for_image, args.generate_caption, hits[:3])
        print("\nGenerated description:\n", desc)
