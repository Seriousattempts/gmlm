#!/usr/bin/env python3
"""
One-click Milvus bootstrap + bulk-load for Sentence-Transformer embeddings.
Cursor Dev-Container / MCP-runner friendly.
"""

import json, os, re, subprocess, sys, time
from pathlib import Path

# ─── USER CONFIG ─────────────────────────────────────────────
MILVUS_VERSION   = "v2.4.0"
COLLECTION_NAME  = "my_embeddings"
EMBEDDING_DIM    = 384
TEXT_MAX_LEN     = 4096
# ─────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
COMPOSE = ROOT / "docker-compose.yml"
COMPOSE_URL = (
    f"https://github.com/milvus-io/milvus/releases/download/"
    f"{MILVUS_VERSION}/milvus-standalone-docker-compose.yml"
)

def sh(cmd): subprocess.run(cmd, check=True)

def ensure_compose():
    if COMPOSE.exists():
        print("✅ docker-compose.yml present")
    else:
        print("⬇️  Downloading Milvus compose …")
        sh(["wget", COMPOSE_URL, "-O", str(COMPOSE)])

    txt = COMPOSE.read_text(encoding="utf-8")
    patched = re.sub(r'^\s*version:.*$', "# version removed", txt,
                     flags=re.MULTILINE)
    if txt != patched:
        COMPOSE.write_text(patched, encoding="utf-8")
        print("📝 Patched obsolete version line")

def up():
    print("🐳 Starting Milvus …")
    sh(["docker", "compose", "up", "-d"])
    print("⏳ Waiting 15 s for Milvus to bootstrap")
    time.sleep(15)

def load():
    from pymilvus import (
        connections, FieldSchema, CollectionSchema,
        DataType, Collection, utility
    )
    connections.connect(host="localhost", port="19530",
                        user="root", password="Milvus")
    print("🔗 Connected to Milvus")

    if COLLECTION_NAME in utility.list_collections():
        utility.drop_collection(COLLECTION_NAME)
        print(f"🗑️  Dropped old collection {COLLECTION_NAME}")

    schema = CollectionSchema([
        FieldSchema("id", DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        FieldSchema("text", DataType.VARCHAR, max_length=TEXT_MAX_LEN)
    ])
    coll = Collection(COLLECTION_NAME, schema)
    print("📂 Created collection")

    emb = ROOT / "embeddings.npy"
    meta = ROOT / "metadata.json"
    if not emb.exists() or not meta.exists():
        sys.exit("❌ embeddings.npy or metadata.json missing")

    import numpy as np
    vectors = np.load(emb)
    meta = json.loads(meta.read_text(encoding="utf-8"))
    ids   = [m["global_index"] for m in meta]
    texts = [m["text"][:TEXT_MAX_LEN] for m in meta]

    coll.insert([ids, vectors, texts])
    coll.flush()
    print(f"✅ Inserted {len(ids)} vectors")

    coll.create_index(
        field_name="embedding",
        index_params={"metric_type":"COSINE", "index_type":"HNSW",
                      "M":16, "efConstruction":200}
    )
    print("⚡ HNSW index built")

    return coll

def query(coll):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    q = "quick sanity check"
    vec = model.encode([q])[0]
    res = coll.search(
        [vec], "embedding", {"metric_type":"COSINE"}, 3,
        output_fields=["text"])
    for hit in res[0]:
        print(f"{hit.score:0.3f} → {hit.entity.get('text')[:80]}…")

def main():
    ensure_compose()
    up()
    coll = load()
    query(coll)
    print("\n🎉 Milvus up & loaded at localhost:19530  (user=root  pass=Milvus)")

if __name__ == "__main__":
    main()
