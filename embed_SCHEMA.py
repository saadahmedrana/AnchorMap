import os, json, numpy as np, time, requests
from dotenv import load_dotenv

# Load the .env file (from project root)
load_dotenv()

AALTO_KEY = os.getenv("AALTO_KEY")
if not AALTO_KEY:
    raise EnvironmentError("⚠️  Please set AALTO_KEY in your .env file.")

# ---------- FILE PATHS ----------
SRC = "JSONSCHEMAFORSHIP.jsonld"

if not os.path.exists(SRC):
    raise FileNotFoundError(f"❌ Ontology file not found: {SRC}")

# Automatically derive output names from SRC
BASENAME = os.path.splitext(os.path.basename(SRC))[0]

CHUNKS_OUT = f"chunks_{BASENAME}.jsonl"
VEC_OUT    = f"{BASENAME}_vectors.npy"
IDS_OUT    = f"{BASENAME}_ids.json"
TXT_OUT    = f"{BASENAME}_texts.json"

# ---------- API CONFIG ----------
EMBED_MODEL = "text-embedding-3-large"
EMBED_URL = "https://aalto-openai-apigw.azure-api.net/v1/openai/text-embedding-3-large/embeddings"
HEADERS = {
    "Content-Type": "application/json",
    "Ocp-Apim-Subscription-Key": AALTO_KEY
}

# ---------- SETTINGS ----------
BATCH_SIZE = 50
SLEEP_BETWEEN_CALLS = 1.0  # seconds


def flatten_value(v):
    if isinstance(v, dict):
        return "; ".join(f"{kk}: {flatten_value(vv)}" for kk, vv in v.items())
    if isinstance(v, list):
        return ", ".join(flatten_value(x) for x in v)
    return str(v)


def make_chunks(src_path, out_path):
    """Convert ontology JSON-LD into flattened text chunks."""
    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    graph = data.get("@graph", [])
    count = 0

    with open(out_path, "w", encoding="utf-8") as w:
        for node in graph:
            if "@id" not in node:
                continue

            iri = node["@id"]
            lines = [f"Entity ID: {iri}."]

            for k, v in node.items():
                if k == "@id":
                    continue
                val = flatten_value(v).strip()
                if val:
                    lines.append(f"{k}: {val}.")

            text = " ".join(lines)
            chunk = {"id": iri, "text": text}
            w.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            count += 1

    print(f"✅ Created {count} chunks → {out_path}")


def embed_batch(texts):
    payload = {"input": texts, "model": EMBED_MODEL}
    r = requests.post(EMBED_URL, headers=HEADERS, json=payload)
    if r.status_code != 200:
        raise RuntimeError(f"Embedding request failed: {r.status_code}\n{r.text[:500]}")
    return [item["embedding"] for item in r.json()["data"]]


def embed_all():
    with open(CHUNKS_OUT, "r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]

    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]

    embeddings = []
    total = len(texts)

    print(f"🧠 Total chunks to embed: {total}")

    for i in range(0, total, BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        print(f" → Embedding batch {i // BATCH_SIZE + 1} ({len(batch)} items)")
        embs = embed_batch(batch)
        embeddings.extend(embs)
        time.sleep(SLEEP_BETWEEN_CALLS)

    np.save(VEC_OUT, np.array(embeddings, dtype=np.float32))
    json.dump(ids, open(IDS_OUT, "w", encoding="utf-8"), indent=2)
    json.dump(texts, open(TXT_OUT, "w", encoding="utf-8"), indent=2)

    print(f"✅ Embeddings saved:")
    print(f"   • {VEC_OUT}")
    print(f"   • {IDS_OUT}")
    print(f"   • {TXT_OUT}")


if __name__ == "__main__":
    make_chunks(SRC, CHUNKS_OUT)
    embed_all()
