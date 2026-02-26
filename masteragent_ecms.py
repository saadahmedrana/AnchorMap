#!/usr/bin/env python3
# ===========================================================


import os, re, json, time, csv
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from rdflib import Graph, Namespace, RDF

# ---------------- PATHS ----------------
HERE = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(HERE, "..", ".env"))
load_dotenv()

AALTO_KEY = os.getenv("AALTO_KEY")
if not AALTO_KEY:
    raise EnvironmentError("Set AALTO_KEY in .env or environment")

# ---------------- CONFIG ----------------


TOP_K = 5
MIN_SIM = 0.45
SIM_GAP = 0.06

MAX_RETRIES = 2
BACKOFF = 8

# Decision thresholds
NO_MATCH_THR = 0.40
HUMAN_REVIEW_THR = 0.45

# --- API endpoints ---
EMBED_URL = "https://aalto-openai-apigw.azure-api.net/v1/openai/text-embedding-3-large/embeddings"
LLM_URL   = "https://aalto-openai-apigw.azure-api.net/v1/openai/deployments/gpt-4.1-2025-04-14/chat/completions"
HEADERS   = {"Content-Type": "application/json", "Ocp-Apim-Subscription-Key": AALTO_KEY}

# --- ontology embeddings ---
ONTO_VECS = os.path.join(HERE, "VariblesDefinedmotor_vectors.npy")
ONTO_IDS  = os.path.join(HERE, "VariblesDefinedmotor_ids.json")
ONTO_TXTS = os.path.join(HERE, "VariblesDefinedmotor_texts.json")

# --- skip list (REQUIRED) ---
SKIP_CSV  = os.path.join(HERE, "skip_variables.csv")

# Outputs (one-file)
OUT_CSV   = os.path.join(HERE, "eval_results_ecms_onefile.csv")
AUDIT_CSV = os.path.join(HERE, "routing_audit_ecms_onefile.csv")


# --- namespaces ---
sosa = Namespace("http://www.w3.org/ns/sosa/")
fmu  = Namespace("http://example.com/fmu#")

# ---------------- THROTTLE ----------------
# Limit is 100 req/min. Your spec: sleep 2 seconds after every 2 requests.
THROTTLE_EVERY_N_REQUESTS = 1
THROTTLE_SLEEP_SECONDS = 1.7
_api_calls = 0

def throttle():
    global _api_calls
    _api_calls += 1
    if _api_calls % THROTTLE_EVERY_N_REQUESTS == 0:
        time.sleep(THROTTLE_SLEEP_SECONDS)

# Optional: cache embeddings to reduce calls + timeouts
_EMBED_CACHE = {}

# ===========================================================
# HELPERS
# ===========================================================

def norm(s: str) -> str:
    return "" if s is None else str(s).strip().lower()

def load_skip_set_required():
    if not os.path.exists(SKIP_CSV):
        raise FileNotFoundError(f"❌ skip list is REQUIRED but missing: {SKIP_CSV}")

    df = pd.read_csv(SKIP_CSV)
    cols = {c.lower(): c for c in df.columns}

    name_col = None
    for cand in ["original_name", "oem_variable", "variable", "name"]:
        if cand in cols:
            name_col = cols[cand]
            break
    if not name_col:
        raise ValueError(f"❌ {SKIP_CSV} must contain a column like original_name/variable/name")

    skip = set(df[name_col].dropna().map(norm).tolist())
    print(f"🧹 Skip set loaded: {len(skip)} variables → {SKIP_CSV}")
    return skip

def embed_text(text, dim):
    key = text.strip()
    if key in _EMBED_CACHE:
        return _EMBED_CACHE[key]

    payload = {"input": key, "model": "text-embedding-3-large"}
    timeout_s = 90

    for attempt in range(MAX_RETRIES + 5):
        try:
            throttle()
            r = requests.post(EMBED_URL, headers=HEADERS, json=payload, timeout=timeout_s)

            if r.status_code == 200:
                vec = np.array(r.json()["data"][0]["embedding"], dtype=np.float32)
                _EMBED_CACHE[key] = vec
                return vec

            if r.status_code == 403:
                raise RuntimeError("403: Not on Aalto network / VPN")

            if r.status_code == 429:
                time.sleep(min(60, BACKOFF * (2 ** attempt)))
                continue

            # other non-200 → wait then retry
            time.sleep(min(30, BACKOFF * (attempt + 1)))
            continue

        except requests.exceptions.RequestException:
            # ReadTimeout, ConnectionError etc.
            time.sleep(min(60, BACKOFF * (2 ** attempt)))

    # exhausted retries → return zeros (do not crash)
    return np.zeros(dim, dtype=np.float32)

def cosine_similarity(v, M):
    v = v / (np.linalg.norm(v) + 1e-9)
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    return np.dot(M, v)

def reason_llm(query, candidates):
    prompt = f"""
Map OEM variable to ontology.

Variable:
{query}

Candidates:
{json.dumps(candidates, indent=2)}

Return strict JSON:
{{"best_match":"","confidence":0.0,"reason":""}}
"""
    timeout_s = 240

    for attempt in range(MAX_RETRIES + 5):
        try:
            throttle()
            r = requests.post(
                LLM_URL,
                headers=HEADERS,
                json={
                    "model": "gpt-4.1-2025-04-14",
                    "messages": [
                        {"role": "system", "content": "Return JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.1,
                },
                timeout=timeout_s,
            )

            if r.status_code == 200:
                content = r.json()["choices"][0]["message"]["content"]
                start, end = content.find("{"), content.rfind("}") + 1
                try:
                    return json.loads(content[start:end])
                except Exception:
                    return {"best_match": "", "confidence": 0.0, "reason": "LLM parse error"}

            if r.status_code == 403:
                raise RuntimeError("403: Not on Aalto network / VPN")

            if r.status_code == 429:
                time.sleep(min(60, BACKOFF * (2 ** attempt)))
                continue

            return {"best_match": "", "confidence": 0.0, "reason": f"LLM API {r.status_code}"}

        except requests.exceptions.RequestException:
            time.sleep(min(60, BACKOFF * (2 ** attempt)))

    return {"best_match": "", "confidence": 0.0, "reason": "LLM request failed (timeout/retries exhausted)"}

def is_ood(name):
    return bool(re.search(r"\bPkt|\bPLC|\bFW|\bDbgVar|\bChecksum", name or "", re.I))

def route_by_conf(conf: float) -> str:
    if conf <= NO_MATCH_THR:
        return "NO_MATCH"
    if conf <= HUMAN_REVIEW_THR:
        return "HUMAN_REVIEW"
    return "ACCEPT"

def extract_fmu_variable_names(ttl_path):
    g = Graph()
    g.parse(ttl_path, format="turtle")

    vars_list = []
    for s, _, _ in g.triples((None, RDF.type, sosa.ObservableProperty)):
        for _, _, o in g.triples((s, fmu.hasFMUVariableName, None)):
            vars_list.append(str(o))

    # stable unique
    seen, out = set(), []
    for v in vars_list:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out

# ===========================================================
# MAIN (ONE FILE)
# ===========================================================

from pathlib import Path

def run_multifile():
    # load ontology
    onto_vecs = np.load(ONTO_VECS)
    onto_ids  = json.load(open(ONTO_IDS, "r", encoding="utf-8"))
    onto_txts = json.load(open(ONTO_TXTS, "r", encoding="utf-8"))
    dim = onto_vecs.shape[1]

    # REQUIRED skip list
    skip_set = load_skip_set_required()


    ttl_files = sorted(Path(HERE).glob("*.ttl"))
    if not ttl_files:
        raise RuntimeError("❌ No .ttl files found in this folder")

    print("\n" + "="*90)
    print(f"▶ MULTI-FILE RUN: {len(ttl_files)} files (*.ttl)")
    print("="*90)

    all_results, all_audit = [], []

    for ttl in ttl_files:
        ttl_path = str(ttl)
        base_file = ttl.name

        print(f"\n--- {base_file} ---")
        extracted = extract_fmu_variable_names(ttl_path)

        kept, skipped = [], []
        for v in extracted:
            (skipped if norm(v) in skip_set else kept).append(v)

        print(f"Extracted: {len(extracted)} | Skipped: {len(skipped)} | Sent to API: {len(kept)}")

        # log skipped
        for name in skipped:
            row = {
                "file": base_file,
                "original_name": name,
                "best_match": "",
                "confidence": 0.0,
                "reason": "Skipped (Not found in standard / not defined in ontology)",
                "status": "SKIPPED_NOT_IN_STANDARD",
            }
            all_results.append(row)
            all_audit.append({**row, "method":"SKIP_LIST","top_sim":0.0,"margin":0.0,"top_candidates":"[]"})

        # process kept
        for name in kept:
            if is_ood(name):
                row = {
                    "file": base_file,
                    "original_name": name,
                    "best_match": "",
                    "confidence": 0.0,
                    "reason": "OOD",
                    "status": "NO_MATCH",
                }
                all_results.append(row)
                all_audit.append({**row, "method":"OOD_GATE","top_sim":0.0,"margin":0.0,"top_candidates":"[]"})
                continue

            query = f"Variable '{name}' from OEM dataset"
            qvec = embed_text(query, dim)
            sims = cosine_similarity(qvec, onto_vecs)

            idx = np.argsort(sims)[::-1][:TOP_K]
            top_sim = float(sims[idx[0]]) if len(idx) else 0.0
            second_sim = float(sims[idx[1]]) if len(idx) > 1 else 0.0
            margin = float(top_sim - second_sim)

            top = [{"id": onto_ids[i], "sim": float(sims[i]), "text": onto_txts[i]} for i in idx]
            top_candidates_compact = json.dumps(
                [{"id": t["id"], "sim": round(t["sim"], 4)} for t in top],
                ensure_ascii=False
            )

            if top_sim < MIN_SIM:
                final_conf = float(top_sim)
                status = route_by_conf(final_conf)
                row = {
                    "file": base_file,
                    "original_name": name,
                    "best_match": "",
                    "confidence": final_conf,
                    "reason": f"Low similarity (top_sim={top_sim:.3f} < MIN_SIM={MIN_SIM})",
                    "status": status
                }
                all_results.append(row)
                all_audit.append({**row, "method":"LOW_SIM","top_sim":top_sim,"margin":margin,"top_candidates":top_candidates_compact})
                continue

            if len(idx) > 1 and margin >= SIM_GAP:
                best = onto_ids[idx[0]]
                final_conf = float(min(0.99, top_sim))
                status = route_by_conf(final_conf)
                row = {
                    "file": base_file,
                    "original_name": name,
                    "best_match": best,
                    "confidence": final_conf,
                    "reason": f"Auto by margin (top_sim={top_sim:.3f}, margin={margin:.3f})",
                    "status": status
                }
                all_results.append(row)
                all_audit.append({**row, "method":"AUTO_MARGIN","top_sim":top_sim,"margin":margin,"top_candidates":top_candidates_compact})
                continue

            llm = reason_llm(query, top)
            llm_best = llm.get("best_match", "") or ""
            llm_conf = float(llm.get("confidence", 0.0) or 0.0)
            status = route_by_conf(llm_conf)

            row = {
                "file": base_file,
                "original_name": name,
                "best_match": llm_best,
                "confidence": llm_conf,
                "reason": llm.get("reason", ""),
                "status": status
            }
            all_results.append(row)
            all_audit.append({**row, "method":"LLM","top_sim":top_sim,"margin":margin,"top_candidates":top_candidates_compact})

    # write combined outputs
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file","original_name","best_match","confidence","reason","status"])
        w.writeheader()
        w.writerows(all_results)

    with open(AUDIT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["file","original_name","best_match","confidence","status","reason","method","top_sim","margin","top_candidates"]
        )
        w.writeheader()
        w.writerows(all_audit)

    print(f"\n✅ Saved results → {OUT_CSV}")
    print(f"🧾 Saved audit   → {AUDIT_CSV}")
    print(f"📞 Total API calls (embed + llm): {_api_calls}")


    counts = {}
    for r in all_results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("\nRouting counts (including skipped):")
    for k in sorted(counts.keys()):
        print(f"  {k}: {counts[k]}")


if __name__ == "__main__":
    run_multifile()
