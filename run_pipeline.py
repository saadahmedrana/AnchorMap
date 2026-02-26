#!/usr/bin/env python3
"""
runme.py - interactive pipeline runner for AnchorMap

Flow:
  (optional) embed_SCHEMA.py
  masteragent_ecms.py (one selected TTL)
  clean skipped rows -> eval_results_ecms_clean.csv
  human_review_cli.py -> human_decisions.csv + final_mappings.csv
  variable_renamer_agent.py -> corrected .ttl + rename_log.csv

Embeddings stay in repo root.
All run artifacts go under: RUNS/run_<timestamp>/
Outputs (.ttl) go under: OUTPUT FILES/run_<timestamp>/
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime

import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "Input Files"
RUNS_DIR = ROOT / "RUNS"
OUTPUT_DIR = ROOT / "OUTPUT FILES"

# embeddings expected in ROOT
EMB_VECS = ROOT / "JSONSCHEMAFORSHIP_vectors.npy"
EMB_IDS  = ROOT / "JSONSCHEMAFORSHIP_ids.json"
EMB_TXTS = ROOT / "JSONSCHEMAFORSHIP_texts.json"

MASTERAGENT = ROOT / "masteragent_ecms.py"
EMBEDDER = ROOT / "embed_SCHEMA.py"
HUMAN_REVIEW = ROOT / "human_review_cli.py"
RENAMER = ROOT / "variable_renamer_agent.py"


def run_cmd(cmd, cwd=None):
    """Run a command and stream output live."""
    print("\n$ " + " ".join(str(x) for x in cmd))
    p = subprocess.run(cmd, cwd=cwd, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed (exit={p.returncode}): {' '.join(map(str, cmd))}")


def choose_embeddings():
    print("\nEmbeddings choice:")
    print("  [1] Use locally saved embeddings (recommended)")
    print("  [2] Rebuild embeddings now (runs embed_SCHEMA.py)")
    while True:
        ans = input("> ").strip()
        if ans == "1":
            if not (EMB_VECS.exists() and EMB_IDS.exists() and EMB_TXTS.exists()):
                print("❌ Local embeddings not found in repo root. Choose [2] to rebuild.")
                continue
            print("✅ Using existing embeddings in repo root.")
            return "reuse"
        if ans == "2":
            print("▶ Rebuilding embeddings...")
            run_cmd([sys.executable, str(EMBEDDER)], cwd=ROOT)
            if not (EMB_VECS.exists() and EMB_IDS.exists() and EMB_TXTS.exists()):
                raise RuntimeError("❌ Embedding rebuild ran, but expected embedding files were not created.")
            print("✅ Embeddings rebuilt and saved in repo root.")
            return "rebuilt"
        print("Type 1 or 2.")


def choose_ttl_file():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_DIR}")

    ttls = sorted(INPUT_DIR.glob("*.ttl"))
    if not ttls:
        raise FileNotFoundError(f"No .ttl files found in: {INPUT_DIR}")

    print("\nSelect input TTL file:")
    for i, f in enumerate(ttls, start=1):
        print(f"  [{i}] {f.name}")

    while True:
        ans = input("> ").strip()
        if ans.isdigit():
            idx = int(ans)
            if 1 <= idx <= len(ttls):
                chosen = ttls[idx - 1]
                print(f"✅ Selected: {chosen.name}")
                return chosen
        print("Enter a valid number from the list.")


def clean_skipped(in_csv: Path, out_csv: Path):
    df = pd.read_csv(in_csv)
    before = len(df)
    if "status" not in df.columns:
        raise ValueError(f"Expected 'status' column in {in_csv}, found: {list(df.columns)}")
    df2 = df[df["status"] != "SKIPPED_NOT_IN_STANDARD"].copy()
    removed = before - len(df2)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df2.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"✅ Removed {removed} skipped rows")
    print(f"✅ Saved cleaned file -> {out_csv}")


def summarize(clean_csv: Path, decisions_csv: Path):
    df = pd.read_csv(clean_csv)

    # exclude skipped (should already be removed)
    if "status" in df.columns:
        df = df[df["status"] != "SKIPPED_NOT_IN_STANDARD"].copy()

    counts = df["status"].value_counts(dropna=False).to_dict() if "status" in df.columns else {}

    # human decision stats
    human_accept = human_reject = human_edit = 0
    if decisions_csv.exists():
        dec = pd.read_csv(decisions_csv)
        if not dec.empty and {"decision", "final_match"}.issubset(dec.columns):
            human_accept = int((dec["decision"].astype(str).str.upper() == "ACCEPT").sum())
            human_reject = int((dec["decision"].astype(str).str.upper() == "REJECT").sum())

            # detect edits when we have best_match column
            if "best_match" in dec.columns:
                a = dec[dec["decision"].astype(str).str.upper() == "ACCEPT"].copy()
                a["best_match"] = a["best_match"].astype(str).fillna("").str.strip()
                a["final_match"] = a["final_match"].astype(str).fillna("").str.strip()
                if "best_match" in dec.columns:
                    a = dec[dec["decision"].astype(str).str.upper() == "ACCEPT"].copy()
                    a["best_match"] = a["best_match"].fillna("").astype(str).str.strip()
                    a["final_match"] = a["final_match"].fillna("").astype(str).str.strip()

                    edit_mask = (a["best_match"] != "") & (a["final_match"] != "") & (a["best_match"] != a["final_match"])
                    human_edit = int(edit_mask.sum())

    # Only print non-skipped summary
    print("\n" + "=" * 70)
    print("SUMMARY (skipped variables not shown)")
    print("=" * 70)
    print(f"Auto ACCEPT:    {counts.get('ACCEPT', 0)}")
    print(f"HUMAN_REVIEW:   {counts.get('HUMAN_REVIEW', 0)}")
    print(f"NO_MATCH:       {counts.get('NO_MATCH', 0)}")
    print("-" * 70)
    print(f"Human ACCEPT:   {human_accept}  (including edits)")
    print(f"Human EDIT:     {human_edit}")
    print(f"Human REJECT:   {human_reject}")
    print("=" * 70)


def main():
    print("=== AnchorMap RUNME ===")

    # 1) embeddings choice
    choose_embeddings()

    # 2) pick TTL
    ttl = choose_ttl_file()

    # 3) set run folders
    run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = RUNS_DIR / f"run_{run_id}"
    out_dir = OUTPUT_DIR / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # filenames in run dir
    results_csv = run_dir / "eval_results_ecms_onefile.csv"
    audit_csv = run_dir / "routing_audit_ecms_onefile.csv"
    clean_csv = run_dir / "eval_results_ecms_clean.csv"
    human_decisions = run_dir / "human_decisions.csv"
    final_mappings = run_dir / "final_mappings.csv"

    print(f"\n📁 Run folder:     {run_dir}")
    print(f"📁 Output folder:  {out_dir}")

    # 4) run masteragent
    run_cmd([
        sys.executable, str(MASTERAGENT),
        "--input_ttl", str(ttl),
        "--out_csv", str(results_csv),
        "--audit_csv", str(audit_csv),
    ], cwd=ROOT)

    # 5) clean skipped rows (instead of hardcoded clean_csv.py)
    clean_skipped(results_csv, clean_csv)

    # 6) human review -> produces decisions + FINAL mappings (auto ACCEPT + human ACCEPT)
    run_cmd([
        sys.executable, str(HUMAN_REVIEW),
        "--in_csv", str(clean_csv),
        "--out_decisions", str(human_decisions),
        "--out_final", str(final_mappings),
    ], cwd=ROOT)

    # 7) renamer: apply final mappings to the selected TTL
    run_cmd([
        sys.executable, str(RENAMER),
        "--input_ttl", str(ttl),
        "--mappings_csv", str(final_mappings),
        "--output_dir", str(out_dir),
        "--restrict_by_source_file",
        "--run_id", f"run_{run_id}",
    ], cwd=ROOT)

    # 8) summary
    summarize(clean_csv, human_decisions)

    print("\n✅ Done.")
    print(f"FINAL mappings:  {final_mappings}")
    print(f"Corrected TTLs:  {out_dir}")


if __name__ == "__main__":
    main()