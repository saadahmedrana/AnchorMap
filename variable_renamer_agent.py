#!/usr/bin/env python3
# variable_renamer_agent.py

import argparse
import re
import pandas as pd
from pathlib import Path
from datetime import datetime

def safe_out_path(out_dir: Path, stem: str, suffix: str = ".ttl") -> Path:
    """Avoid overwriting by auto-incrementing filenames."""
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate = out_dir / f"{stem}{suffix}"
    if not candidate.exists():
        return candidate
    i = 2
    while True:
        candidate = out_dir / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1

def compile_patterns(mappings):
    """
    Build regex patterns to reduce accidental partial replacements.
    We replace exact token matches (not inside alphanum/_).
    """
    compiled = []
    for orig, repl in mappings:
        o = re.escape(orig)
        token_pat = re.compile(rf"(?<![A-Za-z0-9_]){o}(?![A-Za-z0-9_])")
        compiled.append((orig, repl, token_pat))
    return compiled

def apply_replacements(text: str, compiled):
    changes = []
    new_text = text
    for orig, repl, pat in compiled:
        matches = list(pat.finditer(new_text))
        if not matches:
            continue
        new_text, n = pat.subn(repl, new_text)
        if n:
            changes.append((orig, repl, n))
    return new_text, changes

def norm_empty(x):
    if x is None:
        return ""
    s = str(x).strip()
    return "" if s.lower() in {"nan", "none"} else s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_ttl", default="", help="Single input .ttl file (recommended for one-file runs)")
    ap.add_argument("--input_dir", default="", help="Folder with input .ttl files (batch mode)")
    ap.add_argument("--mappings_csv", required=True, help="FINAL mappings CSV (from human_review_cli.py)")
    ap.add_argument("--output_dir", required=True, help="Folder to write corrected .ttl file(s)")
    ap.add_argument("--run_id", default="", help="Optional run id label for logs")
    ap.add_argument("--restrict_by_source_file", action="store_true",
                    help="Only apply mappings whose source_file matches the input file name (recommended)")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    mappings_csv = Path(args.mappings_csv)

    if not mappings_csv.exists():
        raise FileNotFoundError(f"Mappings CSV not found: {mappings_csv}")

    # Decide inputs
    ttl_files = []
    if args.input_ttl:
        ttl_path = Path(args.input_ttl)
        if not ttl_path.exists():
            raise FileNotFoundError(f"Input TTL not found: {ttl_path}")
        ttl_files = [ttl_path]
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input dir not found: {input_dir}")
        ttl_files = sorted(input_dir.glob("*.ttl"))
        if not ttl_files:
            raise FileNotFoundError(f"No .ttl files found in {input_dir}")
    else:
        raise ValueError("Provide either --input_ttl or --input_dir")

    # Load mappings
    acc = pd.read_csv(mappings_csv)
    if acc.empty:
        print("✅ No mappings in final CSV. Nothing to rename.")
        return

    # Expected cols from our updated human_review_cli.py
    required = {"original_name", "final_match"}
    if not required.issubset(set(acc.columns)):
        raise ValueError(f"mappings_csv must contain {required}. Found: {list(acc.columns)}")

    has_file_col = "source_file" in acc.columns and acc["source_file"].astype(str).str.strip().ne("").any()

    # Normalize
    acc["original_name"] = acc["original_name"].apply(norm_empty)
    acc["final_match"] = acc["final_match"].apply(norm_empty)
    if "source_file" in acc.columns:
        acc["source_file"] = acc["source_file"].apply(norm_empty)

    # Drop empties / identity
    acc = acc[(acc["original_name"] != "") & (acc["final_match"] != "") & (acc["original_name"] != acc["final_match"])].copy()
    if acc.empty:
        print("✅ No valid mappings after cleaning empties/identity. Nothing to rename.")
        return

    run_id = args.run_id.strip() or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_rows = []

    for ttl in ttl_files:
        sub = acc
        if args.restrict_by_source_file and has_file_col:
            sub = acc[acc["source_file"].astype(str).str.strip().eq(ttl.name)].copy()

        mappings = [(r["original_name"], r["final_match"]) for _, r in sub.iterrows()]
        if not mappings:
            out_path = safe_out_path(output_dir, f"{ttl.stem}_corrected")
            out_path.write_text(ttl.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
            print(f"ℹ️ {ttl.name}: no applicable mappings -> copied as {out_path.name}")
            continue

        compiled = compile_patterns(mappings)
        original_text = ttl.read_text(encoding="utf-8", errors="ignore")
        new_text, changes = apply_replacements(original_text, compiled)

        out_path = safe_out_path(output_dir, f"{ttl.stem}_corrected")
        out_path.write_text(new_text, encoding="utf-8")

        total_repls = sum(n for _, _, n in changes)
        print(f"✅ {ttl.name} -> {out_path.name} ({len(changes)} mappings, {total_repls} replacements)")

        for orig, repl, n in changes:
            log_rows.append({
                "run_id": run_id,
                "input_file": ttl.name,
                "output_file": out_path.name,
                "original_name": orig,
                "final_match": repl,
                "replacements": n
            })

    # Write log
    log_path = output_dir / "rename_log.csv"
    if log_rows:
        log_df = pd.DataFrame(log_rows)
        if log_path.exists():
            old = pd.read_csv(log_path)
            log_df = pd.concat([old, log_df], ignore_index=True)
        log_df.to_csv(log_path, index=False)
        print(f"\n🧾 Rename log -> {log_path}")
    else:
        print(f"\nℹ️ No replacements were made. (Log not written)")

if __name__ == "__main__":
    main()