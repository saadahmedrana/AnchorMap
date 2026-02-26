#!/usr/bin/env python3
# human_review_cli.py

import argparse
import pandas as pd
from pathlib import Path

DEFAULT_ACCEPT_KEYS = {"y", "yes", "a", "accept"}
DEFAULT_REJECT_KEYS = {"n", "no", "r", "reject", "skip"}

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def norm_empty(x):
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none"}:
        return ""
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="Cleaned masteragent CSV (e.g., eval_results_ecms_clean.csv)")
    ap.add_argument("--out_decisions", required=True, help="Write audit decisions CSV (human decisions)")
    ap.add_argument("--out_final", required=True, help="Write FINAL mappings CSV (auto ACCEPT + human ACCEPT)")
    ap.add_argument("--resume", action="store_true", help="Resume if out_decisions exists")
    args = ap.parse_args()

    in_csv = Path(args.in_csv)
    out_decisions = Path(args.out_decisions)
    out_final = Path(args.out_final)

    df = pd.read_csv(in_csv)

    # Detect columns from your clean CSV
    col_original = pick_col(df, ["original_name", "original", "source_term", "term"])
    col_match    = pick_col(df, ["best_match", "proposed_match", "match", "suggested_match"])
    col_status   = pick_col(df, ["status", "routing", "decision_status"])
    col_score    = pick_col(df, ["score", "similarity", "cosine", "confidence"])
    col_file     = pick_col(df, ["source_file", "file", "input_file", "ttl_file"])

    if not col_original or not col_match or not col_status:
        raise ValueError(
            f"Could not find required columns.\n"
            f"Need at least: original_name, best_match, status.\n"
            f"Found columns: {list(df.columns)}"
        )

    if not col_file:
        # fallback: assume single file run
        df["source_file"] = ""
        col_file = "source_file"

    # Normalize empties
    df[col_match] = df[col_match].apply(norm_empty)
    df[col_original] = df[col_original].apply(norm_empty)
    df[col_file] = df[col_file].apply(norm_empty)

    # ----------------------------
    # 1) Start final mappings with AUTO ACCEPT
    # ----------------------------
    auto_accept = df[(df[col_status].astype(str).str.strip() == "ACCEPT") & (df[col_match] != "")].copy()
    final_map = auto_accept[[col_file, col_original, col_match]].copy()
    final_map.columns = ["source_file", "original_name", "final_match"]

    # Use dict for override behavior
    final_dict = {
        (r["source_file"], r["original_name"]): r["final_match"]
        for _, r in final_map.iterrows()
    }

    # ----------------------------
    # 2) HUMAN REVIEW loop
    # ----------------------------
    hr = df[df[col_status].astype(str).str.strip().eq("HUMAN_REVIEW")].copy()

    decided = set()
    decisions_rows = []

    if args.resume and out_decisions.exists():
        prev = pd.read_csv(out_decisions)
        for _, r in prev.iterrows():
            decided.add((norm_empty(r.get("source_file","")), norm_empty(r.get("original_name",""))))
        decisions_rows = prev.to_dict("records")
        print(f"ℹ️ Resuming: loaded {len(decisions_rows)} previous decisions.")

    if hr.empty:
        print("✅ No HUMAN_REVIEW rows found. Nothing to review.")
    else:
        print(f"🧑‍⚖️ HUMAN REVIEW: {len(hr)} items\n")

        for _, row in hr.iterrows():
            original = norm_empty(row[col_original])
            proposed = norm_empty(row[col_match])
            score = norm_empty(row[col_score]) if col_score else ""
            srcfile = norm_empty(row[col_file])

            key = (srcfile, original)
            if key in decided:
                continue

            print("=" * 80)
            print(f"File:      {srcfile}")
            print(f"Original:  {original}")
            print(f"Proposed:  {proposed if proposed else '(empty)'}")
            if score:
                print(f"Score:     {score}")

            if not proposed:
                print("Decision? [e]dit (enter correct match) / [r]eject / [q]uit")
            else:
                print("Decision? [a]ccept / [r]eject / [e]dit proposed / [q]uit")

            while True:
                ans = input("> ").strip().lower()

                if ans in DEFAULT_ACCEPT_KEYS:
                    if not proposed:
                        print("No proposed match to accept. Use [e] to enter a match, or [r] to reject.")
                        continue
                    final = proposed
                    decisions_rows.append({
                        "source_file": srcfile,
                        "original_name": original,
                        "best_match": proposed,
                        "decision": "ACCEPT",
                        "final_match": final,
                    })
                    final_dict[key] = final  # override / add
                    decided.add(key)
                    break

                elif ans in DEFAULT_REJECT_KEYS:
                    decisions_rows.append({
                        "source_file": srcfile,
                        "original_name": original,
                        "best_match": proposed,
                        "decision": "REJECT",
                        "final_match": "",
                    })
                    # no mapping added
                    decided.add(key)
                    break

                elif ans in {"e", "edit"}:
                    new_val = input("Type corrected match (ontology id) (blank to cancel): ").strip()
                    if new_val:
                        decisions_rows.append({
                            "source_file": srcfile,
                            "original_name": original,
                            "best_match": proposed,
                            "decision": "ACCEPT",
                            "final_match": new_val,
                        })
                        final_dict[key] = new_val  # override / add
                        decided.add(key)
                        break

                elif ans in {"q", "quit"}:
                    print("🛑 Quitting review early. Saving progress...")
                    break

                else:
                    print("Type: a / r / e / q")

            if ans in {"q", "quit"}:
                break

    # ----------------------------
    # 3) Write outputs
    # ----------------------------
    out_decisions.parent.mkdir(parents=True, exist_ok=True)
    out_final.parent.mkdir(parents=True, exist_ok=True)

    decisions_df = pd.DataFrame(decisions_rows)
    if decisions_df.empty:
        decisions_df = pd.DataFrame(columns=["source_file", "original_name", "best_match", "decision", "final_match"])
    decisions_df.to_csv(out_decisions, index=False)

    final_rows = [
        {"source_file": sf, "original_name": orig, "final_match": match}
        for (sf, orig), match in sorted(final_dict.items())
        if norm_empty(match) != ""
    ]
    final_df = pd.DataFrame(final_rows)
    if final_df.empty:
        final_df = pd.DataFrame(columns=["source_file", "original_name", "final_match"])
    final_df.to_csv(out_final, index=False)

    print(f"\n✅ Saved human decisions -> {out_decisions}")
    print(f"✅ Saved FINAL mappings  -> {out_final}")
    print(f"   (auto ACCEPT + human ACCEPT; human overrides auto)")

if __name__ == "__main__":
    main()