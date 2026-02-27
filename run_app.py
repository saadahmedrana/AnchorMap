import streamlit as st
import subprocess
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# -----------------------------
# Paths
# -----------------------------
ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "Input Files"
RUNS_DIR = ROOT / "RUNS"
OUTPUT_DIR = ROOT / "OUTPUT FILES"

MASTERAGENT = ROOT / "masteragent_ecms.py"
EMBEDDER = ROOT / "embed_SCHEMA.py"
RENAMER = ROOT / "variable_renamer_agent.py"

EMB_VECS = ROOT / "JSONSCHEMAFORSHIP_vectors.npy"
EMB_IDS  = ROOT / "JSONSCHEMAFORSHIP_ids.json"
EMB_TXTS = ROOT / "JSONSCHEMAFORSHIP_texts.json"


# -----------------------------
# Helpers
# -----------------------------
def run_cmd(cmd, cwd=ROOT):
    """Run a command and return (returncode, stdout+stderr)."""
    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
    )
    output = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
    return p.returncode, output
def reset_app_state():
    """Clear run-specific session state so user can start a new run cleanly."""
    keys_to_clear = [
        "run_started", "run_id", "run_dir", "out_dir", "selected_ttl",
        "results_csv", "audit_csv", "clean_csv", "final_mappings_csv", "human_decisions_csv",
        "df_clean", "logs",
        "hr_index", "hr_decisions", "_clear_edit_box", "edit_box",
        "downloads_ready", "download_payloads",
    ]
    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()

def ensure_session_defaults():
    st.session_state.setdefault("run_started", False)
    st.session_state.setdefault("run_id", "")
    st.session_state.setdefault("run_dir", None)
    st.session_state.setdefault("out_dir", None)
    st.session_state.setdefault("selected_ttl", None)
    st.session_state.setdefault("downloads_ready", False)
    st.session_state.setdefault("download_payloads", {})  # name -> byte


    st.session_state.setdefault("results_csv", None)
    st.session_state.setdefault("audit_csv", None)
    st.session_state.setdefault("clean_csv", None)
    st.session_state.setdefault("final_mappings_csv", None)
    st.session_state.setdefault("human_decisions_csv", None)

    st.session_state.setdefault("df_clean", None)
    st.session_state.setdefault("logs", "")

    # Human review state
    st.session_state.setdefault("hr_index", 0)
    st.session_state.setdefault("hr_decisions", {})  # key -> dict(decision, final_match, best_match, confidence)

    # IMPORTANT: flag to clear edit_box safely BEFORE widget creation
    st.session_state.setdefault("_clear_edit_box", False)


def append_log(txt: str):
    st.session_state["logs"] = (st.session_state.get("logs", "") + "\n" + txt).strip()


def workflow_banner():
    df_clean = st.session_state.get("df_clean")
    if not (st.session_state.get("run_started") and isinstance(df_clean, pd.DataFrame)):
        st.info("Workflow: Run → Human review (if needed) → Export")
        return

    hr = df_clean[df_clean["status"].astype(str).str.strip().eq("HUMAN_REVIEW")]
    total = len(hr)
    decided = len(st.session_state.get("hr_decisions", {}))
    if total == 0:
        st.success("Workflow: Run complete → Export ready")
    else:
        st.info(f"Workflow: Run complete → Human review ({decided}/{total}) → Export")


def clean_skipped(df: pd.DataFrame) -> pd.DataFrame:
    if "status" not in df.columns:
        return df.copy()
    return df[df["status"] != "SKIPPED_NOT_IN_STANDARD"].copy()


def build_final_mappings(df_clean: pd.DataFrame, hr_decisions: dict) -> pd.DataFrame:
    """
    final_mappings.csv should contain:
      source_file, original_name, final_match

    Use:
      - AUTO_ACCEPT rows from df_clean where status == ACCEPT and best_match not empty
      - HUMAN decisions override for those keys (mainly for HUMAN_REVIEW rows)
    """
    # Start with auto-accepted
    auto = df_clean[df_clean["status"].astype(str).str.strip().eq("ACCEPT")].copy()
    auto["best_match"] = auto["best_match"].astype(str).fillna("").str.strip()
    auto = auto[auto["best_match"].str.len().gt(0)].copy()

    final = pd.DataFrame({
        "source_file": auto["file"].astype(str).str.strip(),
        "original_name": auto["original_name"].astype(str).str.strip(),
        "final_match": auto["best_match"].astype(str).str.strip(),
        "origin": "AUTO_ACCEPT",
    })

    # Apply human decisions
    human_rows = []
    for _, d in hr_decisions.items():
        decision = str(d.get("decision", "")).upper().strip()
        source_file = str(d.get("file", "")).strip()
        original = str(d.get("original_name", "")).strip()
        best_match = str(d.get("best_match", "")).strip()
        final_match = str(d.get("final_match", "")).strip()

        if decision == "ACCEPT" and final_match:
            origin = "HUMAN_ACCEPT"
            if best_match and final_match and best_match != final_match:
                origin = "HUMAN_EDIT"
            human_rows.append({
                "source_file": source_file,
                "original_name": original,
                "final_match": final_match,
                "origin": origin,
            })
        # REJECT => no mapping included

    human_df = pd.DataFrame(human_rows)

    if not human_df.empty:
        key_cols = ["source_file", "original_name"]
        final = final.set_index(key_cols)
        human_df = human_df.set_index(key_cols)

        final.update(human_df)
        final = pd.concat([final, human_df[~human_df.index.isin(final.index)]], axis=0)
        final = final.reset_index()

    final = final.sort_values(["source_file", "original_name"]).reset_index(drop=True)
    return final


def summarize(df_clean: pd.DataFrame, hr_decisions: dict):
    counts = df_clean["status"].value_counts(dropna=False).to_dict() if "status" in df_clean.columns else {}

    human_accept = human_reject = human_edit = 0
    for d in hr_decisions.values():
        dec = str(d.get("decision", "")).upper().strip()
        if dec == "ACCEPT":
            human_accept += 1
            bm = str(d.get("best_match", "")).strip()
            fm = str(d.get("final_match", "")).strip()
            if bm and fm and bm != fm:
                human_edit += 1
        elif dec == "REJECT":
            human_reject += 1

    st.markdown("### ✅ Summary (skipped not shown)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Auto ACCEPT", int(counts.get("ACCEPT", 0)))
    c2.metric("HUMAN_REVIEW", int(counts.get("HUMAN_REVIEW", 0)))
    c3.metric("NO_MATCH", int(counts.get("NO_MATCH", 0)))

    st.markdown("---")
    c4, c5, c6 = st.columns(3)
    c4.metric("Human ACCEPT", human_accept)
    c5.metric("Human EDIT", human_edit)
    c6.metric("Human REJECT", human_reject)

# -----------------------------
# UI (Professional layout)
# -----------------------------
st.set_page_config(page_title="AnchorMap", layout="wide")
ensure_session_defaults()

# Header
st.markdown("# AnchorMap")
st.caption("Variable standardisation pipeline for maritime engineering artefacts.")

# Sidebar configuration
with st.sidebar:
    st.markdown("## Configuration")

    embed_option = st.radio(
        "Schema embeddings",
        ["Use existing embeddings", "Rebuild embeddings"],
        index=0,
    )

    st.markdown("---")

    ttl_files = sorted(INPUT_DIR.glob("*.ttl"))
    if not ttl_files:
        st.error(f"No .ttl files found in: {INPUT_DIR}")
        st.stop()

    ttl_names = [f.name for f in ttl_files]
    selected_ttl_name = st.selectbox("Input TTL file", ttl_names)
    selected_ttl = INPUT_DIR / selected_ttl_name

    st.markdown("---")

    run_clicked = st.button("Run pipeline", use_container_width=True)
    reset_clicked = st.button("Reset run", use_container_width=True)

if reset_clicked:
    reset_app_state()

# Embeddings status / action
embeddings_ok = EMB_VECS.exists() and EMB_IDS.exists() and EMB_TXTS.exists()

if embed_option == "Use existing embeddings":
    if embeddings_ok:
        st.success("Embeddings found.")
    else:
        st.warning("Embeddings not found. Switch to 'Rebuild embeddings'.")
else:
    colA, colB = st.columns([1, 3])
    with colA:
        rebuild = st.button("Rebuild embeddings", use_container_width=True)
    with colB:
        st.info("Rebuild only when the schema changes. Otherwise use existing embeddings.")

    if rebuild:
        with st.status("Rebuilding embeddings…", expanded=False) as status:
            rc, out = run_cmd([sys.executable, str(EMBEDDER)], cwd=ROOT)
            append_log(out)

            if rc == 0 and (EMB_VECS.exists() and EMB_IDS.exists() and EMB_TXTS.exists()):
                status.update(label="Embeddings rebuilt successfully.", state="complete")
            else:
                status.update(label="Embedding rebuild failed. See Logs.", state="error")

# Tabs for the rest of the app
tab_run, tab_review, tab_export, tab_logs = st.tabs(["Run", "Human review", "Export", "Logs"])

# -----------------------------
# Run tab
# -----------------------------
with tab_run:
    st.subheader("Run")
    workflow_banner()

    st.write("Selected input:", f"`{selected_ttl.name}`")

    if run_clicked:
        run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        run_dir = RUNS_DIR / f"run_{run_id}"
        out_dir = OUTPUT_DIR / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        results_csv = run_dir / "eval_results_ecms_onefile.csv"
        audit_csv = run_dir / "routing_audit_ecms_onefile.csv"
        clean_csv = run_dir / "eval_results_ecms_clean.csv"

        st.session_state.update({
            "run_started": True,
            "run_id": run_id,
            "run_dir": run_dir,
            "out_dir": out_dir,
            "selected_ttl": selected_ttl,
            "results_csv": results_csv,
            "audit_csv": audit_csv,
            "clean_csv": clean_csv,
            "final_mappings_csv": run_dir / "final_mappings.csv",
            "human_decisions_csv": run_dir / "human_decisions.csv",
            "df_clean": None,
            "hr_index": 0,
            "hr_decisions": {},
            "_clear_edit_box": True,
            "downloads_ready": False,
            "download_payloads": {},
        })

        with st.status("Running retrieval and routing…", expanded=False) as status:
            rc, out = run_cmd([
                sys.executable, str(MASTERAGENT),
                "--input_ttl", str(selected_ttl),
                "--out_csv", str(results_csv),
                "--audit_csv", str(audit_csv),
            ], cwd=ROOT)
            append_log(out)

            if rc != 0:
                status.update(label="Pipeline run failed. See Logs.", state="error")
            else:
                df = pd.read_csv(results_csv)
                df_clean = clean_skipped(df)
                df_clean.to_csv(clean_csv, index=False, encoding="utf-8")
                st.session_state["df_clean"] = df_clean
                status.update(label="Run completed.", state="complete")

    df_clean = st.session_state.get("df_clean")
    if st.session_state.get("run_started") and isinstance(df_clean, pd.DataFrame):
        st.markdown("### Routing overview")
        counts = df_clean["status"].value_counts()
        c1, c2, c3 = st.columns(3)
        c1.metric("ACCEPT", int(counts.get("ACCEPT", 0)))
        c2.metric("HUMAN_REVIEW", int(counts.get("HUMAN_REVIEW", 0)))
        c3.metric("NO_MATCH", int(counts.get("NO_MATCH", 0)))

        with st.expander("Preview results table", expanded=True):
            st.dataframe(df_clean, use_container_width=True)
    df_clean = st.session_state.get("df_clean")
    if st.session_state.get("run_started") and isinstance(df_clean, pd.DataFrame):
        hr_count = int((df_clean["status"].astype(str).str.strip() == "HUMAN_REVIEW").sum())

        if hr_count > 0:
            st.info(f"Next: go to the **Human review** tab to review {hr_count} item(s).")
        else:
            st.success("No human review needed. Next: go to the **Export** tab to finalize and download outputs.")

# -----------------------------
# Human Review tab
# -----------------------------
with tab_review:
    st.subheader("Human review")
    workflow_banner()

    df_clean = st.session_state.get("df_clean")
    if not (st.session_state.get("run_started") and isinstance(df_clean, pd.DataFrame)):
        st.info("Run the pipeline first.")
    else:
        hr = df_clean[df_clean["status"].astype(str).str.strip().eq("HUMAN_REVIEW")].copy().reset_index(drop=True)

        if hr.empty:
            st.success("No items require review.")
        else:
            idx = int(st.session_state.get("hr_index", 0))
            idx = max(0, min(idx, len(hr) - 1))
            st.session_state["hr_index"] = idx

            row = hr.iloc[idx]
            file_name = str(row.get("file", "")).strip()
            original = str(row.get("original_name", "")).strip()
            proposed_raw = row.get("best_match", "")
            proposed = "" if pd.isna(proposed_raw) else str(proposed_raw).strip()
            conf = row.get("confidence", "")

            key = f"{file_name}|||{original}"
            existing = st.session_state["hr_decisions"].get(key, {})

            decided_count = len(st.session_state["hr_decisions"])
            st.caption(f"Item {idx+1} of {len(hr)} • Decisions recorded: {decided_count}/{len(hr)}")

            st.markdown("**Source file**")
            st.code(file_name, language="text")

            st.markdown("**Original label**")
            st.code(original, language="text")

            cols = st.columns(2)
            cols[0].markdown("**Proposed match**")
            cols[0].code(proposed if proposed else "(none)", language="text")
            cols[1].markdown("**Confidence**")
            cols[1].code(str(conf), language="text")

            st.markdown("---")

            nav1, nav2, nav3, nav4 = st.columns([1, 1, 2, 2])
            with nav1:
                if st.button("Previous", use_container_width=True, disabled=(idx == 0)):
                    st.session_state["hr_index"] = max(0, idx - 1)
                    st.session_state["_clear_edit_box"] = True
                    st.rerun()
            with nav2:
                if st.button("Next", use_container_width=True, disabled=(idx == len(hr) - 1)):
                    st.session_state["hr_index"] = min(len(hr) - 1, idx + 1)
                    st.session_state["_clear_edit_box"] = True
                    st.rerun()

            # Decision buttons
            d1, d2 = st.columns(2)
            with d1:
                if st.button("Accept proposed", use_container_width=True, disabled=(not proposed)):
                    st.session_state["hr_decisions"][key] = {
                        "decision": "ACCEPT",
                        "final_match": proposed,
                        "best_match": proposed,
                        "confidence": conf,
                        "file": file_name,
                        "original_name": original,
                    }
                    if idx < len(hr) - 1:
                        st.session_state["hr_index"] += 1
                    st.session_state["_clear_edit_box"] = True
                    st.rerun()

            with d2:
                if st.button("Reject (no mapping)", use_container_width=True):
                    st.session_state["hr_decisions"][key] = {
                        "decision": "REJECT",
                        "final_match": "",
                        "best_match": proposed,
                        "confidence": conf,
                        "file": file_name,
                        "original_name": original,
                    }
                    if idx < len(hr) - 1:
                        st.session_state["hr_index"] += 1
                    st.session_state["_clear_edit_box"] = True
                    st.rerun()

            # -----------------------------
            # Override match (FIX: no yellow warning)
            # -----------------------------
            st.markdown("**Override match**")

            # Decide what the textbox should show for THIS item
            default_text = ""
            if existing.get("decision") == "ACCEPT":
                default_text = str(existing.get("final_match", "")).strip()

            # IMPORTANT:
            # Only set session_state BEFORE creating the widget, and never pass value= with key=
            if st.session_state.get("_clear_edit_box", False):
                st.session_state["edit_box"] = default_text
                st.session_state["_clear_edit_box"] = False
            elif "edit_box" not in st.session_state:
                st.session_state["edit_box"] = default_text  # first render

            edit_val = st.text_input(
                "Ontology identifier",
                key="edit_box",
                placeholder="e.g., rated_current",
            )

            if st.button("Apply override and accept", use_container_width=True):
                edit_val = (edit_val or "").strip()
                if not edit_val:
                    st.warning("Enter a non-empty ontology identifier.")
                else:
                    st.session_state["hr_decisions"][key] = {
                        "decision": "ACCEPT",
                        "final_match": edit_val,
                        "best_match": proposed,
                        "confidence": conf,
                        "file": file_name,
                        "original_name": original,
                    }
                    if idx < len(hr) - 1:
                        st.session_state["hr_index"] += 1

                    # Clear box on next item
                    st.session_state["_clear_edit_box"] = True
                    st.rerun()
            decided_count = len(st.session_state["hr_decisions"])
            total_to_review = len(hr)

            if decided_count >= total_to_review:
                st.success("All review decisions are recorded.")
                st.info("Next: go to the **Export** tab to finalize mappings and run the renamer.")
            else:
                st.caption(f"Continue reviewing until {total_to_review} decisions are recorded.")
                

# -----------------------------
# Export tab
# -----------------------------
with tab_export:
    st.subheader("Export")
    workflow_banner()

    df_clean = st.session_state.get("df_clean")
    if not (st.session_state.get("run_started") and isinstance(df_clean, pd.DataFrame)):
        st.info("Run the pipeline first.")
    else:
        hr = df_clean[df_clean["status"].astype(str).str.strip().eq("HUMAN_REVIEW")].copy().reset_index(drop=True)
        all_reviewed = (len(st.session_state.get("hr_decisions", {})) >= len(hr))

        st.write("Ready to finalize:", "Yes" if all_reviewed else "No")

        finalize = st.button(
            "Finalize mappings and run renamer",
            use_container_width=True,
            disabled=(not all_reviewed),
        )

        if finalize:
            run_dir: Path = st.session_state["run_dir"]
            out_dir: Path = st.session_state["out_dir"]
            selected_ttl: Path = st.session_state["selected_ttl"]
            final_mappings_csv: Path = st.session_state["final_mappings_csv"]
            human_decisions_csv: Path = st.session_state["human_decisions_csv"]

            final_df = build_final_mappings(df_clean, st.session_state.get("hr_decisions", {}))

            hr_rows = []
            for _, d in st.session_state.get("hr_decisions", {}).items():
                hr_rows.append({
                    "source_file": d.get("file", ""),
                    "original_name": d.get("original_name", ""),
                    "best_match": d.get("best_match", ""),
                    "decision": d.get("decision", ""),
                    "final_match": d.get("final_match", ""),
                    "confidence": d.get("confidence", ""),
                })

            pd.DataFrame(hr_rows).to_csv(human_decisions_csv, index=False, encoding="utf-8")
            final_df.to_csv(final_mappings_csv, index=False, encoding="utf-8")

            with st.status("Running renamer…", expanded=False) as status:
                rc, out = run_cmd([
                    sys.executable, str(RENAMER),
                    "--input_ttl", str(selected_ttl),
                    "--mappings_csv", str(final_mappings_csv),
                    "--output_dir", str(out_dir),
                    "--restrict_by_source_file",
                    "--run_id", f"run_{st.session_state['run_id']}",
                ], cwd=ROOT)
                append_log(out)

                if rc != 0:
                    status.update(label="Renamer failed. See Logs.", state="error")
                else:
                    status.update(label="Renamer completed.", state="complete")

                    summarize(df_clean, st.session_state.get("hr_decisions", {}))

                    payloads = {}
                    corrected = sorted(Path(out_dir).glob("*_corrected.ttl"))
                    for p in corrected:
                        payloads[p.name] = p.read_bytes()

                    payloads[final_mappings_csv.name] = final_mappings_csv.read_bytes()
                    payloads[Path(st.session_state["clean_csv"]).name] = Path(st.session_state["clean_csv"]).read_bytes()

                    st.session_state["download_payloads"] = payloads
                    st.session_state["downloads_ready"] = True

        st.markdown("### Downloads")
        if st.session_state.get("downloads_ready") and st.session_state.get("download_payloads"):
            for fname, fbytes in st.session_state["download_payloads"].items():
                mime = "text/turtle" if fname.endswith(".ttl") else "text/csv"
                st.download_button(
                    label=f"Download {fname}",
                    data=fbytes,
                    file_name=fname,
                    mime=mime,
                    key=f"dl_{st.session_state.get('run_id','')}_{fname}",
                    use_container_width=True,
                )
        else:
            st.caption("Finalize to generate outputs.")

# -----------------------------
# Logs tab
# -----------------------------
with tab_logs:
    st.subheader("Logs")
    workflow_banner()
    st.caption("Pipeline output (runtime messages and errors).")

    logs = st.session_state.get("logs", "").strip()

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Clear logs", use_container_width=True):
            st.session_state["logs"] = ""
            st.rerun()

    if not logs:
        st.info("No logs yet. Run the pipeline to see output here.")
    else:
        st.text_area("Output", value=logs, height=350)
        st.download_button(
            "Download logs (.txt)",
            data=logs.encode("utf-8"),
            file_name=f"anchormap_logs_{st.session_state.get('run_id','')}.txt",
            mime="text/plain",
            use_container_width=True,
        )