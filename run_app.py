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
# UI
# -----------------------------
st.set_page_config(page_title="AnchorMap GUI", layout="wide")
ensure_session_defaults()

st.title("🚀 AnchorMap GUI: Multi-Agent Variable Standardisation Pipeline")

# -----------------------------
# Step 1: Embeddings
# -----------------------------
st.header("1️⃣ Schema Indexing Agent (Embeddings / Anchor Space)")

embed_option = st.radio(
    "Choose embedding option:",
    ["Use existing embeddings (recommended)", "Rebuild embeddings now"],
    index=0,
)

if embed_option.startswith("Use existing"):
    if EMB_VECS.exists() and EMB_IDS.exists() and EMB_TXTS.exists():
        st.success("✅ Found embeddings in repo root.")
    else:
        st.warning("⚠️ Embeddings not found in root. Please rebuild embeddings.")
else:
    if st.button("Run Schema Indexing Agent", width="stretch"):
        with st.spinner("Rebuilding embeddings..."):
            rc, out = run_cmd([sys.executable, str(EMBEDDER)], cwd=ROOT)
            append_log(out)
        if rc == 0 and EMB_VECS.exists() and EMB_IDS.exists() and EMB_TXTS.exists():
            st.success("✅ Embeddings rebuilt successfully!")
        else:
            st.error("❌ Embedding rebuild failed. Check logs below.")


# -----------------------------
# Step 2: TTL selection
# -----------------------------
st.header("2️⃣ Select Input TTL")

ttl_files = sorted(INPUT_DIR.glob("*.ttl"))
if not ttl_files:
    st.error(f"No .ttl files found in: {INPUT_DIR}")
    st.stop()

ttl_names = [f.name for f in ttl_files]
selected_ttl_name = st.selectbox("Choose a TTL file", ttl_names)
selected_ttl = INPUT_DIR / selected_ttl_name


# -----------------------------
# Step 3: Run pipeline up to masteragent + clean
# -----------------------------
st.header("3️⃣ Retrieval and Reasoning Agent + Confidence Routing Agent ")

run_clicked = st.button("Run Retrieval and Reasoning Agent", width="stretch")
if run_clicked:
    run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = RUNS_DIR / f"run_{run_id}"
    out_dir = OUTPUT_DIR / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_csv = run_dir / "eval_results_ecms_onefile.csv"
    audit_csv = run_dir / "routing_audit_ecms_onefile.csv"
    clean_csv = run_dir / "eval_results_ecms_clean.csv"

    st.session_state["run_started"] = True
    st.session_state["run_id"] = run_id
    st.session_state["run_dir"] = run_dir
    st.session_state["out_dir"] = out_dir
    st.session_state["selected_ttl"] = selected_ttl

    st.session_state["results_csv"] = results_csv
    st.session_state["audit_csv"] = audit_csv
    st.session_state["clean_csv"] = clean_csv

    st.session_state["final_mappings_csv"] = run_dir / "final_mappings.csv"
    st.session_state["human_decisions_csv"] = run_dir / "human_decisions.csv"

    st.session_state["df_clean"] = None
    st.session_state["hr_index"] = 0
    st.session_state["hr_decisions"] = {}
    st.session_state["_clear_edit_box"] = True  # clear when HR first opens

    with st.spinner("Retrieval and Reasoning Agent running..."):
        rc, out = run_cmd([
            sys.executable, str(MASTERAGENT),
            "--input_ttl", str(selected_ttl),
            "--out_csv", str(results_csv),
            "--audit_csv", str(audit_csv),
        ], cwd=ROOT)
        append_log(out)

    if rc != 0:
        st.error("❌ Retrieval and Reasoning Agent failed. Check logs below.")
    else:
        df = pd.read_csv(results_csv)
        df_clean = clean_skipped(df)
        df_clean.to_csv(clean_csv, index=False, encoding="utf-8")
        st.session_state["df_clean"] = df_clean
        st.success("✅ Retrieval and Reasoning Agent completed and results saved.")


df_clean = st.session_state.get("df_clean")
if st.session_state.get("run_started") and isinstance(df_clean, pd.DataFrame):
    st.subheader("📊 Routing Summary")
    st.write(df_clean["status"].value_counts())

    st.subheader("📄 Results Preview ")
    st.dataframe(df_clean, width="stretch")


# -----------------------------
# Step 4: Human review (GUI)
# -----------------------------
st.header("4️⃣ Human Review")

if not (st.session_state.get("run_started") and isinstance(df_clean, pd.DataFrame)):
    st.info("Run the Retrieval and Reasoning Agent first.")
else:
    hr = df_clean[df_clean["status"].astype(str).str.strip().eq("HUMAN_REVIEW")].copy().reset_index(drop=True)

    if hr.empty:
        st.success("✅ No HUMAN_REVIEW items. You can finalize and run the Variable Renaming Agent.")
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

        k = f"{file_name}|||{original}"
        existing = st.session_state["hr_decisions"].get(k, {})

        if len(st.session_state["hr_decisions"]) >= len(hr):
            st.success("✅ All HUMAN_REVIEW items have decisions recorded.")
            st.info("Go to Step 5 to finalize mappings and run the Variable Renaming Agent.")
        else:
            st.markdown(f"**Item {idx+1} / {len(hr)}**")
            st.markdown(f"**File:** `{file_name}`")
            st.markdown(f"**Original:** `{original}`")
            st.markdown(f"**Proposed:** `{proposed if proposed else '(empty)'}`")
            st.markdown(f"**Confidence:** `{conf}`")
            st.markdown("---")

            nav1, nav2, _ = st.columns([1, 1, 3])
            with nav1:
                if st.button("⬅️ Prev", width="stretch"):
                    st.session_state["hr_index"] = max(0, idx - 1)
                    st.session_state["_clear_edit_box"] = True
                    st.rerun()
            with nav2:
                if st.button("Next ➡️", width="stretch"):
                    st.session_state["hr_index"] = min(len(hr) - 1, idx + 1)
                    st.session_state["_clear_edit_box"] = True
                    st.rerun()

            colA, colB = st.columns([1, 1])
            with colA:
                if st.button("✅ Accept Proposed", width="stretch", disabled=(not proposed)):
                    st.session_state["hr_decisions"][k] = {
                        "decision": "ACCEPT",
                        "final_match": proposed,
                        "best_match": proposed,
                        "confidence": conf,
                        "file": file_name,
                        "original_name": original,
                    }
                    if st.session_state["hr_index"] < len(hr) - 1:
                        st.session_state["hr_index"] += 1
                    st.session_state["_clear_edit_box"] = True
                    st.rerun()

            with colB:
                if st.button("❌ Reject (No mapping)", width="stretch"):
                    st.session_state["hr_decisions"][k] = {
                        "decision": "REJECT",
                        "final_match": "",
                        "best_match": proposed,
                        "confidence": conf,
                        "file": file_name,
                        "original_name": original,
                    }
                    if st.session_state["hr_index"] < len(hr) - 1:
                        st.session_state["hr_index"] += 1
                    st.session_state["_clear_edit_box"] = True
                    st.rerun()

            st.markdown("**✍️ Edit (enter ontology id and accept)**")

            # Clear edit box safely BEFORE widget creation
            if st.session_state.get("_clear_edit_box", False):
                st.session_state["edit_box"] = ""
                st.session_state["_clear_edit_box"] = False

            edit_val = st.text_input(
                "Corrected match (ontology id)",
                value=str(existing.get("final_match", "")) if existing.get("decision") == "ACCEPT" else "",
                key="edit_box",
                placeholder="e.g., rated_current",
            )

            if st.button("Apply Edit (Accept)", width="stretch"):
                edit_val = (edit_val or "").strip()
                if not edit_val:
                    st.warning("Type a non-empty ontology id to accept.")
                else:
                    st.session_state["hr_decisions"][k] = {
                        "decision": "ACCEPT",
                        "final_match": edit_val,
                        "best_match": proposed,
                        "confidence": conf,
                        "file": file_name,
                        "original_name": original,
                    }
                    if st.session_state["hr_index"] < len(hr) - 1:
                        st.session_state["hr_index"] += 1
                    st.session_state["_clear_edit_box"] = True
                    st.rerun()

        decided_count = len(st.session_state["hr_decisions"])
        st.info(f"Decisions recorded: {decided_count} / {len(hr)}")


# -----------------------------
# Step 5: Finalize + Rename
# -----------------------------
st.header("5️⃣ Finalize + Run Variable Renaming Agent")

if not (st.session_state.get("run_started") and isinstance(df_clean, pd.DataFrame)):
    st.info("Run Step 3 first.")
else:
    finalize = st.button("Finalize + Run Renamer", width="stretch")
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

        st.success(f"✅ Saved final mappings: {final_mappings_csv.name}")
        st.success(f"✅ Saved human decisions: {human_decisions_csv.name}")

        with st.spinner("Running variable renamer..."):
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
            st.error("❌ Variable Renaming failed. Check logs below.")
        else:
            st.success("✅ Renamer completed.")
            summarize(df_clean, st.session_state.get("hr_decisions", {}))
                        # ---- Cache download payloads so downloads survive reruns ----
            payloads = {}

            # corrected TTLs
            corrected = sorted(Path(out_dir).glob("*_corrected.ttl"))
            for p in corrected:
                payloads[p.name] = p.read_bytes()

            # mappings + cleaned results
            payloads[final_mappings_csv.name] = final_mappings_csv.read_bytes()
            payloads[Path(st.session_state["clean_csv"]).name] = Path(st.session_state["clean_csv"]).read_bytes()

            st.session_state["download_payloads"] = payloads
            st.session_state["downloads_ready"] = True
            # -----------------------------
# Downloads (stable across reruns)
# -----------------------------
            st.header("📥 Downloads")

            if st.session_state.get("downloads_ready") and st.session_state.get("download_payloads"):
                for fname, fbytes in st.session_state["download_payloads"].items():
                    mime = "text/turtle" if fname.endswith(".ttl") else "text/csv"
                    st.download_button(
                        label=f"Download {fname}",
                        data=fbytes,
                        file_name=fname,
                        mime=mime,
                        key=f"dl_{st.session_state.get('run_id','')}_{fname}",
                    )
            else:
                st.info("Run Step 5 to generate outputs, then download them here.")
           


# -----------------------------
# Logs
# -----------------------------
st.header("🪵 Logs")
st.text_area("Pipeline logs (stdout/stderr)", value=st.session_state.get("logs", ""), height=260)