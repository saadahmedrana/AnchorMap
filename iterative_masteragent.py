#!/usr/bin/env python3
"""
Sweep HUMAN_REVIEW_THR values without touching your original scripts.

For each HR threshold, this script:
  1) runs a patched copy of masteragent_ecms.py (only HR threshold + output paths changed)
  2) cleans the produced CSV (drops SKIPPED_NOT_IN_STANDARD)
  3) runs a patched copy of eval_results_ecms.py (only input CSV + OUT_DIR changed)
  4) extracts key metrics and writes a sweep summary CSV

Outputs:
  sweep_results/
    hr_0.40/
      eval_results_ecms_onefile.csv
      routing_audit_ecms_onefile.csv
      eval_results_ecms_clean.csv
      eval_outputs/   (figures + reports + scored CSVs)
    hr_0.45/
    ...
  sweep_results/sweep_summary.csv
"""

import re
import sys
import subprocess
from pathlib import Path
import pandas as pd


# ---------------- USER CONFIG ----------------
HR_THRESHOLDS = [0.41]

MASTERAGENT = "masteragent_ecms.py"
EVAL_SCRIPT = "eval_results_ecms.py"
GT_XLSX = "CorrectNamesMappings.xlsx"  # used by eval script, leave as-is

SWEEP_ROOT = "sweep_results"  # folder created next to scripts
# --------------------------------------------


def run(cmd, cwd: Path):
    """Run command and stream output. Raise if fails."""
    print("\n$ " + " ".join(map(str, cmd)))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def patch_masteragent(src_text: str, hr_thr: float, out_csv: Path, audit_csv: Path) -> str:
    # Replace only the HUMAN_REVIEW_THR assignment (first occurrence)
    txt = re.sub(
        r"(?m)^(HUMAN_REVIEW_THR\s*=\s*)([0-9]*\.?[0-9]+)\s*$",
        rf"\g<1>{hr_thr:.2f}",
        src_text,
        count=1,
    )

    # Patch OUT_CSV and AUDIT_CSV to absolute paths so outputs never overwrite
    txt = re.sub(
        r"(?m)^OUT_CSV\s*=\s*os\.path\.join\(HERE,\s*\"eval_results_ecms_onefile\.csv\"\)\s*$",
        f'OUT_CSV   = r"{out_csv.as_posix()}"',
        txt,
        count=1,
    )
    txt = re.sub(
        r"(?m)^AUDIT_CSV\s*=\s*os\.path\.join\(HERE,\s*\"routing_audit_ecms_onefile\.csv\"\)\s*$",
        f'AUDIT_CSV = r"{audit_csv.as_posix()}"',
        txt,
        count=1,
    )
    return txt


def patch_eval(src_text: str, results_csv: Path, out_dir: Path) -> str:
    # Patch RESULTS_CSV = Path("...")  -> absolute
    txt = re.sub(
        r"(?m)^RESULTS_CSV\s*=\s*Path\(\"[^\"]+\"\)\s*$",
        f'RESULTS_CSV = Path(r"{results_csv.as_posix()}")',
        src_text,
        count=1,
    )

    # Patch OUT_DIR = Path("eval_outputs") -> absolute
    txt = re.sub(
        r"(?m)^OUT_DIR\s*=\s*Path\(\"eval_outputs\"\)\s*$",
        f'OUT_DIR = Path(r"{out_dir.as_posix()}")',
        txt,
        count=1,
    )
    return txt


def extract_metrics_from_report(report_path: Path) -> dict:
    """
    Parses the evaluation_report.txt created by eval_results_ecms.py.
    Returns dict with TP/FP/FN/TN, precision, recall, f1, accuracy, etc.
    """
    text = report_path.read_text(encoding="utf-8", errors="ignore")

    def grab(pattern, cast=float, default=None):
        m = re.search(pattern, text)
        return cast(m.group(1)) if m else default

    metrics = {
        "TP": grab(r"\bTP:\s*(\d+)", int),
        "FP": grab(r"\bFP:\s*(\d+)", int),
        "FN": grab(r"\bFN:\s*(\d+)", int),
        "TN": grab(r"\bTN:\s*(\d+)", int),
        "precision": grab(r"Precision:\s*([0-9.]+)", float),
        "recall": grab(r"Recall.*?:\s*([0-9.]+)", float),
        "f1": grab(r"F1-score:\s*([0-9.]+)", float),
        "accuracy": grab(r"Accuracy:\s*([0-9.]+)", float),
        "specificity": grab(r"Specificity.*?:\s*([0-9.]+)", float),
        "npv": grab(r"\bNPV:\s*([0-9.]+)", float),
        "balanced_acc": grab(r"Balanced Acc:\s*([0-9.]+)", float),
        "mcc": grab(r"\bMCC:\s*([0-9.\-]+)", float),
        "ACCEPT_total": grab(r"\bACCEPT:\s*(\d+)", int),
        "HUMAN_REVIEW_total": grab(r"\bHUMAN_REVIEW:\s*(\d+)", int),
        "NO_MATCH_total": grab(r"\bNO_MATCH:\s*(\d+)", int),
        "ACCEPT_correct": grab(r"ACCEPT -> correct:\s*(\d+)", int),
        "ACCEPT_wrong": grab(r"ACCEPT -> wrong:\s*(\d+)", int),
        "NO_MATCH_correct": grab(r"NO_MATCH -> correct reject:\s*(\d+)", int),
        "NO_MATCH_should_accept": grab(r"NO_MATCH -> should-have-accepted:\s*(\d+)", int),
        "HR_should_accept": grab(r"HUMAN_REVIEW -> should accept:\s*(\d+)", int),
        "HR_should_reject": grab(r"HUMAN_REVIEW -> should reject:\s*(\d+)", int),
    }
    return metrics


def main():
    base_dir = Path(__file__).resolve().parent
    sweep_root = base_dir / SWEEP_ROOT
    sweep_root.mkdir(exist_ok=True)

    masteragent_path = base_dir / MASTERAGENT
    eval_path = base_dir / EVAL_SCRIPT

    if not masteragent_path.exists():
        raise FileNotFoundError(masteragent_path)
    if not eval_path.exists():
        raise FileNotFoundError(eval_path)

    masteragent_src = masteragent_path.read_text(encoding="utf-8")
    eval_src = eval_path.read_text(encoding="utf-8")

    summary_rows = []

    for thr in HR_THRESHOLDS:
        tag = f"hr_{thr:.2f}"
        run_dir = sweep_root / tag
        run_dir.mkdir(parents=True, exist_ok=True)

        out_onefile = run_dir / "eval_results_ecms_onefile.csv"
        out_audit = run_dir / "routing_audit_ecms_onefile.csv"
        out_clean = run_dir / "eval_results_ecms_clean.csv"
        out_eval_dir = run_dir / "eval_outputs"
        out_eval_dir.mkdir(exist_ok=True)

        print("\n" + "=" * 90)
        print(f"▶ SWEEP RUN: HUMAN_REVIEW_THR = {thr:.2f}  →  {run_dir}")
        print("=" * 90)

        # --- 1) Patched masteragent ---
        tmp_master = base_dir / "_tmp_masteragent_sweep.py"
        tmp_master.write_text(
            patch_masteragent(masteragent_src, thr, out_onefile, out_audit),
            encoding="utf-8",
        )

        run([sys.executable, str(tmp_master)], cwd=base_dir)

        # --- 2) Clean CSV (same logic as clean_csv.py) ---
        df = pd.read_csv(out_onefile)
        df_clean = df[df["status"] != "SKIPPED_NOT_IN_STANDARD"].copy()
        df_clean.to_csv(out_clean, index=False, encoding="utf-8")
        print(f"✅ Cleaned: removed {len(df) - len(df_clean)} skipped rows -> {out_clean.name}")

        # --- 3) Patched eval script ---
        tmp_eval = base_dir / "_tmp_eval_sweep.py"
        tmp_eval.write_text(
            patch_eval(eval_src, out_clean, out_eval_dir),
            encoding="utf-8",
        )

        run([sys.executable, str(tmp_eval)], cwd=base_dir)

        # --- 4) Collect metrics ---
        report_path = out_eval_dir / "evaluation_report.txt"
        if not report_path.exists():
            raise FileNotFoundError(f"Expected report not found: {report_path}")

        metrics = extract_metrics_from_report(report_path)
        metrics["human_review_thr"] = thr
        metrics["run_dir"] = str(run_dir)
        summary_rows.append(metrics)

        print(f"✅ Logged metrics for HR={thr:.2f}: precision={metrics.get('precision')}, recall={metrics.get('recall')}")

    # cleanup temp scripts (leave if you prefer)
    for p in [base_dir / "_tmp_masteragent_sweep.py", base_dir / "_tmp_eval_sweep.py"]:
        try:
            p.unlink()
        except Exception:
            pass

    # Write sweep summary
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = sweep_root / "sweep_summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")
    print("\n" + "=" * 90)
    print("✅ Sweep complete.")
    print(f"📄 Summary saved -> {summary_csv}")
    print("=" * 90)


if __name__ == "__main__":
    main()
