#!/usr/bin/env python3
"""
ECMS Evaluation Figures (clean + information-dense)

What you get (only 2 core figures):
  1) Confusion Matrix (AUTO ONLY: ACCEPT vs NO_MATCH) with explicit TP/FP/FN/TN legend
  2) Routing Outcome Breakdown (ACCEPT / NO_MATCH / HUMAN_REVIEW) showing proportions
     - ACCEPT: Correct vs Wrong
     - NO_MATCH: Correct Reject vs Should-Have-Accepted
     - HUMAN_REVIEW: Should Accept vs Should Reject (NOT included in confusion matrix)

Also prints:
  - precision / recall / F1 (auto-only)
  - accuracy, specificity, NPV, balanced accuracy, MCC (auto-only)
  - counts and key breakdowns
"""

import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.patheffects as pe

# ---------------- CONFIG ----------------
RESULTS_CSV = Path("eval_results_ecms_clean.csv")
GT_XLSX = Path("CorrectNamesMappings.xlsx")
GT_SHEET = None  # set sheet name if needed
OUT_DIR = Path("eval_outputs")
OUT_DIR.mkdir(exist_ok=True)

DONTEXIST_TOKEN = "DONTEXIST"

# ---------------- LOAD DATA ----------------
res = pd.read_csv(RESULTS_CSV)

# Normalize key columns
for c in ["original_name", "best_match", "status"]:
    if c in res.columns:
        res[c] = res[c].astype(str).str.strip()

# Clean best_match empties
res["best_match"] = (
    res["best_match"]
    .replace({"nan": "", "None": ""})
    .fillna("")
    .astype(str)
    .str.strip()
)

# ---------------- LOAD GROUND TRUTH (robust) ----------------
gt_raw = pd.read_excel(GT_XLSX, sheet_name=GT_SHEET)

if isinstance(gt_raw, dict):
    first_sheet = next(iter(gt_raw.keys()))
    print(f"ℹ️ sheet_name=None -> using first sheet: {first_sheet}")
    gt_df = gt_raw[first_sheet]
else:
    gt_df = gt_raw

gt_df = gt_df.iloc[:, :2].copy()
gt_df.columns = ["original_name", "correct_match"]
gt_df["original_name"] = gt_df["original_name"].astype(str).str.strip()
gt_df["correct_match"] = (
    gt_df["correct_match"].astype(str).str.strip()
    .replace({"nan": "", "None": ""})
    .fillna("")
    .astype(str).str.strip()
)

gt_map = (
    gt_df.dropna(subset=["original_name"])
         .set_index("original_name")["correct_match"]
         .to_dict()
)
print(f"✅ Ground truth mappings loaded: {len(gt_map)}")

# ---------------- ATTACH GROUND TRUTH ----------------
res["correct_match"] = res["original_name"].map(gt_map).fillna("").astype(str).str.strip()
res["has_gt"] = res["correct_match"].ne("")

# DONTEXIST handling
res["is_dontexist"] = res["correct_match"].str.upper().eq(DONTEXIST_TOKEN)
# "Actual positive" = GT says there is a real mapping (not DONTEXIST and not empty)
res["has_gt_pos"] = res["has_gt"] & (~res["is_dontexist"]) & res["correct_match"].ne("")

missing_in_gt = res.loc[~res["has_gt"], "original_name"].unique().tolist()
(OUT_DIR / "missing_in_ground_truth.txt").write_text("\n".join(missing_in_gt))
print(f"⚠️ Missing in ground truth: {len(missing_in_gt)} (saved to missing_in_ground_truth.txt)")

# ---------------- HELPERS ----------------
def is_correct_prediction(row) -> bool:
    """Correct only if GT-positive exists AND best_match equals correct_match."""
    if not bool(row.get("has_gt_pos", False)):
        return False
    return str(row["best_match"]).strip() == str(row["correct_match"]).strip()

def should_accept(row) -> bool:
    """Should accept iff GT-positive exists (i.e., correct_match is a real mapping)."""
    return bool(row.get("has_gt_pos", False))

# ---------------- SPLITS ----------------
accepted = res[res["status"] == "ACCEPT"].copy()
human = res[res["status"] == "HUMAN_REVIEW"].copy()
nomatch = res[res["status"] == "NO_MATCH"].copy()

# ACCEPT -> correct vs wrong
accepted["is_correct"] = accepted.apply(is_correct_prediction, axis=1)
accepted_correct = int(accepted["is_correct"].sum())
accepted_wrong = int(len(accepted) - accepted_correct)

# NO_MATCH -> correct reject vs should-have-accepted
nomatch["should_accept"] = nomatch.apply(should_accept, axis=1)
rejected_should_accept = int(nomatch["should_accept"].sum())          # false rejects (FN)
rejected_correct_reject = int(len(nomatch) - rejected_should_accept)  # correct rejects (TN-ish bucket)

# HUMAN_REVIEW -> should accept vs should reject (not in confusion)
human["should_accept"] = human.apply(should_accept, axis=1)
human_should_accept = int(human["should_accept"].sum())
human_should_reject = int(len(human) - human_should_accept)

# Totals
n_accept = int(len(accepted))
n_human = int(len(human))
n_nomatch = int(len(nomatch))
n_total = int(len(res))

# ============================================================
# ✅ AUTO-ONLY CONFUSION MATRIX (EXCLUDES HUMAN_REVIEW)
# Predicted Positive = ACCEPT
# Predicted Negative = NO_MATCH
# Actual Positive = GT-positive exists (correct mapping exists)
# Actual Negative = DONTEXIST OR missing GT OR empty GT
# ============================================================
auto = res[res["status"].isin(["ACCEPT", "NO_MATCH"])].copy()

def classify_auto(row):
    actual_pos = bool(row.get("has_gt_pos", False))
    pred_pos = (row["status"] == "ACCEPT")

    if pred_pos:
        # ACCEPT: TP if correct mapping else FP (includes DONTEXIST accepts)
        if actual_pos and is_correct_prediction(row):
            return "TP"
        else:
            return "FP"
    else:
        # NO_MATCH: FN if actual_pos else TN
        if actual_pos:
            return "FN"
        else:
            return "TN"

auto["confusion_auto"] = auto.apply(classify_auto, axis=1)
counts_auto = auto["confusion_auto"].value_counts()

TP = int(counts_auto.get("TP", 0))
FP = int(counts_auto.get("FP", 0))
FN = int(counts_auto.get("FN", 0))
TN = int(counts_auto.get("TN", 0))

# ---------------- METRICS (auto-only) ----------------
def safe_div(a, b):
    return (a / b) if b else 0.0

precision = safe_div(TP, TP + FP)
recall = safe_div(TP, TP + FN)  # sensitivity
f1 = safe_div(2 * precision * recall, precision + recall)

accuracy = safe_div(TP + TN, TP + TN + FP + FN)
specificity = safe_div(TN, TN + FP)  # TNR
npv = safe_div(TN, TN + FN)

balanced_accuracy = 0.5 * (recall + specificity)

# Matthews Correlation Coefficient
mcc_den = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
mcc = safe_div((TP * TN - FP * FN), mcc_den)

# ---------------- REPORT ----------------
report = f"""
================ ECMS EVALUATION =================

Total evaluated rows: {n_total}
Ground-truth-covered rows: {int(res['has_gt'].sum())}
GT-positive rows (real mappings): {int(res['has_gt_pos'].sum())}
Missing in GT: {len(missing_in_gt)}

AUTO-ONLY CONFUSION (EXCLUDES HUMAN_REVIEW)
Confusion:
  TP: {TP}
  FP: {FP}
  FN: {FN}
  TN: {TN}

Auto-only metrics:
  Precision:         {precision:.3f}
  Recall (TPR):      {recall:.3f}
  F1-score:          {f1:.3f}
  Accuracy:          {accuracy:.3f}
  Specificity (TNR): {specificity:.3f}
  NPV:               {npv:.3f}
  Balanced Acc:      {balanced_accuracy:.3f}
  MCC:               {mcc:.3f}

Routing totals:
  ACCEPT:       {n_accept}
  HUMAN_REVIEW: {n_human}
  NO_MATCH:     {n_nomatch}

Routing breakdowns:
  ACCEPT -> correct: {accepted_correct}
  ACCEPT -> wrong:   {accepted_wrong}

  NO_MATCH -> correct reject:        {rejected_correct_reject}
  NO_MATCH -> should-have-accepted:  {rejected_should_accept}

  HUMAN_REVIEW -> should accept: {human_should_accept}
  HUMAN_REVIEW -> should reject: {human_should_reject}

==================================================
"""
print(report)
(OUT_DIR / "evaluation_report.txt").write_text(report)


# ---------------- FIGURE: CONFUSION MATRIX (AUTO ONLY) ----------------
cm = np.array([[TP, FN],
               [FP, TN]], dtype=int)

fig, ax = plt.subplots(figsize=(5.6, 4.8))
im = ax.imshow(cm, cmap="Blues")  # classic

# ticks + labels (paper standard)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["ACCEPT", "NO_MATCH"], fontsize=12)
ax.set_yticklabels(["Positive", "Negative"], fontsize=12)

ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("Actual", fontsize=12)
ax.set_title("Confusion Matrix", fontsize=16, pad=10)

# draw cell borders (clean grid look)
ax.set_xticks(np.arange(-.5, 2, 1), minor=True)
ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
ax.tick_params(which="minor", bottom=False, left=False)

# annotate with counts only (simple + professional)
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]),
                ha="center", va="center",
                fontsize=14, color="black")

# optional colorbar (paper standard is either yes or no; keep if you like)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=10)

plt.tight_layout()
plt.savefig(OUT_DIR / "figure_confusion_matrix_auto.png", dpi=300, bbox_inches="tight")
plt.close()


# ---------------- FIGURE 2: ROUTING OUTCOME BREAKDOWN (PROPORTIONS) ----------------
# Remove note; move legend outside so it never overlaps.
cats = ["ACCEPT", "NO_MATCH", "HUMAN_REVIEW"]
bucket_totals = np.array([n_accept, n_nomatch, n_human], dtype=float)

good = np.array([accepted_correct, rejected_correct_reject, human_should_accept], dtype=float)
bad  = np.array([accepted_wrong,   rejected_should_accept,  human_should_reject], dtype=float)

good_pct = np.where(bucket_totals > 0, 100.0 * good / bucket_totals, 0.0)
bad_pct  = np.where(bucket_totals > 0, 100.0 * bad  / bucket_totals, 0.0)

fig, ax = plt.subplots(figsize=(10.0, 5.6))
x = np.arange(len(cats))

ax.bar(x, good_pct, label="Correct / Should-Accept")
ax.bar(x, bad_pct, bottom=good_pct, label="Wrong / Should-Reject")

ax.set_xticks(x)
ax.set_xticklabels([
    f"ACCEPT\n(n={n_accept})",
    f"NO_MATCH\n(n={n_nomatch})",
    f"HUMAN_REVIEW\n(n={n_human})"
])
ax.set_ylim(0, 100)
ax.set_ylabel("Proportion within bucket (%)")
ax.set_title("Routing Outcomes (Proportions)")

# Annotate with counts (keep, but slightly smaller)
for i in range(len(cats)):
    if bucket_totals[i] > 0:
        ax.text(i, good_pct[i] / 2, f"{int(good[i])}", ha="center", va="center", fontsize=11)
        ax.text(i, good_pct[i] + bad_pct[i] / 2, f"{int(bad[i])}", ha="center", va="center", fontsize=11)

# Legend outside to avoid overlap
ax.legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=False,
    borderaxespad=0.0,
    fontsize=10
)

plt.tight_layout()
plt.savefig(OUT_DIR / "figure_routing_outcomes_proportions.png", dpi=300, bbox_inches="tight")
plt.close()


# ---------------- SAVE OUTPUT FILES ----------------
res.to_csv(OUT_DIR / "eval_results_scored.csv", index=False)
auto.to_csv(OUT_DIR / "eval_results_auto_only_scored.csv", index=False)

routing_summary = pd.DataFrame({
    "status": ["ACCEPT", "NO_MATCH", "HUMAN_REVIEW"],
    "total":  [n_accept, n_nomatch, n_human],
    "segment_1_label": ["Correct ACCEPT", "Correct Reject", "Should Accept"],
    "segment_1_count": [accepted_correct, rejected_correct_reject, human_should_accept],
    "segment_2_label": ["Wrong ACCEPT", "Should Have Accepted", "Should Reject"],
    "segment_2_count": [accepted_wrong, rejected_should_accept, human_should_reject],
})
routing_summary.to_csv(OUT_DIR / "routing_summary.csv", index=False)

print("✅ Evaluation complete. Outputs saved in:", OUT_DIR.resolve())
print("🖼️ Figures saved:")
print("  - figure_confusion_matrix_auto.png")
print("  - figure_routing_outcomes_proportions.png")
print("📄 Reports saved:")
print("  - evaluation_report.txt")
print("  - missing_in_ground_truth.txt")
print("  - routing_summary.csv")
print("  - eval_results_scored.csv")
print("  - eval_results_auto_only_scored.csv")
