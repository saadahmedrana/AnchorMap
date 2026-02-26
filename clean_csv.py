import pandas as pd
from pathlib import Path

IN_FILE = Path("eval_results_ecms_onefile.csv")   # change if needed
OUT_FILE = Path("eval_results_ecms_clean.csv")

df = pd.read_csv(IN_FILE)

# Drop skipped rows
df_clean = df[df["status"] != "SKIPPED_NOT_IN_STANDARD"]

df_clean.to_csv(OUT_FILE, index=False, encoding="utf-8")

print(f"✅ Removed {len(df) - len(df_clean)} skipped rows")
print(f"✅ Saved cleaned file -> {OUT_FILE}")
