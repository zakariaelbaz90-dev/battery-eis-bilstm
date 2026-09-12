"""
figures/lopo_figure.py — regenerates Fig. 9 (LOCO vs. LOPO accuracy and
detectability comparison, AP3).

Unlike tsweep_figure.py and calibration_figure.py, this does NOT retrain or
re-run inference -- the LOPO models themselves take real training time (that's
what ../lopo_training.ipynb does, from scratch, 5 seeds x {C1,C3} x
{AP3-out,AP4-out}). This script only reads the results that notebook already
saved to results/lopo_results_multiseed.pkl and plots them.

Run ../lopo_training.ipynb first if that file doesn't exist yet.

Usage:
    python figures/lopo_figure.py
Output:
    figures/output/fig_lopo.png
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import RESULTS_DIR  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(exist_ok=True)

RESULTS_PATH = RESULTS_DIR / "lopo_results_multiseed.pkl"
if not RESULTS_PATH.exists():
    raise FileNotFoundError(
        f"{RESULTS_PATH} not found. Run ../lopo_training.ipynb first to train the "
        "LOPO models and produce this file -- it is not something this plotting "
        "script can generate on its own."
    )

with open(RESULTS_PATH, "rb") as f:
    all_results = pickle.load(f)
df = pd.DataFrame(all_results)

summary = df.groupby(["config", "held_out"]).agg(
    ood_mae_mean=("ood_mae", "mean"), ood_mae_std=("ood_mae", "std"),
    auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"),
    n_seeds=("train_seed", "nunique"),
).reset_index()
print(summary.to_string(index=False))

# LOCO reference values -- from results/all_results.pkl (Table 3 in the paper).
# Filled in manually here since they come from a different results file/pipeline
# (the standard 8-fold LOCO run, not the LOPO run) and are already verified
# against the paper's Table 3.
LOCO_REFERENCE = {
    ("C1", "AP3"): {"mae": 5.07, "auroc": 0.69},
    ("C3", "AP3"): {"mae": 4.24, "auroc": 0.67},
}

configs_order = ["C1", "C3"]
x = np.arange(len(configs_order))
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

ax = axes[0]
loco_mae = [LOCO_REFERENCE[(c, "AP3")]["mae"] for c in configs_order]
lopo_mae = [summary[(summary.config == c) & (summary.held_out == "AP3")]["ood_mae_mean"].values[0] for c in configs_order]
lopo_err = [summary[(summary.config == c) & (summary.held_out == "AP3")]["ood_mae_std"].values[0] for c in configs_order]
ax.bar(x - width / 2, loco_mae, width, label="LOCO", color="#93c5fd")
ax.bar(x + width / 2, lopo_mae, width, yerr=lopo_err, capsize=4,
       label=f"LOPO ({int(summary['n_seeds'].iloc[0])}-seed mean+/-std)", color="#1d4ed8")
ax.set_xticks(x)
ax.set_xticklabels(configs_order)
ax.set_ylabel("AP3 SoC MAE (%)")
ax.set_title("(a) Accuracy: LOCO vs. LOPO (AP3)")
ax.legend(fontsize=8)

ax = axes[1]
loco_auroc = [LOCO_REFERENCE[(c, "AP3")]["auroc"] for c in configs_order]
lopo_auroc = [summary[(summary.config == c) & (summary.held_out == "AP3")]["auroc_mean"].values[0] for c in configs_order]
lopo_auroc_err = [summary[(summary.config == c) & (summary.held_out == "AP3")]["auroc_std"].values[0] for c in configs_order]
ax.bar(x - width / 2, loco_auroc, width, label="LOCO", color="#fca5a5")
ax.bar(x + width / 2, lopo_auroc, width, yerr=lopo_auroc_err, capsize=4,
       label=f"LOPO ({int(summary['n_seeds'].iloc[0])}-seed mean+/-std)", color="#dc2626")
ax.axhline(0.5, ls="--", color="gray", lw=1, label="Chance")
ax.set_xticks(x)
ax.set_xticklabels(configs_order)
ax.set_ylabel("AUROC (AP3 vs. ID)")
ax.set_ylim(0, 0.75)
ax.set_title("(b) OOD detectability: LOCO vs. LOPO (AP3)")
ax.legend(fontsize=8)

plt.tight_layout()
out_path = OUT_DIR / "fig_lopo.png"
plt.savefig(out_path, dpi=200)
print(f"\nSaved: {out_path}")
