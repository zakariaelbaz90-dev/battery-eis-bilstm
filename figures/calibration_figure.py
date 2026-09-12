"""
figures/calibration_figure.py — regenerates Fig. 7 (calibration: reliability
diagram + expected calibration error), pooling all 8 LOCO folds for C1 and C3.

Usage:
    python figures/calibration_figure.py
Output:
    figures/output/fig_calibration.png
"""
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_dataset, load_fold, mc_predict  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(exist_ok=True)

d = load_dataset()
X_seq_np, X_tab_np = d["X_seq"], d["X_tab"]
y_SoC_norm_np, y_SoH_norm_np = d["y_SoC_norm"], d["y_SoH_norm"]
cell_ids, protocols = d["cell_ids"], d["protocols"]


def eval_all_folds(config, T=50, seed=321):
    all_mu_soc, all_sig_soc, all_y_soc = [], [], []
    all_mu_soh, all_sig_soh, all_y_soh = [], [], []
    for cell_id in range(1, 9):
        model, sc_seq, sc_tab = load_fold(config, cell_id)
        mask = cell_ids == cell_id
        use_ecm = config in ("C2", "C4", "C5")
        Xs = sc_seq.transform(X_seq_np[mask].reshape(-1, 3)).reshape(-1, 51, 3).astype(np.float32)
        Xt = sc_tab.transform(X_tab_np[mask]).astype(np.float32) if use_ecm else None
        x_seq_t = torch.tensor(Xs)
        x_tab_t = torch.tensor(Xt) if use_ecm else None
        preds = mc_predict(model, x_seq_t, x_tab_t, T=T, seed=seed)
        all_mu_soc.append(preds["mu_SoC"].numpy() * 100)
        all_sig_soc.append(preds["sigma_total_SoC"].numpy() * 100)
        all_y_soc.append(y_SoC_norm_np[mask] * 100)
        all_mu_soh.append(preds["mu_SoH"].numpy() * 100)
        all_sig_soh.append(preds["sigma_total_SoH"].numpy() * 100)
        all_y_soh.append(y_SoH_norm_np[mask] * 100)
    return (np.concatenate(all_mu_soc), np.concatenate(all_sig_soc), np.concatenate(all_y_soc),
            np.concatenate(all_mu_soh), np.concatenate(all_sig_soh), np.concatenate(all_y_soh))


def reliability_curve(mu, sigma, y, nominal_levels):
    empirical = []
    for p in nominal_levels:
        z = norm.ppf(0.5 + p / 2)
        cov = np.mean((y >= mu - z * sigma) & (y <= mu + z * sigma))
        empirical.append(cov)
    return np.array(empirical)


def ece(mu, sigma, y, nominal_levels):
    emp = reliability_curve(mu, sigma, y, nominal_levels)
    return float(np.mean(np.abs(emp - nominal_levels)))


nominal = np.linspace(0.05, 0.95, 10)
results = {}
for cfg in ["C1", "C3"]:
    mu_soc, sig_soc, y_soc, mu_soh, sig_soh, y_soh = eval_all_folds(cfg)
    results[cfg] = dict(mu_soc=mu_soc, sig_soc=sig_soc, y_soc=y_soc,
                         mu_soh=mu_soh, sig_soh=sig_soh, y_soh=y_soh)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

ax = axes[0]
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
for cfg, color in [("C1", "#9ca3af"), ("C3", "#2563eb")]:
    r = results[cfg]
    emp_soc = reliability_curve(r["mu_soc"], r["sig_soc"], r["y_soc"], nominal)
    ax.plot(nominal, emp_soc, marker="o", color=color, label=f"{cfg} SoC")
ax.set_xlabel("Nominal coverage")
ax.set_ylabel("Empirical coverage")
ax.set_title("(a) SoC reliability diagram")
ax.legend(fontsize=8)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)

ax = axes[1]
ece_vals = {}
for cfg in ["C1", "C3"]:
    r = results[cfg]
    ece_vals[(cfg, "SoC")] = ece(r["mu_soc"], r["sig_soc"], r["y_soc"], nominal)
    ece_vals[(cfg, "SoH")] = ece(r["mu_soh"], r["sig_soh"], r["y_soh"], nominal)
labels = ["C1 SoC", "C3 SoC", "C1 SoH", "C3 SoH"]
vals = [ece_vals[("C1", "SoC")], ece_vals[("C3", "SoC")], ece_vals[("C1", "SoH")], ece_vals[("C3", "SoH")]]
colors = ["#9ca3af", "#2563eb", "#d1d5db", "#93c5fd"]
ax.bar(labels, vals, color=colors)
ax.set_ylabel("Expected calibration error (ECE)")
ax.set_title("(b) ECE by state and configuration")
for i, v in enumerate(vals):
    ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)

plt.tight_layout()
out_path = OUT_DIR / "fig_calibration.png"
plt.savefig(out_path, dpi=200)
print(f"Saved: {out_path}")
print("SoC ECE: C1=%.3f C3=%.3f | SoH ECE: C1=%.3f C3=%.3f" % (
    ece_vals[("C1", "SoC")], ece_vals[("C3", "SoC")], ece_vals[("C1", "SoH")], ece_vals[("C3", "SoH")]))
