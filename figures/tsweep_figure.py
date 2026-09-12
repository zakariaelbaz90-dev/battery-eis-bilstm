"""
figures/tsweep_figure.py — regenerates Fig. 6 (MC-Dropout sample count T sweep).

Re-runs inference on the already-trained C3 checkpoints at T in {1,5,10,25,50},
three independent seeds each. No retraining involved -- this is pure re-inference,
so it runs in well under a minute on CPU.

Usage (from anywhere; paths are resolved relative to the repo root):
    python figures/tsweep_figure.py
Output:
    figures/output/fig_tsweep.png
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_dataset, load_fold, mc_predict, picp, REPO_ROOT  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(exist_ok=True)

Ts = [1, 5, 10, 25, 50]
SEEDS = [11, 22, 33]
DETERMINISTIC_MAE = 4.58  # from Table 4 / results/mc_vs_det_results.pkl (C3, T=1 no dropout)

d = load_dataset()
X_seq_np, X_tab_np = d["X_seq"], d["X_tab"]
y_SoC_norm_np = d["y_SoC_norm"]
cell_ids, protocols = d["cell_ids"], d["protocols"]


def eval_fold_at_T(cell_id, T, seed):
    model, sc_seq, _ = load_fold("C3", cell_id)
    mask = cell_ids == cell_id
    Xs = sc_seq.transform(X_seq_np[mask].reshape(-1, 3)).reshape(-1, 51, 3).astype(np.float32)
    x_seq_t = np.asarray(Xs)
    import torch
    preds = mc_predict(model, torch.tensor(x_seq_t), None, T=T, seed=seed)
    mu = preds["mu_SoC"].numpy() * 100
    y = y_SoC_norm_np[mask] * 100
    sig_epi = preds["sigma_epi_SoC"].numpy() * 100
    sig_tot = preds["sigma_total_SoC"].numpy() * 100
    mae = float(np.mean(np.abs(mu - y)))
    return mae, sig_epi, mu, y, sig_tot, protocols[mask][0]


agg = {T: {"mae": [], "id": [], "ood": [], "repi": [], "auroc": [], "picp": []} for T in Ts}

for T in Ts:
    for seed in SEEDS:
        per_fold_mae = {}
        sig_by_prot = {"AP1": [], "AP2": [], "AP3": [], "AP4": []}
        all_mu, all_y, all_sig = [], [], []
        for cell_id in range(1, 9):
            mae, sig_epi, mu, y, sig_tot, prot = eval_fold_at_T(cell_id, T, seed + cell_id)
            per_fold_mae[cell_id] = mae
            sig_by_prot[prot].append(sig_epi)
            all_mu.append(mu)
            all_y.append(y)
            all_sig.append(sig_tot)
        maes = np.array(list(per_fold_mae.values()))
        agg[T]["mae"].append(maes.mean())
        agg[T]["id"].append(maes[[0, 1, 2, 3]].mean())
        agg[T]["ood"].append(maes[[4, 5, 6, 7]].mean())
        id_sig = np.concatenate(sig_by_prot["AP1"] + sig_by_prot["AP2"])
        ap3_sig = np.concatenate(sig_by_prot["AP3"])
        agg[T]["repi"].append(ap3_sig.mean() / id_sig.mean() if T > 1 else np.nan)
        y_true = np.concatenate([np.zeros(len(id_sig)), np.ones(len(ap3_sig))])
        y_score = np.concatenate([id_sig, ap3_sig])
        agg[T]["auroc"].append(roc_auc_score(y_true, y_score) if T > 1 else 0.5)
        agg[T]["picp"].append(picp(np.concatenate(all_mu), np.concatenate(all_sig), np.concatenate(all_y)))


def m(T, k):
    return np.mean(agg[T][k])


def s(T, k):
    return np.std(agg[T][k])


fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

ax = axes[0]
ax.errorbar(Ts, [m(T, "mae") for T in Ts], yerr=[s(T, "mae") for T in Ts], marker="o", color="#2563eb", capsize=3)
ax.axhline(DETERMINISTIC_MAE, ls="--", color="gray", label=f"Deterministic ({DETERMINISTIC_MAE}%)")
ax.set_xscale("log")
ax.set_xlabel("MC samples $T$")
ax.set_ylabel("Overall SoC MAE (%)")
ax.set_title("(a) Accuracy vs. $T$")
ax.legend(fontsize=8)
ax.set_xticks(Ts)
ax.set_xticklabels(Ts)

ax = axes[1]
Ts_epi = [T for T in Ts if T > 1]
ax.errorbar(Ts_epi, [m(T, "repi") for T in Ts_epi], yerr=[s(T, "repi") for T in Ts_epi],
            marker="s", color="#dc2626", label=r"$r_{epi}$ (AP3/ID)")
ax.set_xscale("log")
ax.set_xlabel("MC samples $T$")
ax.set_ylabel(r"$r_{epi}$", color="#dc2626")
ax2 = ax.twinx()
ax2.errorbar(Ts, [m(T, "auroc") for T in Ts], yerr=[s(T, "auroc") for T in Ts], marker="^", color="#059669", label="AUROC")
ax2.set_ylabel("AUROC", color="#059669")
ax2.axhline(0.5, ls=":", color="gray", lw=1)
ax.set_title("(b) Epistemic ratio & AUROC vs. $T$\n(undefined at $T{=}1$: needs $T{\\geq}2$ samples)", fontsize=9)
ax.set_xticks(Ts)
ax.set_xticklabels(Ts)

ax = axes[2]
ax.errorbar(Ts, [m(T, "picp") for T in Ts], yerr=[s(T, "picp") for T in Ts], marker="d", color="#7c3aed", capsize=3)
ax.axhline(0.90, ls="--", color="gray", label="Nominal 0.90")
ax.set_xscale("log")
ax.set_xlabel("MC samples $T$")
ax.set_ylabel("PICP$_{SoC}$")
ax.set_title("(c) Coverage vs. $T$")
ax.legend(fontsize=8)
ax.set_xticks(Ts)
ax.set_xticklabels(Ts)

plt.tight_layout()
out_path = OUT_DIR / "fig_tsweep.png"
plt.savefig(out_path, dpi=200)
print(f"Saved: {out_path}")

print(f"\n{'T':>3} {'MAE%':>8} {'ID%':>7} {'OOD%':>7} {'r_epi':>8} {'AUROC':>7} {'PICP':>6}")
for T in Ts:
    repi_val = m(T, "repi") if T > 1 else float("nan")
    print(f"{T:>3} {m(T,'mae'):>7.2f}% {m(T,'id'):>6.2f}% {m(T,'ood'):>6.2f}% "
          f"{repi_val:>8.3f} {m(T,'auroc'):>7.3f} {m(T,'picp'):>6.3f}")
