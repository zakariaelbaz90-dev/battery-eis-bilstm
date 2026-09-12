"""
figures/ecm_shuffle_diagnostic.py — substantiates the "representation shortcut"
claim in Section 3.1: does C2 actually rely on the ECM branch, or is it only
weakly influenced by it?

Test: re-evaluate the trained C2 model with each spectrum's true ECM vector
replaced by another sample's ECM vector from the same test set, keeping the
spectrum itself and its true SoC label unchanged. If C2 only weakly used the
ECM branch, accuracy should degrade modestly toward C1's level (no ECM branch
at all). If it degrades far past C1, the model has come to depend on the ECM
input rather than merely being biased by it.

This does not retrain anything -- it re-runs inference on the existing C2 and
C1 checkpoints, so it's fast (well under a minute on CPU).

Usage:
    python figures/ecm_shuffle_diagnostic.py
Output:
    Printed table (no figure) -- these numbers back the sentence in Section 3.1
    substantiating the shortcut mechanism.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_dataset, load_fold, mc_predict  # noqa: E402

d = load_dataset()
X_seq_np, X_tab_np = d["X_seq"], d["X_tab"]
y_SoC_norm_np = d["y_SoC_norm"]
cell_ids, protocols = d["cell_ids"], d["protocols"]


def eval_fold(config, cell_id, shuffle_ecm=False, seed=0):
    model, sc_seq, sc_tab = load_fold(config, cell_id)
    mask = cell_ids == cell_id
    Xs = sc_seq.transform(X_seq_np[mask].reshape(-1, 3)).reshape(-1, 51, 3).astype(np.float32)
    X_tab_this = X_tab_np[mask].copy()
    if shuffle_ecm:
        rng = np.random.RandomState(seed)
        # Permute ECM rows *within this fold's test set*, breaking the
        # correspondence between each spectrum and its own true fitted ECM
        # parameters, while every spectrum keeps its own correct SoC label.
        perm = rng.permutation(X_tab_this.shape[0])
        X_tab_this = X_tab_this[perm]
    Xt = sc_tab.transform(X_tab_this).astype(np.float32)
    x_seq_t = torch.tensor(Xs)
    x_tab_t = torch.tensor(Xt)
    preds = mc_predict(model, x_seq_t, x_tab_t, T=50, seed=777)
    mu_soc = preds["mu_SoC"].numpy() * 100
    y_soc = y_SoC_norm_np[mask] * 100
    return float(np.mean(np.abs(mu_soc - y_soc))), protocols[mask][0]


def eval_fold_no_ecm(config, cell_id):
    """For C1, which has no ECM branch at all (x_tab is simply not used)."""
    model, sc_seq, _ = load_fold(config, cell_id)
    mask = cell_ids == cell_id
    Xs = sc_seq.transform(X_seq_np[mask].reshape(-1, 3)).reshape(-1, 51, 3).astype(np.float32)
    preds = mc_predict(model, torch.tensor(Xs), None, T=50, seed=777)
    mu_soc = preds["mu_SoC"].numpy() * 100
    y_soc = y_SoC_norm_np[mask] * 100
    return float(np.mean(np.abs(mu_soc - y_soc))), protocols[mask][0]


print(f"{'Cell':>4} {'Prot':>5} {'C2 normal':>10} {'C2 shuffled-ECM':>16} {'C1 (no ECM, ref.)':>18}")
normal_all, shuffled_all, c1_all = [], [], []
id_normal, id_shuffled = [], []
ood_normal, ood_shuffled = [], []
for cell_id in range(1, 9):
    mae_normal, prot = eval_fold("C2", cell_id, shuffle_ecm=False)
    mae_shuf, _ = eval_fold("C2", cell_id, shuffle_ecm=True, seed=123)
    mae_c1, _ = eval_fold_no_ecm("C1", cell_id)
    print(f"{cell_id:>4} {prot:>5} {mae_normal:>9.2f}% {mae_shuf:>15.2f}% {mae_c1:>17.2f}%")
    normal_all.append(mae_normal)
    shuffled_all.append(mae_shuf)
    c1_all.append(mae_c1)
    (id_normal if prot in ("AP1", "AP2") else ood_normal).append(mae_normal)
    (id_shuffled if prot in ("AP1", "AP2") else ood_shuffled).append(mae_shuf)

print(f"\nOverall  C2 normal:   {np.mean(normal_all):.2f}%")
print(f"Overall  C2 shuffled: {np.mean(shuffled_all):.2f}%")
print(f"Overall  C1 (no ECM): {np.mean(c1_all):.2f}%")
print(f"ID       C2 normal: {np.mean(id_normal):.2f}%   shuffled: {np.mean(id_shuffled):.2f}%")
print(f"OOD      C2 normal: {np.mean(ood_normal):.2f}%   shuffled: {np.mean(ood_shuffled):.2f}%")
print(
    "\nInterpretation: if C2 only weakly used the ECM branch, shuffled accuracy "
    "should land near C1's level. A collapse far past C1 indicates the model "
    "depends on the ECM input rather than being merely biased by it."
)
