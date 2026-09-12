"""
common.py — shared model architecture, data loading, and MC-Dropout inference
utilities used by every script in figures/.

This is the same architecture and preprocessing as notebooks/bilstm_degradation.ipynb,
factored out so the figure-generation scripts don't each redefine it. If you retrain
or modify the model, keep this file and the training notebook in sync.

Paths are resolved relative to this file's location, so scripts in figures/ can
import it regardless of the current working directory:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from common import ...
"""
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent
DATA_PATH = REPO_ROOT / "data" / "leonardi_final.npz"
RESULTS_DIR = REPO_ROOT / "results"
CKPT_DIR = RESULTS_DIR / "bilstm_full_results"

device = torch.device("cpu")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_dataset():
    """Load the preprocessed tensor and per-cell/protocol metadata."""
    d = np.load(DATA_PATH, allow_pickle=True)
    return {
        "X_seq": d["X_seq"],
        "X_tab": d["X_tab"],
        "y_SoC_norm": d["y_SoC_norm"],
        "y_SoH_norm": d["y_SoH_norm"],
        "cell_ids": d["cell_ids"],
        "protocols": d["protocols"],
        "y_CL": d["y_CL"],
        "y_LLI": d["y_LLI"],
        "y_LAM": d["y_LAM"],
    }


def load_deg_stats():
    """CL/LLI/LAM min-max normalisation stats, saved alongside the LOCO results."""
    with open(RESULTS_DIR / "all_results.pkl", "rb") as f:
        saved = pickle.load(f)
    return saved["deg_stats"], saved


# ---------------------------------------------------------------------------
# Model (must match notebooks/bilstm_degradation.ipynb exactly)
# ---------------------------------------------------------------------------
class LSTMBranch(nn.Module):
    """Bidirectional LSTM over the 51-frequency EIS spectrum."""

    def __init__(self, in_channels=3, d_model=64, hidden_size=96, n_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        self.lstm = nn.LSTM(
            input_size=d_model, hidden_size=hidden_size, num_layers=n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0.0, bidirectional=True,
        )
        self.output_proj = nn.Linear(hidden_size * 2, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.input_proj(x)
        x, _ = self.lstm(x)
        x = x.mean(dim=1)
        x = self.output_proj(x)
        x = self.dropout(x)
        return x


class MLPBranch(nn.Module):
    """Dense MLP over the 9 fitted ECM parameters (used by C2/C4/C5 only)."""

    def __init__(self, in_features=9, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(p=dropout),
        )

    def forward(self, x):
        return self.net(x)


class BiLSTM_Config(nn.Module):
    """Configurable BiLSTM model. use_ecm=True adds the MLPBranch (C2/C4/C5);
    use_deg=True adds the auxiliary CL/LLI/LAM regression head (C3/C4/C5)."""

    def __init__(self, d_model=64, hidden_size=96, n_layers=2, dropout=0.1, use_ecm=False, use_deg=False):
        super().__init__()
        self.use_ecm = use_ecm
        self.use_deg = use_deg
        self.branch_a = LSTMBranch(3, d_model, hidden_size, n_layers, dropout)
        if use_ecm:
            self.branch_b = MLPBranch(9, d_model, dropout)
            fusion_in = d_model * 2
        else:
            self.branch_b = None
            fusion_in = d_model

        self.fusion_SoC = nn.Sequential(
            nn.Linear(fusion_in, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(p=dropout),
            nn.Linear(d_model, d_model // 2), nn.LayerNorm(d_model // 2), nn.GELU(), nn.Dropout(p=dropout),
        )
        self.fusion_SoH = nn.Sequential(
            nn.Linear(fusion_in, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(p=dropout)
        )
        self.head_SoC = nn.Linear(d_model // 2, 2)  # (mu, log_sigma)
        self.head_SoH = nn.Linear(d_model, 2)
        self.head_deg = nn.Linear(fusion_in, 3) if use_deg else None  # (CL, LLI, LAM)

        with torch.no_grad():
            self.head_SoC.bias[1] = -3.0
            self.head_SoH.bias[1] = -3.0

    def forward(self, x_seq, x_tab=None):
        h_a = self.branch_a(x_seq)
        if self.use_ecm and x_tab is not None:
            h = torch.cat([h_a, self.branch_b(x_tab)], dim=-1)
        else:
            h = h_a
        out_SoC = self.head_SoC(self.fusion_SoC(h))
        out_SoH = self.head_SoH(self.fusion_SoH(h))
        mu_SoC = torch.sigmoid(out_SoC[:, 0])
        ls_SoC = out_SoC[:, 1]
        mu_SoH = torch.sigmoid(out_SoH[:, 0]) * 0.23 + 0.77
        ls_SoH = out_SoH[:, 1]
        deg_pred = torch.sigmoid(self.head_deg(h)) if self.head_deg is not None else None
        return mu_SoC, ls_SoC, mu_SoH, ls_SoH, deg_pred


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------
def load_fold(config, cell_id, ckpt_dir=None):
    """Load a trained LOCO fold's model + its fitted scalers.

    config: one of 'C1','C2','C3','C4','C5'
    cell_id: 1..8 (the cell held out in this fold)
    """
    ckpt_dir = Path(ckpt_dir) if ckpt_dir is not None else CKPT_DIR
    use_ecm = config in ("C2", "C4", "C5")
    use_deg = config in ("C3", "C4", "C5")
    model = BiLSTM_Config(use_ecm=use_ecm, use_deg=use_deg)
    sd = torch.load(ckpt_dir / config / f"model_cell{cell_id:02d}.pt", map_location="cpu")
    model.load_state_dict(sd)
    with open(ckpt_dir / config / "scalers.pkl", "rb") as f:
        scalers = pickle.load(f)
    sc_seq, sc_tab = scalers[cell_id]["seq"], scalers[cell_id]["tab"]
    return model, sc_seq, sc_tab


# ---------------------------------------------------------------------------
# MC-Dropout inference
# ---------------------------------------------------------------------------
def mc_predict(model, x_seq, x_tab, T=50, seed=None):
    """Run T stochastic forward passes with dropout active, and decompose the
    predictive variance into epistemic (across-pass) and aleatoric (mean of the
    per-pass heteroscedastic variance) components.

    IMPORTANT: if `seed` is given, the global torch RNG state is saved before
    seeding and restored afterward. Without this, seeding here would silently
    contaminate the weight initialisation of any model trained *after* this
    call in the same process -- this exact bug once made a "multi-seed" LOPO
    training run produce identical results for 3 of 4 conditions. Do not
    remove the save/restore even if it looks redundant.
    """
    if seed is not None:
        rng_state = torch.get_rng_state()
        torch.manual_seed(seed)

    model.train()  # keep dropout active -- this IS the point of MC-Dropout
    mus_soc, mus_soh, vars_soc, vars_soh, deg_preds = [], [], [], [], []
    with torch.no_grad():
        for _ in range(T):
            mu_s, ls_s, mu_h, ls_h, deg = model(x_seq, x_tab)
            ls_s = torch.clamp(ls_s, -6.0, 2.0)
            ls_h = torch.clamp(ls_h, -6.0, 2.0)
            mus_soc.append(mu_s)
            mus_soh.append(mu_h)
            vars_soc.append(torch.exp(2.0 * ls_s))
            vars_soh.append(torch.exp(2.0 * ls_h))
            if deg is not None:
                deg_preds.append(deg)

    mus_soc = torch.stack(mus_soc)
    mus_soh = torch.stack(mus_soh)
    vars_soc = torch.stack(vars_soc)
    vars_soh = torch.stack(vars_soh)

    # Epistemic variance is undefined (identically zero) for T=1: variance of a
    # single sample requires >=2 samples to be non-trivial. This is reported
    # explicitly in the paper rather than silently treated as "small".
    epi_soc = mus_soc.var(dim=0) if T > 1 else torch.zeros_like(mus_soc[0])
    epi_soh = mus_soh.var(dim=0) if T > 1 else torch.zeros_like(mus_soh[0])
    alea_soc = vars_soc.mean(dim=0)
    alea_soh = vars_soh.mean(dim=0)

    if seed is not None:
        torch.set_rng_state(rng_state)

    return {
        "mu_SoC": mus_soc.mean(dim=0),
        "mu_SoH": mus_soh.mean(dim=0),
        "sigma_epi_SoC": epi_soc.sqrt(),
        "sigma_epi_SoH": epi_soh.sqrt(),
        "sigma_alea_SoC": alea_soc.sqrt(),
        "sigma_alea_SoH": alea_soh.sqrt(),
        "sigma_total_SoC": (epi_soc + alea_soc).sqrt(),
        "sigma_total_SoH": (epi_soh + alea_soh).sqrt(),
        "deg_pred": torch.stack(deg_preds).mean(dim=0) if deg_preds else None,
    }


def picp(mu, sigma, y, z=1.645):
    """Prediction interval coverage probability at ~90% nominal (z=1.645)."""
    return float(np.mean((y >= mu - z * sigma) & (y <= mu + z * sigma)))
