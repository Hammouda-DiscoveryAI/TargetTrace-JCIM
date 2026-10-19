import copy
import gc
import itertools
import json
import pickle
import random
import time as _time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from sklearn.metrics import (
    average_precision_score, precision_recall_curve,
    roc_auc_score, roc_curve,
)
from scipy.stats import pearsonr as _pearsonr, spearmanr as _spearmanr
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from database import get_training_data, log_run, mark_all_trained
from embedders import (ESM_DIM, _PROT_POOLED, _load, _seq_key,
                       embed_proteins, load_residues_subset, unload_models)
from features import FP_DIM, _load_fp_cache, load_fp_subset, precompute_fps
from model import TargetTrace3

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP    = device.type == "cuda"
_N_WORKERS = 0   # DataLoader workers: 0 avoids fork-from-thread deadlock

_ZERO_ESM = np.zeros(ESM_DIM, dtype=np.float32)


# ── Focal loss ────────────────────────────────────────────────────────────

def _focal_loss(logits: torch.Tensor, targets: torch.Tensor,
                gamma: float = 2.0) -> torch.Tensor:
    """
    Focal loss (Lin et al. 2017).  Down-weights easy examples so the gradient
    focuses on the hard boundary cases that determine AUC.
    FL = -(1 - p_t)^gamma * log(p_t)
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = torch.exp(-bce)
    return ((1.0 - p_t) ** gamma * bce).mean()


# ── Evaluation helpers ─────────────────────────────────────────────────────

def _ef(y_true, y_score, frac: float) -> float:
    """Enrichment factor at `frac` of the ranked list."""
    y_true  = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n_actives = int(y_true.sum())
    if n_actives == 0:
        return 1.0
    n_top = max(1, int(frac * len(y_true)))
    top_idx = np.argsort(y_score)[::-1][:n_top]
    return float((y_true[top_idx].sum() / n_top) / (n_actives / len(y_true)))


def _bedroc(y_true, y_score, alpha: float = 20.0) -> float:
    """BEDROC — Boltzmann-weighted enrichment at early recovery (alpha=20 standard)."""
    try:
        from rdkit.ML.Scoring.Scoring import CalcBEDROC
        pairs = sorted(zip(y_score, y_true), reverse=True)
        return float(CalcBEDROC([[s, l] for s, l in pairs], 1, alpha))
    except Exception:
        return float("nan")


# ── Helpers ────────────────────────────────────────────────────────────────

def _seed_everything(seed: int) -> torch.Generator:
    """Seed python, numpy and torch; returns a torch Generator for DataLoader shuffling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _fast_stratified_sample(df: pd.DataFrame, max_samples: int,
                            seed: int = 42) -> pd.DataFrame:
    """Vectorized stratified sample — keeps target distribution intact."""
    frac = max_samples / len(df)
    rng  = np.random.default_rng(seed)
    df   = df.copy()
    df["_r"]  = rng.random(len(df))
    df["_rk"] = df.groupby("target_name", sort=False)["_r"].rank(method="first")
    counts    = df["target_name"].value_counts()
    df["_kp"] = df["target_name"].map(counts).mul(frac).clip(lower=1).round().astype("int32")
    return df.loc[df["_rk"] <= df["_kp"]].drop(columns=["_r","_rk","_kp"]).reset_index(drop=True)


def _scaffold_split(df: pd.DataFrame, val_frac: float = 0.2, seed: int = 42):
    """Train/val split by Murcko scaffold — no scaffold appears in both sets."""
    rng = np.random.default_rng(seed)

    smi_to_scaffold: dict = {}
    for smi in df["smiles"].unique():
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            smi_to_scaffold[smi] = smi
        else:
            try:
                sc = GetScaffoldForMol(mol)
                smi_to_scaffold[smi] = Chem.MolToSmiles(sc, isomericSmiles=False) if sc else smi
            except Exception:
                smi_to_scaffold[smi] = smi

    sc_col = df["smiles"].map(smi_to_scaffold)
    scaffold_counts = sc_col.value_counts(dropna=True)
    scaffolds = scaffold_counts.index.tolist()
    rng.shuffle(scaffolds)

    val_target    = int(val_frac * len(df))
    val_scaffolds: set = set()
    val_count     = 0
    for sc in scaffolds:
        if val_count < val_target:
            val_scaffolds.add(sc)
            val_count += int(scaffold_counts[sc])

    val_mask = sc_col.isin(val_scaffolds)
    return df[~val_mask].reset_index(drop=True), df[val_mask].reset_index(drop=True)


def _cold_target_split(df: pd.DataFrame, val_frac: float = 0.2, seed: int = 42):
    """Train/val split by protein target — no target appears in both sets."""
    rng     = np.random.default_rng(seed)
    targets = df["target_name"].unique().copy()
    rng.shuffle(targets)
    n_val       = max(1, int(val_frac * len(targets)))
    val_targets = set(targets[:n_val])
    val_mask    = df["target_name"].isin(val_targets)
    return df[~val_mask].reset_index(drop=True), df[val_mask].reset_index(drop=True)


def _double_cold_split(df: pd.DataFrame, val_frac: float = 0.2, seed: int = 42):
    """
    Double-cold split: independently designates cold scaffolds and cold targets.
    Val  = rows with a cold scaffold AND a cold target (truly unseen on both axes).
    Train = rows with a warm scaffold AND a warm target.
    Falls back to cold-target if too few double-cold pairs exist in the data.
    """
    rng = np.random.default_rng(seed)

    # Build scaffold labels (same logic as _scaffold_split)
    smi_to_sc: dict = {}
    for smi in df["smiles"].unique():
        mol = Chem.MolFromSmiles(str(smi))
        smi_to_sc[smi] = smi
        if mol is not None:
            try:
                sc = GetScaffoldForMol(mol)
                if sc:
                    smi_to_sc[smi] = Chem.MolToSmiles(sc, isomericSmiles=False)
            except Exception:
                pass

    df = df.copy()
    df["_sc"] = df["smiles"].map(smi_to_sc)

    # Independently designate cold scaffolds and cold targets
    scaffolds = df["_sc"].unique().copy()
    rng.shuffle(scaffolds)
    cold_scaffolds = set(scaffolds[: max(1, int(val_frac * len(scaffolds)))])

    targets = df["target_name"].unique().copy()
    np.random.default_rng(seed + 1).shuffle(targets)
    cold_targets = set(targets[: max(1, int(val_frac * len(targets)))])

    is_cold_sc  = df["_sc"].isin(cold_scaffolds)
    is_cold_tgt = df["target_name"].isin(cold_targets)

    val_df   = df[ is_cold_sc &  is_cold_tgt].drop(columns=["_sc"]).reset_index(drop=True)
    train_df = df[~is_cold_sc & ~is_cold_tgt].drop(columns=["_sc"]).reset_index(drop=True)

    if len(val_df) < 20 or len(train_df) < 20:
        # Intersection too sparse; degrade to cold-target and signal the caller
        return *_cold_target_split(df.drop(columns=["_sc"]), val_frac, seed), True

    return train_df, val_df, False


def _split_once(df: pd.DataFrame, split_mode: str, frac: float, seed: int):
    """One held-out split under split_mode. Returns (rest_df, held_df, fell_back)."""
    if split_mode == "cold_target":
        return (*_cold_target_split(df, val_frac=frac, seed=seed), False)
    if split_mode == "double_cold":
        return _double_cold_split(df, val_frac=frac, seed=seed)
    return (*_scaffold_split(df, val_frac=frac, seed=seed), False)


def _three_way_split(df: pd.DataFrame, split_mode: str,
                     val_frac: float, test_frac: float, seed: int):
    """
    Train / validation / test split, applying the same split rule twice:
    first the test partition is held out of the full data, then the validation
    partition is held out of the remainder. Validation is used for checkpoint
    selection and early stopping only; test is scored once, after training.
    Returns (train_df, val_df, test_df, fell_back).
    """
    rest_df, test_df, fb1 = _split_once(df, split_mode, test_frac, seed)
    val_frac_of_rest = val_frac / max(1e-9, 1.0 - test_frac)
    train_df, val_df, fb2 = _split_once(rest_df, split_mode, val_frac_of_rest, seed + 1)
    return train_df, val_df, test_df, (fb1 or fb2)


def _partition_summary(name: str, part: pd.DataFrame) -> dict:
    return {
        "partition": name,
        "pairs":     int(len(part)),
        "compounds": int(part["smiles"].nunique()),
        "targets":   int(part["target_name"].nunique()),
        "actives":   int((part["active"] == 1).sum()),
        "inactives": int((part["active"] == 0).sum()),
    }


def _load_caches_parallel(needed_smiles: set, status_cb=None) -> tuple:
    """Load FP subset and protein pool in parallel threads."""
    done = [0]

    def _load_fps():
        result = load_fp_subset(needed_smiles, status_cb=status_cb)
        done[0] += 1
        if status_cb:
            status_cb(f"Step 4/5 — Loading caches [{done[0]}/2] (fingerprints done)…")
        return result

    def _load_prots():
        result = _load(_PROT_POOLED)
        done[0] += 1
        if status_cb:
            status_cb(f"Step 4/5 — Loading caches [{done[0]}/2] (proteins done)…")
        return result

    with ThreadPoolExecutor(max_workers=2) as ex:
        f_fp   = ex.submit(_load_fps)
        f_pool = ex.submit(_load_prots)
        return f_fp.result(), f_pool.result()


# ── Dataset ────────────────────────────────────────────────────────────────

class TargetTrace3Dataset(Dataset):
    """
    DTI dataset: each sample is a positive (mol, protein) binding pair.
    Yields (fp, esm_pooled, prot_idx, pic50_norm).
    prot_idx indexes into the preloaded GPU residue tensor for efficient cross-attention.
    """

    def __init__(self, df, pic50_mean: float, pic50_std: float,
                 fp_cache: dict, prot_pool: dict,
                 prot_key_to_idx: dict, tgt_to_key: dict):

        self._fp:   dict[str, np.ndarray] = {}
        self._prot: dict[str, np.ndarray] = {}

        for smiles in df["smiles"].unique():
            fp = fp_cache.get(str(smiles))
            if fp is not None:
                self._fp[str(smiles)] = fp

        for key in prot_key_to_idx:
            self._prot[key] = prot_pool.get(key, _ZERO_ESM.copy()).astype(np.float32)

        del prot_pool
        gc.collect()

        valid_smiles  = set(self._fp.keys())
        valid_targets = set(tgt_to_key.keys())
        df2 = df[df["smiles"].isin(valid_smiles) &
                 df["target_name"].isin(valid_targets)].copy()

        df2["prot_key"] = df2["target_name"].map(tgt_to_key)
        df2["prot_idx"] = df2["prot_key"].map(prot_key_to_idx)
        df2["pic50_n"]  = (pd.to_numeric(df2["pic50"], errors="coerce")
                           .sub(pic50_mean).div(pic50_std))

        self._smiles_arr   = df2["smiles"].tolist()
        self._target_arr   = df2["target_name"].tolist()
        self._active_arr   = (df2["active"].to_numpy(dtype=np.int64)
                              if "active" in df2.columns
                              else np.ones(len(df2), dtype=np.int64))
        self._prot_key_arr = df2["prot_key"].tolist()
        self._prot_idx_arr = df2["prot_idx"].to_numpy(dtype=np.int32)
        self._pic50_arr    = df2["pic50_n"].to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self._smiles_arr)

    def __getitem__(self, idx):
        smiles   = self._smiles_arr[idx]
        prot_key = self._prot_key_arr[idx]
        return (
            torch.tensor(self._fp[smiles],     dtype=torch.float32),
            torch.tensor(self._prot[prot_key], dtype=torch.float32),
            int(self._prot_idx_arr[idx]),
            float(self._pic50_arr[idx]),
        )


def _evaluate(model, loader, ds: "TargetTrace3Dataset",
              prot_res_t, prot_mask_t, p_mean: float, p_std: float,
              seed: int, prot_idx_to_target: dict) -> tuple[dict, pd.DataFrame]:
    """
    Score a held-out partition (loader must not shuffle).

    Every row is scored against its own protein and labelled by its measured
    `active` value, so real inactives count as negatives. Each active row also
    gets one protein-shuffle decoy (the molecule paired with another row's
    protein, both residues AND pooled ESM-2 swapped), labelled 0; decoys that
    land on the same protein are dropped. Decoy draws come from a fixed-seed
    generator, so every epoch scores the same decoys.

    Metrics:
      AUC / AUPRC / EF / BEDROC — actives vs (real inactives + decoys);
                                   used for checkpoint selection
      AUC_real                  — actives vs real inactives only
      AUC_decoy                 — actives vs protein-shuffle decoys only
    Returns (metrics, per-pair prediction table).
    """
    model.eval()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    scores, decoy_rows = [], []
    p50_preds = []
    offset = 0
    with torch.no_grad():
        for fp, esm_p, prot_idx, pic50s in loader:
            B = len(fp)
            neg_perm = torch.randperm(B, generator=gen)
            fp, esm_p, prot_idx = [b.to(device, non_blocking=True)
                                   for b in [fp, esm_p, prot_idx]]
            neg_perm = neg_perm.to(device)

            with autocast(device_type="cuda", enabled=USE_AMP):
                act_pos, p50_pos = model(fp, esm_p, prot_res_t[prot_idx],
                                         prot_mask_t[prot_idx])
                neg_idx = prot_idx[neg_perm]
                act_neg, _ = model(fp, esm_p[neg_perm], prot_res_t[neg_idx],
                                   prot_mask_t[neg_idx])
            scores.extend(torch.sigmoid(act_pos).cpu().float().numpy().tolist())
            p50_preds.extend((p50_pos.cpu().float().numpy() * p_std + p_mean).tolist())

            neg_s    = torch.sigmoid(act_neg).cpu().float().numpy()
            neg_idx  = neg_idx.cpu().numpy()
            own_idx  = prot_idx.cpu().numpy()
            for j in range(B):
                row = offset + j
                if ds._active_arr[row] == 1 and neg_idx[j] != own_idx[j]:
                    decoy_rows.append((row, int(neg_idx[j]), float(neg_s[j])))
            offset += B

    scores  = np.asarray(scores, dtype=np.float64)
    labels  = ds._active_arr[:len(scores)].astype(int)
    pic50_t = ds._pic50_arr[:len(scores)] * p_std + p_mean

    pos_s  = scores[labels == 1]
    inact_s = scores[labels == 0]
    dec_s  = np.array([d[2] for d in decoy_rows], dtype=np.float64)

    def _auc(pos, neg):
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        return float(roc_auc_score(y, np.r_[pos, neg]))

    y_true_a  = np.r_[np.ones(len(pos_s)), np.zeros(len(inact_s) + len(dec_s))]
    y_score_a = np.r_[pos_s, inact_s, dec_s]
    has_both  = len(np.unique(y_true_a)) == 2

    m = {
        "AUC":       float(roc_auc_score(y_true_a, y_score_a)) if has_both else float("nan"),
        "AUPRC":     float(average_precision_score(y_true_a, y_score_a)) if has_both else float("nan"),
        "AUC_real":  _auc(pos_s, inact_s),
        "AUC_decoy": _auc(pos_s, dec_s),
        "EF1%":      _ef(y_true_a, y_score_a, 0.01),
        "EF5%":      _ef(y_true_a, y_score_a, 0.05),
        "BEDROC":    _bedroc(y_true_a, y_score_a),
        "n_actives":   int(len(pos_s)),
        "n_inactives": int(len(inact_s)),
        "n_decoys":    int(len(dec_s)),
    }
    if has_both:
        fpr, tpr, _ = roc_curve(y_true_a, y_score_a)
        step = max(1, len(fpr) // 200)
        m["_roc"] = list(zip(fpr[::step].tolist(), tpr[::step].tolist()))
        prec_c, rec_c, _ = precision_recall_curve(y_true_a, y_score_a)
        step = max(1, len(prec_c) // 200)
        m["_pr"] = list(zip(rec_c[::step].tolist(), prec_c[::step].tolist()))
    else:
        m["_roc"] = m["_pr"] = []

    # pIC50 regression on rows with a measured value, scored against own protein
    p50_preds = np.asarray(p50_preds, dtype=np.float64)
    ok = ~np.isnan(pic50_t)
    if ok.sum() >= 2:
        pt_a, pp_a = pic50_t[ok], p50_preds[ok]
        sr  = _spearmanr(pt_a, pp_a)
        pr_ = _pearsonr(pt_a, pp_a)
        m["pIC50 MAE"]  = float(np.mean(np.abs(pt_a - pp_a)))
        m["pIC50 RMSE"] = float(np.sqrt(np.mean((pt_a - pp_a) ** 2)))
        m["Spearman"]   = float(getattr(sr,  "statistic", getattr(sr,  "correlation", 0.0)))
        m["Pearson"]    = float(getattr(pr_, "statistic", pr_[0] if isinstance(pr_, tuple) else 0.0))
        n_s   = min(len(pt_a), 500)
        idx_s = (np.random.default_rng(seed).choice(len(pt_a), n_s, replace=False)
                 if len(pt_a) > n_s else np.arange(len(pt_a)))
        m["_scatter"] = list(zip(pt_a[idx_s].tolist(), pp_a[idx_s].tolist()))
    else:
        m["pIC50 MAE"] = m["pIC50 RMSE"] = m["Spearman"] = m["Pearson"] = 0.0
        m["_scatter"] = []

    n = len(scores)
    preds = pd.DataFrame({
        "smiles":      ds._smiles_arr[:n],
        "target_name": ds._target_arr[:n],
        "kind":        "measured",
        "label":       labels,
        "score":       scores,
        "pic50_true":  pic50_t,
        "pic50_pred":  p50_preds,
    })
    if decoy_rows:
        decoys = pd.DataFrame({
            "smiles":      [ds._smiles_arr[r] for r, _, _ in decoy_rows],
            "target_name": [prot_idx_to_target.get(pi, "") for _, pi, _ in decoy_rows],
            "kind":        "decoy",
            "label":       0,
            "score":       dec_s,
            "pic50_true":  np.nan,
            "pic50_pred":  np.nan,
        })
        preds = pd.concat([preds, decoys], ignore_index=True)
    return m, preds


def _collate(batch):
    fps, esm_ps, prot_idxs, pic50s = zip(*batch)
    return (
        torch.stack(fps),
        torch.stack(esm_ps),
        torch.tensor(prot_idxs, dtype=torch.long),
        torch.tensor(pic50s,    dtype=torch.float32),
    )


# ── Checkpoint helpers ─────────────────────────────────────────────────────

def _load_ckpt() -> TargetTrace3 | None:
    ckpts = sorted(MODELS_DIR.glob("*.pkl"))
    if not ckpts:
        return None
    try:
        with open(ckpts[-1], "rb") as f:
            data = pickle.load(f)
        m = data.get("model")
        if (m is None or not hasattr(m, "act_head")
                or not hasattr(m, "cross_attn") or not hasattr(m, "cross_attn_2")):
            return None
        return m
    except Exception:
        return None


def load_latest() -> tuple:
    """Returns (model, target_names, target_keys, target_prot_embs, norm_stats)."""
    ckpts = sorted(MODELS_DIR.glob("*.pkl"))
    if not ckpts:
        return None, [], [], None, {}
    try:
        with open(ckpts[-1], "rb") as f:
            data = pickle.load(f)
        m = data.get("model")
        if (m is None or not hasattr(m, "act_head")
                or not hasattr(m, "cross_attn") or not hasattr(m, "cross_attn_2")):
            return None, [], [], None, {}
        norm = data.get("norm_stats", {"pic50_mean": 7.0, "pic50_std": 1.5})
        return (
            m.to(device),
            data.get("target_names", []),
            data.get("target_keys",  []),
            data.get("target_prot_embs"),   # np.ndarray (N, 480) or None
            norm,
        )
    except Exception:
        return None, [], [], None, {}


# ── Training ───────────────────────────────────────────────────────────────

def train(
    epochs:      int = 10,
    max_samples: int | None = None,
    split_mode:  str = "scaffold",
    progress_cb: Callable[[int, int, float, float, float], None] | None = None,
    status_cb:   Callable[[str], None] | None = None,
    model_factory: Callable[[], TargetTrace3] | None = None,
    ckpt_dir:      Path | None = None,
    seed:          int = 0,
    split_seed:    int = 42,
    val_frac:      float = 0.1,
    test_frac:     float = 0.2,
    resume:        bool = False,
) -> tuple[TargetTrace3 | None, list, list, np.ndarray | None, dict, list[dict], str]:
    """
    Returns (model, target_names, target_keys, target_prot_embs, norm_stats, epoch_log, message).
    split_mode: "scaffold" | "cold_target" | "double_cold"

    Data are split three ways (train / validation / test, default 70/10/20).
    Checkpoint selection and early stopping use validation only; the test
    partition is scored once, with the best checkpoint, after training ends.
    Per-pair validation and test predictions, partition sizes and test
    metrics are written next to the checkpoint.

    seed:       training seed (weight init, batch order, negatives). Vary this
                for repeated runs on the same split.
    split_seed: seed for subsampling and the train/val/test split.
    resume:     warm-start from the latest checkpoint in MODELS_DIR (scaffold
                split, default model only). Off by default: a checkpoint trained
                on a different split may have seen this run's test pairs.

    model_factory: overrides the default TargetTrace3() constructor — e.g. to
        train an architectural ablation variant from scratch under the same
        protocol. When given, the run never resumes from an existing checkpoint
        (always starts fresh) and checkpoints are written to ckpt_dir instead of
        the shared MODELS_DIR, so load_latest() (used everywhere else) can never
        accidentally pick up an ablated model.
    ckpt_dir: directory to save checkpoints to. Defaults to MODELS_DIR.
    """
    save_dir = ckpt_dir or MODELS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    t0 = _time.time()
    loader_gen = _seed_everything(seed)

    def _status(msg: str):
        if status_cb:
            status_cb(msg)

    # ── 1. Read data ────────────────────────────────────────────────────────
    _status("Step 1/5 — Reading database…")
    df = get_training_data()
    df = df[df["target_name"].notna() & df["smiles"].notna()].reset_index(drop=True)
    if len(df) < 10:
        return None, [], [], None, {}, [], "Need at least 10 valid samples."
    _status(f"Step 1/5 — {len(df):,} records  ({_time.time()-t0:.1f}s)")

    # ── 2. Stratified sample + scaffold split ───────────────────────────────
    if max_samples and len(df) > max_samples:
        _status(f"Step 2/5 — Sampling {max_samples:,} / {len(df):,}…")
        df = _fast_stratified_sample(df, max_samples, seed=split_seed)
    _status(f"Step 2/5 — Using {len(df):,} records")

    _split_label = {"scaffold": "scaffold", "cold_target": "cold-target", "double_cold": "double-cold"}
    _status(f"Step 2/5 — {_split_label.get(split_mode, split_mode)} split…")
    train_df, val_df, test_df, _dc_fell_back = _three_way_split(
        df, split_mode, val_frac=val_frac, test_frac=test_frac, seed=split_seed)
    if _dc_fell_back:
        _status("Step 2/5 — WARNING: double-cold fell back to cold-target "
                "(insufficient cold-scaffold × cold-target pairs in this dataset)")
    split_info = {
        "split_mode": split_mode, "split_seed": split_seed, "seed": seed,
        "val_frac": val_frac, "test_frac": test_frac,
        "double_cold_fell_back": bool(_dc_fell_back),
        "partitions": [_partition_summary("train", train_df),
                       _partition_summary("val",   val_df),
                       _partition_summary("test",  test_df)],
    }
    _status(f"Step 2/5 — {len(train_df):,} train / {len(val_df):,} val / "
            f"{len(test_df):,} test  ({_time.time()-t0:.1f}s)")

    # norm_stats from train only — val/test pic50 must not inform normalisation
    valid_p    = train_df["pic50"].dropna().astype(float)
    p_mean     = float(valid_p.mean()) if len(valid_p) > 0 else 7.0
    p_std      = max(float(valid_p.std()) if len(valid_p) > 1 else 1.5, 0.1)
    norm_stats = {"pic50_mean": p_mean, "pic50_std": p_std}

    # Build target → sequence mapping (full df, so all targets are covered)
    tgt_seq_map = (df.groupby("target_name")["protein_sequence"]
                     .first().fillna("").astype(str))
    tgt_to_key  = {tgt: _seq_key(seq) for tgt, seq in tgt_seq_map.items()}

    unique_smiles = df["smiles"].unique().tolist()
    unique_seqs   = df["protein_sequence"].dropna().unique().tolist()

    # ── 3. Cache checks ─────────────────────────────────────────────────────
    _status(f"Step 3/5 — Cache check: {len(unique_smiles):,} SMILES / {len(unique_seqs)} proteins…")
    precompute_fps(unique_smiles, status_cb=status_cb)
    embed_proteins(unique_seqs,  status_cb=status_cb)   # saves pooled + residues
    unload_models()
    _status(f"Step 3/5 — Caches ready  ({_time.time()-t0:.1f}s)")

    # ── 4. Load caches in parallel ───────────────────────────────────────────
    needed = set(str(s) for s in unique_smiles)
    _status(f"Step 4/5 — Loading {len(needed):,} FP + {len(unique_seqs)} protein embeddings…")
    fp_cache, prot_pool = _load_caches_parallel(needed, status_cb)

    # Load residue cache for cross-attention
    needed_keys   = set(tgt_to_key.values())
    residue_cache = load_residues_subset(needed_keys, status_cb=status_cb)
    _status(f"Step 4/5 — Caches loaded  ({_time.time()-t0:.1f}s)")

    # ── 5. Build datasets ────────────────────────────────────────────────────
    # Global protein index shared between train and val
    all_targets     = sorted(tgt_to_key.keys())
    all_keys        = [tgt_to_key[t] for t in all_targets]
    unique_prot_keys = list(dict.fromkeys(all_keys))
    prot_key_to_idx = {k: i for i, k in enumerate(unique_prot_keys)}

    # Save target protein embeddings (pooled) for inference — before del prot_pool
    target_prot_embs = np.stack([
        prot_pool.get(k, np.zeros(ESM_DIM, dtype=np.float32)).astype(np.float32)
        for k in all_keys
    ])  # (N_targets, 480)

    _status("Step 5/5 — Building datasets…")
    # Split each partition into positives (active=1) and real inactives (active=0)
    pos_train_df = train_df[train_df["active"] == 1].reset_index(drop=True)
    neg_train_df = train_df[train_df["active"] == 0].reset_index(drop=True)

    train_ds = TargetTrace3Dataset(pos_train_df, p_mean, p_std,
                                   fp_cache, prot_pool, prot_key_to_idx, tgt_to_key)
    val_ds   = TargetTrace3Dataset(val_df,       p_mean, p_std,
                                   fp_cache, prot_pool, prot_key_to_idx, tgt_to_key)
    test_ds  = TargetTrace3Dataset(test_df,      p_mean, p_std,
                                   fp_cache, prot_pool, prot_key_to_idx, tgt_to_key)
    neg_train_ds = TargetTrace3Dataset(neg_train_df, p_mean, p_std,
                                       fp_cache, prot_pool, prot_key_to_idx, tgt_to_key)
    del fp_cache, prot_pool
    gc.collect()

    if len(train_ds) < 10:
        return None, [], [], None, {}, [], "Not enough valid samples after feature extraction."

    _status(f"Step 5/5 — {len(train_ds):,} pos-train | {len(neg_train_ds):,} real-neg | "
            f"{len(val_ds):,} val | {len(test_ds):,} test ({len(unique_prot_keys)} proteins)  ({_time.time()-t0:.1f}s)")

    # Preload all protein residue tensors to GPU once — fast lookup during training
    max_L      = max((r.shape[0] for r in residue_cache.values()), default=1)
    N_prot     = len(unique_prot_keys)
    prot_res_t = torch.zeros(N_prot, max_L, ESM_DIM, device=device, dtype=torch.float32)
    prot_mask_t = torch.ones(N_prot, max_L, device=device, dtype=torch.bool)  # True = padding
    for k, i in prot_key_to_idx.items():
        r = residue_cache.get(k)
        if r is not None:
            L = r.shape[0]
            prot_res_t[i, :L]  = torch.tensor(r, dtype=torch.float32)
            prot_mask_t[i, :L] = False
    del residue_cache
    gc.collect()

    _status("Step 5/5 — Building data loaders…")

    loader_kw    = dict(collate_fn=_collate, num_workers=_N_WORKERS,
                        pin_memory=(device.type == "cuda"))
    # drop_last=True: prevents 1-sample final batch crashing BatchNorm
    train_loader    = DataLoader(train_ds,     batch_size=512, shuffle=True,
                                 drop_last=True, generator=loader_gen, **loader_kw)
    val_loader      = DataLoader(val_ds,       batch_size=512, shuffle=False, **loader_kw)
    test_loader     = DataLoader(test_ds,      batch_size=512, shuffle=False, **loader_kw)
    real_neg_loader = DataLoader(neg_train_ds, batch_size=256, shuffle=True,
                                 drop_last=True, generator=loader_gen, **loader_kw)
    prot_idx_to_target = {}
    for t in all_targets:
        prot_idx_to_target.setdefault(prot_key_to_idx[tgt_to_key[t]], t)

    _status(f"Step 5/5 — GPU training  (device={device}, AMP={'on' if USE_AMP else 'off'})…")
    # Cold splits (and any custom model_factory, e.g. an ablation variant) must
    # start from a fresh model — loading a warm-trained checkpoint would bake
    # prior knowledge (or an incompatible architecture) into the weights.
    if model_factory is not None:
        model = model_factory().to(device)
    elif split_mode != "scaffold" or not resume:
        model = TargetTrace3().to(device)
    else:
        model = (_load_ckpt() or TargetTrace3()).to(device)
    reg_crit = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs, eta_min=1e-6)
    scaler    = GradScaler(enabled=USE_AMP, init_scale=256)

    patience     = 5
    no_improve   = 0
    best_val_auc = -1.0
    best_ckpt    = None
    best_state   = None
    best_val_preds = None
    last_loss = last_mae = 0.0
    epoch_log: list[dict] = []
    epochs_run = 0

    real_neg_iter = itertools.cycle(real_neg_loader)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            fp, esm_p, prot_idx, pic50s = [b.to(device, non_blocking=True) for b in batch]
            B = len(fp)
            optimizer.zero_grad(set_to_none=True)

            # Protein-shuffle negatives: in-batch permutation → protein-space indices
            rand_perm  = torch.randperm(B, device=device)
            same       = rand_perm == torch.arange(B, device=device)
            rand_perm[same] = (rand_perm[same] + 1) % B
            neg_prot_idx = prot_idx[rand_perm]

            pos_res   = prot_res_t[prot_idx]
            pos_mask  = prot_mask_t[prot_idx]
            shuf_res  = prot_res_t[neg_prot_idx]
            shuf_mask = prot_mask_t[neg_prot_idx]

            # Real molecule-level negatives: (inactive_mol, its_correct_protein) → 0
            rn_fp, rn_esm, rn_prot_idx, rn_p50 = [
                b.to(device, non_blocking=True) for b in next(real_neg_iter)]
            Bn = len(rn_fp)
            rn_res  = prot_res_t[rn_prot_idx]
            rn_mask = prot_mask_t[rn_prot_idx]

            # The shuffle negative swaps BOTH protein views (residues and pooled
            # ESM-2): models that read only the pooled vector (no cross-attention
            # ablation) would otherwise see identical inputs with opposite labels.
            fp_all   = torch.cat([fp,       fp,        rn_fp  ])
            esm_all  = torch.cat([esm_p,    esm_p[rand_perm], rn_esm ])
            res_all  = torch.cat([pos_res,  shuf_res,  rn_res ])
            mask_all = torch.cat([pos_mask, shuf_mask, rn_mask])
            act_lbl  = torch.cat([torch.ones(B,   device=device),
                                   torch.zeros(B,  device=device),
                                   torch.zeros(Bn, device=device)])
            p50_all  = torch.cat([pic50s,
                                   torch.full((B,),  float("nan"), device=device),
                                   rn_p50])

            with autocast(device_type="cuda", enabled=USE_AMP):
                act_logit, pic50_pred = model(fp_all, esm_all, res_all, mask_all)
                act_loss   = F.binary_cross_entropy_with_logits(act_logit, act_lbl)
                valid      = ~torch.isnan(p50_all)
                pic50_loss = (reg_crit(pic50_pred[valid], p50_all[valid])
                              if valid.any() else torch.tensor(0.0, device=device))
                loss = act_loss + 0.3 * pic50_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        scheduler.step()
        last_loss = total_loss / max(1, len(train_loader))

        # Validation — real actives vs real inactives + protein-shuffle decoys
        vm, val_preds = _evaluate(model, val_loader, val_ds, prot_res_t, prot_mask_t,
                                  p_mean, p_std, seed=split_seed,
                                  prot_idx_to_target=prot_idx_to_target)
        val_auc  = vm["AUC"]
        last_mae = vm["pIC50 MAE"]

        epochs_run = epoch + 1
        entry = {
            "Epoch":      epochs_run,
            "Loss":       last_loss,
            "AUC":        val_auc,
            "AUC_real":   vm["AUC_real"],
            "AUC_decoy":  vm["AUC_decoy"],
            "AUPRC":      vm["AUPRC"],
            "EF1%":       vm["EF1%"],
            "EF5%":       vm["EF5%"],
            "BEDROC":     vm["BEDROC"],
            "pIC50 MAE":  vm["pIC50 MAE"],
            "pIC50 RMSE": vm["pIC50 RMSE"],
            "Spearman":   vm["Spearman"],
            "Pearson":    vm["Pearson"],
            "_roc":       vm["_roc"],
            "_pr":        vm["_pr"],
            "_scatter":   vm["_scatter"],
        }
        epoch_log.append(entry)
        if progress_cb:
            progress_cb(entry, epochs)

        if val_auc > best_val_auc:
            best_val_auc   = val_auc
            no_improve     = 0
            best_state     = copy.deepcopy(model.state_dict())
            best_val_preds = val_preds
            best_epoch     = epochs_run
        else:
            no_improve += 1
            if no_improve >= patience:
                _status(f"Early stopping at epoch {epochs_run} "
                        f"(no AUC improvement for {patience} epochs)")
                break

    # ── Final, single evaluation of the test partition with the best checkpoint
    test_metrics: dict = {}
    if best_state is not None:
        model.load_state_dict(best_state)
        tm, test_preds = _evaluate(model, test_loader, test_ds, prot_res_t, prot_mask_t,
                                   p_mean, p_std, seed=split_seed + 1,
                                   prot_idx_to_target=prot_idx_to_target)
        test_metrics = {k: v for k, v in tm.items() if not k.startswith("_")}
        _status(f"Test (scored once) — AUC {tm['AUC']:.3f} | AUC_real {tm['AUC_real']:.3f} | "
                f"AUC_decoy {tm['AUC_decoy']:.3f} | AUPRC {tm['AUPRC']:.3f}")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        best_ckpt = save_dir / f"targettrace3_{ts}.pkl"
        split_info["best_epoch"]   = best_epoch
        split_info["best_val_auc"] = best_val_auc
        split_info["test_metrics"] = test_metrics
        with open(best_ckpt, "wb") as f:
            pickle.dump({
                "model":            model,
                "target_names":     all_targets,
                "target_keys":      all_keys,
                "target_prot_embs": target_prot_embs,
                "norm_stats":       norm_stats,
                "split_mode":       split_mode,
                "split_info":       split_info,
            }, f)
        best_val_preds.to_csv(best_ckpt.with_name(best_ckpt.stem + "_val_preds.csv"), index=False)
        test_preds.to_csv(best_ckpt.with_name(best_ckpt.stem + "_test_preds.csv"), index=False)
        with open(best_ckpt.with_name(best_ckpt.stem + "_summary.json"), "w") as f:
            json.dump(split_info, f, indent=2)

    total_samples = len(train_ds) + len(val_ds) + len(test_ds)
    mark_all_trained()
    if best_ckpt:
        log_run(total_samples, len(all_targets), epochs_run, last_loss,
                best_val_auc, last_mae, str(best_ckpt))

    elapsed = _time.time() - t0
    best = next((e for e in reversed(epoch_log) if e["AUC"] == best_val_auc), epoch_log[-1] if epoch_log else {})
    return (model, all_targets, all_keys, target_prot_embs, norm_stats, epoch_log,
            f"Training complete — {epochs_run} epochs, {total_samples:,} samples, "
            f"{len(all_targets)} targets, {elapsed/60:.1f} min total.\n"
            f"Val AUC {best_val_auc:.3f}  |  AUPRC {best.get('AUPRC',0):.3f}  |  "
            f"EF@1% {best.get('EF1%',0):.1f}×  |  pIC50 MAE {best.get('pIC50 MAE',0):.3f}\n"
            f"Test AUC {test_metrics.get('AUC', float('nan')):.3f}  |  "
            f"AUC_real {test_metrics.get('AUC_real', float('nan')):.3f}  |  "
            f"AUPRC {test_metrics.get('AUPRC', float('nan')):.3f}")
