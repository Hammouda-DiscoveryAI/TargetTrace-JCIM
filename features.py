import gc
import json
import multiprocessing as mp
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, QED, rdMolDescriptors

warnings.filterwarnings("ignore", category=DeprecationWarning, module="rdkit")

FP_DIM = 4 * 2048 + 167 + 20   # = 8,379

_CACHE       = Path(__file__).parent / "embed_cache" / "mol_fp.pkl"
_CACHE_KEYS  = Path(__file__).parent / "embed_cache" / "mol_fp.keys.pkl"
_FAST_MATRIX = Path(__file__).parent / "embed_cache" / "mol_fp_matrix.npy"
_FAST_IDX    = Path(__file__).parent / "embed_cache" / "mol_fp_index.pkl"
_FP_DIM_FILE = Path(__file__).parent / "embed_cache" / "fp_dim.txt"
_CACHE.parent.mkdir(exist_ok=True)

_N_WORKERS = max(1, min(os.cpu_count() or 1, 8))


# ── Fingerprint computation ────────────────────────────────────────────────

def mol_fp(smiles: str) -> np.ndarray | None:
    """Compute 8379-dim fingerprint vector for one SMILES. No caching."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    ecfp2 = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 1, 2048), dtype=np.float32)
    ecfp4 = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048), dtype=np.float32)
    ecfp6 = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 3, 2048), dtype=np.float32)
    fcfp4 = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048, useFeatures=True),
                     dtype=np.float32)
    maccs = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)

    try:
        qed_val = QED.qed(mol)
    except Exception:
        qed_val = 0.0

    desc = np.array([
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NOCount(mol),
        Descriptors.MolWt(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.MolMR(mol),
        # New descriptors
        qed_val,
        Descriptors.NumSaturatedRings(mol),
        Descriptors.NumAliphaticRings(mol),
        Descriptors.RingCount(mol),
        Descriptors.NumAromaticHeterocycles(mol),
        Descriptors.NumSaturatedHeterocycles(mol),
        Descriptors.HeavyAtomCount(mol),
        Descriptors.NumValenceElectrons(mol),
        rdMolDescriptors.CalcNumAtomStereoCenters(mol),
        Descriptors.NumAliphaticCarbocycles(mol),
    ], dtype=np.float32)

    fp = np.concatenate([ecfp2, ecfp4, ecfp6, fcfp4, maccs, desc])  # (8379,)
    return np.nan_to_num(fp, nan=0.0, posinf=0.0, neginf=0.0)


def _fp_worker(smi: str):
    return smi, mol_fp(smi)


# ── Cache helpers ──────────────────────────────────────────────────────────

def _fp_cache_valid() -> bool:
    """True if cached FP_DIM matches the current constant."""
    try:
        return int(_FP_DIM_FILE.read_text().strip()) == FP_DIM
    except Exception:
        return False


def _clear_fp_cache(status_cb=None) -> None:
    """Delete all FP cache files so precompute starts from scratch."""
    for p in [_CACHE, _CACHE_KEYS, _FAST_MATRIX, _FAST_IDX, _FP_DIM_FILE]:
        if p.exists():
            p.unlink()
    if status_cb:
        status_cb("FP cache cleared — recomputing with new feature set (one-time)…")


def _load_fp_cache() -> dict:
    if _CACHE.exists():
        with open(_CACHE, "rb") as f:
            return pickle.load(f)
    return {}


def _save_keys(cache: dict) -> None:
    with open(_CACHE_KEYS, "wb") as f:
        pickle.dump(set(cache.keys()), f)


def _write_fp_dim() -> None:
    _FP_DIM_FILE.write_text(str(FP_DIM))


# ── Fast numpy cache ───────────────────────────────────────────────────────

def _build_fast_cache(cache: dict, status_cb=None) -> None:
    # Fill a preallocated array in place and drain `cache` as we go, instead
    # of np.stack()-ing a second full-size copy — at large scale (hundreds of
    # thousands of molecules x FP_DIM) holding both the source dict and the
    # stacked matrix at once can exceed available RAM and get OOM-killed.
    n = len(cache)
    if status_cb:
        status_cb(f"Building fast FP cache ({n:,} entries, one-time)…")
    smiles_list = list(cache.keys())
    matrix = np.empty((n, FP_DIM), dtype=np.float32)
    for i, s in enumerate(smiles_list):
        matrix[i] = cache.pop(s)
    np.save(str(_FAST_MATRIX), matrix)
    del matrix
    with open(_FAST_IDX, "wb") as f:
        pickle.dump(smiles_list, f)
    if status_cb:
        status_cb(f"Fast FP cache built ({n:,} entries → {_FAST_MATRIX.stat().st_size/1e9:.2f} GB)")


def load_fp_subset(needed: set, status_cb=None) -> dict:
    """Load the FP vectors for the given SMILES set (fast numpy path if available)."""
    if _FAST_MATRIX.exists() and _FAST_IDX.exists():
        if status_cb:
            status_cb("Loading FP cache (numpy)…")
        with open(_FAST_IDX, "rb") as f:
            smiles_list = pickle.load(f)
        smi_to_row = {smi: i for i, smi in enumerate(smiles_list)}
        matrix = np.load(str(_FAST_MATRIX))
        out = {smi: matrix[smi_to_row[smi]] for smi in needed if smi in smi_to_row}
        del matrix
        gc.collect()
        return out

    if status_cb:
        status_cb("Loading FP cache (pickle)…")
    if not _CACHE.exists():
        return {}
    with open(_CACHE, "rb") as f:
        full = pickle.load(f)
    out = {k: full[k] for k in needed if k in full}
    del full
    gc.collect()
    return out


# ── Precompute ─────────────────────────────────────────────────────────────

def precompute_fps(smiles_list: list[str], status_cb=None) -> None:
    """
    Compute and cache fingerprints for all unique SMILES not yet cached.
    Automatically clears the cache if FP_DIM has changed.
    """
    # Invalidate cache if the feature set changed
    if not _fp_cache_valid():
        _clear_fp_cache(status_cb)

    unique = [s for s in dict.fromkeys(smiles_list) if s]

    if _CACHE_KEYS.exists():
        with open(_CACHE_KEYS, "rb") as f:
            cached_keys = pickle.load(f)
        todo = [s for s in unique if s not in cached_keys]
        if not todo:
            if not _FAST_MATRIX.exists():
                cache = _load_fp_cache()
                _build_fast_cache(cache, status_cb)
                del cache
                gc.collect()
            if not _fp_cache_valid():
                _write_fp_dim()
            return

    cache = _load_fp_cache()
    todo  = [s for s in unique if s not in cache]
    if not todo:
        _save_keys(cache)
        _write_fp_dim()
        if not _FAST_MATRIX.exists():
            _build_fast_cache(cache, status_cb)
        return

    n = len(todo)
    if status_cb:
        status_cb(f"Fingerprints: computing {n:,} new SMILES on {_N_WORKERS} cores…")

    SAVE_EVERY = 10_000
    ctx = mp.get_context("fork")
    with ctx.Pool(_N_WORKERS) as pool:
        for i, (smi, fp) in enumerate(pool.imap(_fp_worker, todo, chunksize=500)):
            cache[smi] = fp
            if (i + 1) % SAVE_EVERY == 0:
                with open(_CACHE, "wb") as f:
                    pickle.dump(cache, f)
                if status_cb:
                    status_cb(f"Fingerprints: {i+1:,}/{n:,} ({_N_WORKERS} cores)")

    with open(_CACHE, "wb") as f:
        pickle.dump(cache, f)
    _save_keys(cache)
    _write_fp_dim()
    _build_fast_cache(cache, status_cb)
    if status_cb:
        status_cb(f"Fingerprints: {n:,} new → {len(cache):,} total cached")
