
import hashlib
import pickle
from pathlib import Path

import numpy as np
import torch

CACHE_DIR = Path(__file__).parent / "embed_cache"
CACHE_DIR.mkdir(exist_ok=True)

_PROT_POOLED          = CACHE_DIR / "prot_pooled.pkl"
_PROT_POOLED_KEYS     = CACHE_DIR / "prot_pooled.keys.pkl"
_PROT_RESIDUES        = CACHE_DIR / "prot_residues.pkl"
_PROT_RESIDUES_KEYS   = CACHE_DIR / "prot_residues.keys.pkl"

ESM_DIM = 480


# ── Cache helpers ──────────────────────────────────────────────────────────

def _load(path: Path) -> dict:
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}


def _save(path: Path, data: dict) -> None:
    with open(path, "wb") as f:
        pickle.dump(data, f)


def _seq_key(seq: str) -> str:
    return hashlib.md5(seq.encode()).hexdigest()


def _save_prot_keys(pooled_c: dict) -> None:
    with open(_PROT_POOLED_KEYS, "wb") as f:
        pickle.dump(set(pooled_c.keys()), f)


def _save_residue_keys(residue_c: dict) -> None:
    with open(_PROT_RESIDUES_KEYS, "wb") as f:
        pickle.dump(set(residue_c.keys()), f)


# ── ESM-2 ──────────────────────────────────────────────────────────────────

_esm_tok = _esm_mdl = _esm_dev = None
_MAX_TOK = 514   # 512 residues + CLS + EOS


def _load_esm() -> None:
    global _esm_tok, _esm_mdl, _esm_dev
    if _esm_mdl is not None:
        return
    from transformers import EsmModel, EsmTokenizer
    _esm_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _esm_tok = EsmTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
    _esm_mdl = EsmModel.from_pretrained("facebook/esm2_t12_35M_UR50D").to(_esm_dev)
    _esm_mdl.eval()


def embed_proteins(sequences: list[str], batch_size: int = 32,
                   status_cb=None) -> None:
    """
    Compute and cache ESM-2 pooled + residue embeddings for all unique sequences.
    Residue keys are the authoritative cache check — if all residues are cached,
    pooled are guaranteed cached too.
    """
    unique_seqs = [s for s in set(sequences) if s]

    # Fast path: residue keys file is authoritative
    if _PROT_RESIDUES_KEYS.exists():
        with open(_PROT_RESIDUES_KEYS, "rb") as f:
            res_keys = pickle.load(f)
        if all(_seq_key(s) in res_keys for s in unique_seqs):
            return

    pooled_c   = _load(_PROT_POOLED)
    residues_c = _load(_PROT_RESIDUES)
    todo = [s for s in unique_seqs if _seq_key(s) not in residues_c]

    if not todo:
        _save_prot_keys(pooled_c)
        _save_residue_keys(residues_c)
        return

    _load_esm()
    n = len(todo)
    for start in range(0, n, batch_size):
        batch = todo[start:start + batch_size]
        if status_cb:
            status_cb(f"ESM-2: {min(start + batch_size, n)}/{n} proteins")
        inputs = _esm_tok(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=_MAX_TOK,
        ).to(_esm_dev)
        with torch.no_grad():
            out = _esm_mdl(**inputs)
        hidden    = out.last_hidden_state[:, 1:-1, :]   # strip CLS + EOS: (B, L, 480)
        attn_mask = inputs["attention_mask"][:, 1:-1]   # (B, L)

        for j, seq in enumerate(batch):
            real = hidden[j][attn_mask[j].bool()].cpu().numpy()   # (L_real, 480)
            key  = _seq_key(seq)
            pooled_c[key]   = real.mean(0).astype(np.float32)     # (480,)
            residues_c[key] = real.astype(np.float32)              # (L_real, 480)

    _save(_PROT_POOLED,      pooled_c)
    _save(_PROT_RESIDUES,    residues_c)
    _save_prot_keys(pooled_c)
    _save_residue_keys(residues_c)


def load_residues_subset(needed: set, status_cb=None) -> dict:
    """Load {seq_key: np.ndarray(L, 480)} for the given keys.
    Handles legacy cache format where values were stored as (ndarray, int) tuples.
    """
    if not _PROT_RESIDUES.exists():
        return {}
    if status_cb:
        status_cb("Loading protein residue cache…")
    with open(_PROT_RESIDUES, "rb") as f:
        full = pickle.load(f)
    out = {}
    for k in needed:
        if k not in full:
            continue
        v = full[k]
        # Legacy format: (ndarray, int) — extract just the array
        if isinstance(v, tuple):
            v = v[0]
        out[k] = np.asarray(v, dtype=np.float32)
    return out


def get_protein_embedding(sequence: str) -> np.ndarray:
    """Returns pooled ESM-2 (480,). Computes on-the-fly if not cached."""
    pooled_c = _load(_PROT_POOLED)
    key = _seq_key(sequence)
    if key not in pooled_c:
        embed_proteins([sequence])
        pooled_c = _load(_PROT_POOLED)
    return pooled_c.get(key, np.zeros(ESM_DIM, dtype=np.float32)).astype(np.float32)


def unload_models() -> None:
    """Release ESM-2 from GPU/CPU memory after pre-computation."""
    global _esm_tok, _esm_mdl, _esm_dev
    _esm_tok = _esm_mdl = _esm_dev = None
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
