import torch
import torch.nn as nn

from embedders import ESM_DIM
from features import FP_DIM

_MOL  = 256
_PROT = 256
_FUSE = 128


class _MolProtCrossAttn(nn.Module):
    """
    Molecule-queries-protein cross-attention.
    query  = mol embedding  (B, mol_dim)
    keys/values = ESM-2 residue hidden states  (B, L, res_dim)
    output = protein context vector  (B, out_dim)
    """
    def __init__(self, mol_dim: int = _MOL, res_dim: int = ESM_DIM,
                 n_heads: int = 4, out_dim: int = _PROT):
        super().__init__()
        assert out_dim % n_heads == 0
        self.n_heads = n_heads
        self.d_k     = out_dim // n_heads
        self.scale   = self.d_k ** -0.5

        self.q_proj   = nn.Linear(mol_dim, out_dim, bias=False)
        self.k_proj   = nn.Linear(res_dim, out_dim, bias=False)
        self.v_proj   = nn.Linear(res_dim, out_dim, bias=False)
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.norm     = nn.LayerNorm(out_dim)

    def forward(self, mol: torch.Tensor, residues: torch.Tensor,
                pad_mask: torch.Tensor | None = None,
                return_attn: bool = False):
        """
        mol      : (B, mol_dim)
        residues : (B, L, res_dim)
        pad_mask : (B, L)  True = padding (masked out)
        returns  : (B, out_dim)  or  ((B, out_dim), (B, H, L)) if return_attn
        """
        B, L, _ = residues.shape
        H, dk   = self.n_heads, self.d_k

        Q = self.q_proj(mol).view(B, H, dk)                           # (B, H, dk)
        K = self.k_proj(residues).view(B, L, H, dk).transpose(1, 2)  # (B, H, L, dk)
        V = self.v_proj(residues).view(B, L, H, dk).transpose(1, 2)  # (B, H, L, dk)

        attn = (Q.unsqueeze(2) @ K.transpose(-2, -1)) * self.scale   # (B, H, 1, L)
        if pad_mask is not None:
            attn = attn.masked_fill(pad_mask[:, None, None, :], float("-inf"))
        attn = torch.softmax(attn, dim=-1)                            # (B, H, 1, L)

        out = (attn @ V).squeeze(2).reshape(B, H * dk)               # (B, out_dim)
        out = self.norm(self.out_proj(out))
        if return_attn:
            return out, attn.squeeze(2)                               # (B, H, L)
        return out


class TargetTrace3(nn.Module):

    def __init__(self, use_cross_attn: bool = True, use_hadamard: bool = True,
                 ecfp4_only: bool = False):
        """
        use_cross_attn : if False, the model never builds/uses the residue-level
                          cross-attention layers — it always falls back to the
                          mean-pooled ESM-2 path (prot_enc), even when residue
                          tensors are supplied. Used to retrain the "no cross-
                          attention" ablation variant from scratch.
        use_hadamard   : if False, the fusion layer only receives cat[mol, prot]
                          (no explicit mol*prot interaction term). Used to retrain
                          the "no Hadamard" ablation variant from scratch.
        ecfp4_only     : if True, every fingerprint dimension outside the ECFP4
                          block ([2048:4096)) is zeroed before the encoder sees it,
                          for every forward pass (train and eval alike) — so the
                          network can never learn to use the other fingerprint
                          blocks. Used to retrain the "ECFP4-only" ablation variant
                          from scratch.
        """
        super().__init__()
        self.use_cross_attn = use_cross_attn
        self.use_hadamard   = use_hadamard
        self.ecfp4_only     = ecfp4_only

        # ── ① Fingerprint encoder (8379 → 1024 → 512 → 256) ───────────────
        # Two extra hidden layers to handle the larger sparse input
        self.fp_enc = nn.Sequential(
            nn.Linear(FP_DIM, 1024), nn.SiLU(), nn.BatchNorm1d(1024), nn.Dropout(0.4),
            nn.Linear(1024,    512), nn.SiLU(), nn.BatchNorm1d(512),  nn.Dropout(0.3),
            nn.Linear(512,    _MOL), nn.SiLU(),
        )

        # ── ② Protein encoder — mean-pooled-ESM path (fallback, or the only
        # path used when use_cross_attn=False) ─────────────────────────────
        self.prot_enc = nn.Sequential(
            nn.Linear(ESM_DIM, 512), nn.SiLU(), nn.BatchNorm1d(512), nn.Dropout(0.3),
            nn.Linear(512,    _PROT), nn.SiLU(),
        )

        # ── ② Cross-attention: two stacked layers, mol queries ESM-2 residues
        # Layer 1: mol embedding queries residues → protein context
        # Layer 2: protein context re-queries residues with updated representation
        # Residual connection preserves layer-1 signal.
        # Only instantiated when use_cross_attn=True, so the ablated variant
        # never has these weights to begin with (not just disabled at inference).
        if self.use_cross_attn:
            self.cross_attn   = _MolProtCrossAttn(mol_dim=_MOL,  res_dim=ESM_DIM,
                                                  n_heads=4, out_dim=_PROT)
            self.cross_attn_2 = _MolProtCrossAttn(mol_dim=_PROT, res_dim=ESM_DIM,
                                                  n_heads=4, out_dim=_PROT)

        # ── ③ Fusion, with or without the Hadamard interaction term ────────
        # Full: cat[mol(256), prot(256), mol*prot(256)] = 768
        # No-Hadamard: cat[mol(256), prot(256)] = 512
        fusion_in = _MOL + _PROT + (_MOL if self.use_hadamard else 0)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 256), nn.SiLU(),
            nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, _FUSE), nn.SiLU(),
        )

        # ── ④ Output heads ─────────────────────────────────────────────────
        self.act_head   = nn.Linear(_FUSE, 1)   # binding probability (raw logit)
        self.pic50_head = nn.Linear(_FUSE, 1)   # pIC50 (normalised)

    def forward(self, fp: torch.Tensor,
                esm_pooled: torch.Tensor | None = None,
                residues:   torch.Tensor | None = None,
                pad_mask:   torch.Tensor | None = None):
        """
        fp         : (B, FP_DIM)      — multi-radius ECFP + FCFP4 + MACCS + descriptors
        esm_pooled : (B, ESM_DIM)     — mean-pooled ESM-2; used only when residues=None
        residues   : (B, L, ESM_DIM)  — per-residue ESM-2 hidden states (optional)
        pad_mask   : (B, L)  bool     — True = padding position

        Returns:
          act_logit : (B,)  — raw logit for P(binding); apply sigmoid for probability
          pic50_pred: (B,)  — predicted pIC50 (normalised)
        """
        if self.ecfp4_only:
            fp = fp.clone()
            fp[:, :2048]   = 0.0
            fp[:, 4096:]   = 0.0

        mol = self.fp_enc(fp)                        # (B, 256)

        if self.use_cross_attn and residues is not None:
            prot_1 = self.cross_attn(mol,    residues, pad_mask)   # (B, 256)
            prot   = self.cross_attn_2(prot_1, residues, pad_mask) + prot_1  # residual
        elif esm_pooled is not None:
            prot = self.prot_enc(esm_pooled)                  # (B, 256)
        else:
            raise ValueError("Either esm_pooled or residues must be provided.")

        if self.use_hadamard:
            interaction = mol * prot                  # (B, 256) explicit pairwise product
            fused = self.fusion(torch.cat([mol, prot, interaction], dim=1))  # (B, 128)
        else:
            fused = self.fusion(torch.cat([mol, prot], dim=1))  # (B, 128)

        return self.act_head(fused).squeeze(1), self.pic50_head(fused).squeeze(1)
