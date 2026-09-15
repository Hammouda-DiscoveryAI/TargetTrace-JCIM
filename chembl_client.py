"""ChEMBL API helpers — wraps chembl_webresource_client."""
import time

import pandas as pd
import requests
from chembl_webresource_client.new_client import new_client

_activity = new_client.activity
_target   = new_client.target
_molecule = new_client.molecule


# ---------------------------------------------------------------------------
# Target search
# ---------------------------------------------------------------------------

def search_targets(query: str, max_results: int = 20) -> list[dict]:
    results = _target.search(query).filter(target_type="SINGLE PROTEIN").only(
        ["target_chembl_id", "pref_name", "organism", "target_components"]
    )
    return list(results[:max_results])


def get_uniprot_accession(target_obj: dict) -> str | None:
    for comp in target_obj.get("target_components", []):
        acc = comp.get("accession")
        if acc:
            return acc
        for syn in comp.get("component_synonyms", []):
            if syn.get("syn_type") == "UNIPROT":
                return syn.get("component_synonym")
    return None


def fetch_sequence_from_uniprot(accession: str) -> str:
    try:
        resp = requests.get(
            f"https://rest.uniprot.org/uniprotkb/{accession}.fasta", timeout=15
        )
        if resp.status_code != 200:
            return ""
        lines = resp.text.strip().split("\n")
        return "".join(lines[1:])
    except Exception:
        return ""


def get_target_info(target_chembl_id: str) -> tuple[str, str, str]:
    """Returns (target_name, uniprot_accession, protein_sequence)."""
    obj = _target.get(target_chembl_id)
    if not obj:
        return "", "", ""
    name = obj.get("pref_name", "")
    acc  = get_uniprot_accession(obj) or ""
    seq  = fetch_sequence_from_uniprot(acc) if acc else ""
    return name, acc, seq


# ---------------------------------------------------------------------------
# Bioactivity fetch
# ---------------------------------------------------------------------------

def fetch_bioactivity(
    target_chembl_id: str,
    activity_types: list[str] | None = None,
    limit: int = 500,
) -> pd.DataFrame:
    types = activity_types or ["IC50", "Kd", "Ki"]
    records = []
    for atype in types:
        try:
            results = list(
                _activity.filter(
                    target_chembl_id=target_chembl_id,
                    standard_type=atype,
                    standard_units="nM",
                    standard_relation="=",
                )[:limit]
            )
            for r in results:
                smiles = r.get("canonical_smiles")
                value  = r.get("standard_value")
                if not smiles or not value:
                    continue
                try:
                    records.append({
                        "smiles":           smiles,
                        "target_name":      r.get("target_pref_name", ""),
                        "target_chembl_id": target_chembl_id,
                        "binder_id":        r.get("molecule_chembl_id", ""),
                        "standard_type":    atype,
                        "standard_value":   float(value),
                    })
                except (ValueError, TypeError):
                    continue
        except Exception:
            continue
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Compound search
# ---------------------------------------------------------------------------

def search_compounds(query: str, max_results: int = 10) -> list[dict]:
    results = _molecule.search(query).only(
        ["molecule_chembl_id", "pref_name", "molecule_structures"]
    )
    out = []
    for r in list(results[:max_results]):
        structs = r.get("molecule_structures") or {}
        out.append({
            "chembl_id": r.get("molecule_chembl_id", ""),
            "name":      r.get("pref_name", ""),
            "smiles":    structs.get("canonical_smiles", ""),
        })
    return out
