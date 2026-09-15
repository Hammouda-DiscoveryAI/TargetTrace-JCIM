
import argparse
import glob
import math
import sqlite3
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from database import DB_PATH, init_db


def pic50(value_nm: float) -> float | None:
    try:
        v = float(value_nm)
        return 9.0 - math.log10(v) if v > 0 else None
    except (ValueError, TypeError):
        return None


def bootstrap(chunks_dir: Path, limit_per_target: int) -> None:
    init_db()
    files = sorted(chunks_dir.glob("*.csv"))
    if not files:
        print(f"No CSV files found in {chunks_dir}")
        return

    print(f"Found {len(files)} chunk files. Importing up to {limit_per_target} rows per target...")
    total_inserted = 0

    with sqlite3.connect(str(DB_PATH)) as con:
        for fpath in tqdm(files, unit="target"):
            try:
                df = pd.read_csv(fpath)
                df = df.dropna(subset=["smiles", "protein_name", "potency_value_nm"])
                df["potency_value_nm"] = pd.to_numeric(df["potency_value_nm"], errors="coerce")
                df = df.dropna(subset=["potency_value_nm"])
                if df.empty:
                    continue
                df = df.head(limit_per_target)
                for _, row in df.iterrows():
                    p50 = pic50(row["potency_value_nm"])
                    con.execute(
                        """INSERT OR IGNORE INTO bioactivity
                           (smiles, target_name, target_chembl_id, uniprot_accession,
                            protein_sequence, standard_type, standard_value, pic50, source)
                           VALUES (?,?,?,?,?,?,?,?,?)""",
                        (
                            str(row["smiles"]),
                            str(row["protein_name"]),
                            str(row.get("target_chembl_id", "")),
                            str(row.get("uniprot_accession", "")),
                            str(row.get("protein_sequence", "")),
                            str(row.get("property_type", "IC50")),
                            float(row["potency_value_nm"]),
                            p50,
                            "chembl_bootstrap",
                        ),
                    )
                    total_inserted += 1
                con.commit()
            except Exception as e:
                tqdm.write(f"[skip] {fpath.name}: {e}")

    print(f"\nDone. Inserted {total_inserted} records into {DB_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-dir", default="../protein_data_chunks",
                        help="Path to protein_data_chunks/ directory")
    parser.add_argument("--limit", type=int, default=200,
                        help="Max rows per target file (default 200)")
    args = parser.parse_args()

    chunks_path = Path(args.chunks_dir)
    if not chunks_path.exists():
        print(f"Error: {chunks_path} does not exist.")
        raise SystemExit(1)

    bootstrap(chunks_path, args.limit)
