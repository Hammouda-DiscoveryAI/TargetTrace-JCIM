import math
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).parent / "bioactivity.db"


def _conn():
    return sqlite3.connect(str(DB_PATH))


def init_db() -> None:
    with _conn() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS bioactivity (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            smiles           TEXT    NOT NULL,
            target_name      TEXT    NOT NULL,
            target_chembl_id TEXT,
            uniprot_accession TEXT,
            protein_sequence TEXT,
            standard_type    TEXT,
            standard_value   REAL,
            pic50            REAL,
            source           TEXT    DEFAULT 'manual',
            added_at         TEXT    DEFAULT CURRENT_TIMESTAMP,
            trained          INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS training_runs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT,
            num_samples INTEGER,
            num_targets INTEGER,
            epochs      INTEGER,
            train_loss  REAL,
            val_accuracy REAL,
            val_mae      REAL,
            model_path   TEXT
        );
        """)


def _pic50(value_nm: float | None) -> float | None:
    if value_nm is None or value_nm <= 0:
        return None
    return 9.0 - math.log10(float(value_nm))


def add_record(
    smiles: str,
    target_name: str,
    target_chembl_id: str = "",
    uniprot_accession: str = "",
    protein_sequence: str = "",
    standard_type: str = "IC50",
    standard_value: float | None = None,
    source: str = "manual",
) -> None:
    with _conn() as con:
        con.execute(
            """INSERT INTO bioactivity
               (smiles, target_name, target_chembl_id, uniprot_accession,
                protein_sequence, standard_type, standard_value, pic50, source)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                smiles, target_name, target_chembl_id, uniprot_accession,
                protein_sequence, standard_type, standard_value,
                _pic50(standard_value), source,
            ),
        )


def get_training_data() -> pd.DataFrame:
    with _conn() as con:
        # Main query: no protein_sequence column — it's ~1000 chars repeated
        # 5500+ times per unique target, adding gigabytes of redundant data.
        df = pd.read_sql_query(
            """SELECT smiles, target_name, pic50, standard_value
               FROM bioactivity
               WHERE smiles IS NOT NULL AND target_name IS NOT NULL""",
            con,
        )
        # Sequence lookup keyed by target_name, not uniprot_accession.
        # Same target_name always maps to the same protein — one row per target
        # (~425 rows total) is all we need.
        seq_df = pd.read_sql_query(
            """SELECT target_name, MAX(protein_sequence) AS protein_sequence
               FROM bioactivity
               WHERE target_name     IS NOT NULL
                 AND protein_sequence IS NOT NULL
                 AND protein_sequence != ''
               GROUP BY target_name""",
            con,
        )
    seq_map = dict(zip(seq_df["target_name"], seq_df["protein_sequence"]))
    df["protein_sequence"] = df["target_name"].map(seq_map).fillna("")
    # active=1 if Ki/IC50/Kd ≤ 10 µM; treat missing standard_value as active
    df["active"] = (pd.to_numeric(df["standard_value"], errors="coerce")
                    .le(10_000.0).fillna(True).astype(int))
    return df


def untrained_count() -> int:
    with _conn() as con:
        return con.execute("SELECT COUNT(*) FROM bioactivity WHERE trained=0").fetchone()[0]


def total_count() -> int:
    with _conn() as con:
        return con.execute("SELECT COUNT(*) FROM bioactivity").fetchone()[0]


def get_training_stats() -> dict:
    """Fast DB stats using SQL aggregates — no row-level reads. Use this in the UI."""
    with _conn() as con:
        row = con.execute(
            """SELECT COUNT(*) as n,
                      COUNT(DISTINCT target_name) as n_targets,
                      SUM(CASE WHEN pic50 IS NOT NULL THEN 1 ELSE 0 END) as n_pic50
               FROM bioactivity
               WHERE smiles IS NOT NULL AND target_name IS NOT NULL"""
        ).fetchone()
    return {"n": row[0] or 0, "n_targets": row[1] or 0, "n_pic50": row[2] or 0}


def mark_all_trained() -> None:
    with _conn() as con:
        con.execute("UPDATE bioactivity SET trained=1")


def log_run(num_samples, num_targets, epochs, train_loss, val_acc, val_mae, model_path) -> None:
    with _conn() as con:
        con.execute(
            """INSERT INTO training_runs
               (timestamp, num_samples, num_targets, epochs, train_loss, val_accuracy, val_mae, model_path)
               VALUES (?,?,?,?,?,?,?,?)""",
            (datetime.now().isoformat(), num_samples, num_targets, epochs,
             train_loss, val_acc, val_mae, model_path),
        )


def get_history() -> pd.DataFrame:
    with _conn() as con:
        return pd.read_sql_query(
            "SELECT * FROM training_runs ORDER BY timestamp DESC", con
        )


def get_target_distribution() -> pd.DataFrame:
    with _conn() as con:
        return pd.read_sql_query(
            """SELECT target_name, COUNT(*) as count
               FROM bioactivity GROUP BY target_name ORDER BY count DESC""",
            con,
        )
