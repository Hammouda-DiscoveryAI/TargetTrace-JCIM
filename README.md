# TargetTrace — runnable app

## What's here

- `app.py` — the Streamlit app itself (entry point)
- `database.py`, `embedders.py`, `features.py`, `model.py`, `trainer.py` —
  the full dependency closure `app.py` actually imports (traced directly,
  including the ChEMBL-import page's lazy import)
- `chembl_client.py` — used by the app's "ChEMBL Import" page
- `bootstrap.py` — one-time database seeding, referenced by the app's own
  "no trained model found" onboarding message
- `requirements.txt` / `environment.yml` — dependencies (pip and conda paths)
- `runnin_the_app.txt` — full install-and-run instructions, start to finish

## Quick start

See `runnin_the_app.txt` for the complete walkthrough. Short version:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
python bootstrap.py --chunks-dir protein_data_chunks --limit 200   # one-time
streamlit run app.py
```

`bootstrap.py` expects a `protein_data_chunks/` directory of ChEMBL CSV
chunks to seed from — that's data, not code, so it isn't included here (see
the parent `GitHub/` folder's data-acquisition scripts, e.g.
`build_chembl_dataset_v3.py`, if you need to build that data from scratch
instead).

