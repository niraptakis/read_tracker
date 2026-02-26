# read_tracker

This repo now contains two Streamlit apps:

1. `app.py`: long-term reading planner and tracker.
2. `read_app.py`: reading-speed training app focused on faster reading with comprehension retention.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
# Long-term planning / tracking app
streamlit run app.py

# Reading-speed training app
streamlit run read_app.py
```

## Docs
- Planner app docs: this README.
- Reading trainer docs: see `READ_APP.md`.

## Planner App (`app.py`) Summary
- `Book list`: manage streams/books, status, pages, notes, and Gemini plan settings.
- `Weekly engine`: follow planned sessions and log actual reads/pages/minutes.
- `Summary`: track progress and export/import snapshots.
