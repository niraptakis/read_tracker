# read_tracker

## What is this?
A local Streamlit web app for generating, tracking, and evolving long-term reading plans with optional Gemini-assisted planning.

### How to run:
```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install depndencies
pip install -r requirements.txt

# 3. Run Streamlit App
streamlit run app.py
```

## Tabs
- `Book list`: Add and manage streams/books, statuses, page counts, and Gemini planning settings.
- `Weekly engine`: Follow the weekly plan (`Planned`) and log what you actually read (`Actual`), pages, and minutes.
- `Summary`: View progress, weekly totals, quick stats, and export/import or save CSV snapshots.

## How it works
- You define reading streams and books in `Book list`.
- If you use Gemini, the app generates a weekly structure and fills `Planned` sessions.
- In `Weekly engine`, you log real behavior in `Actual` plus `Start Pg`, `End Pg`, and `Minutes`.
- The app calculates pages per session and weekly totals from your logs.
- `Summary` aggregates progress by actual book read, so metrics reflect reality, not just the plan.

## Long-term usage (recommended)
- Start with a 12-month plan (or 52 weeks) to create direction.
- Treat `Planned` as guidance, not a rigid contract.
- After each reading session, log `Actual` + pages + minutes immediately.
- Review `Summary` weekly to spot consistency gaps and overload weeks.
- Every 4-8 weeks, regenerate or adjust the plan based on what you actually finished.
- Use `Save to disk (manual)` snapshots before major changes so you keep historical versions.
- Keep streams (`Status = Stream`) separate from concrete books to plan cleanly and track accurately.
