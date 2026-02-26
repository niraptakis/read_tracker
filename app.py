from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Any

import pandas as pd
import streamlit as st
import json
import re
import os

# Optional LLM support
try:
    from google import genai
except Exception:
    genai = None


# -----------------------------
# Defaults / schema
# -----------------------------
BOOK_STATUSES = ["Stream", "Wishlist", "Active", "Paused", "Finished"]

SESSION_COLS = [
    # 5 sessions (S5 optional)
    ("S1", True),
    ("S2", True),
    ("S3", True),
    ("S4", True),
    ("S5", False),
]

def default_book_list() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Book": "Dan Brown (rotate)", "Status": "Stream", "Total Pages": 0, "Notes": ""},
            {"Book": "Game of Thrones (current)", "Status": "Stream", "Total Pages": 0, "Notes": ""},
            {"Book": "Nonfiction (modular)", "Status": "Stream", "Total Pages": 0, "Notes": ""},
        ]
    )

def default_weekly_engine(weeks: int = 52) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for w in range(1, weeks + 1):
        row: Dict[str, Any] = {"Week": w, "Phase": ""}
        for label, _required in SESSION_COLS:
            # Single book choice per session
            row[f"{label} Book"] = ""
            row[f"{label} Start Pg"] = None
            row[f"{label} End Pg"] = None
            row[f"{label} Minutes"] = None
        rows.append(row)
    return pd.DataFrame(rows)

def normalize_weekly_schema(df_weekly: pd.DataFrame) -> pd.DataFrame:
    """Backfill missing session columns and migrate legacy planned/actual data to '* Book'."""
    df = df_weekly.copy()
    if "Week" not in df.columns:
        df["Week"] = range(1, len(df) + 1)
    if "Phase" not in df.columns:
        df["Phase"] = ""

    for label, _required in SESSION_COLS:
        legacy_book_col = f"{label} Book"
        planned_col = f"{label} Planned"
        actual_col = f"{label} Actual"

        if legacy_book_col not in df.columns:
            if actual_col in df.columns:
                df[legacy_book_col] = df[actual_col]
            elif planned_col in df.columns:
                df[legacy_book_col] = df[planned_col]
            else:
                df[legacy_book_col] = ""

        # If both existed, prefer Actual when present, otherwise Planned.
        if actual_col in df.columns or planned_col in df.columns:
            cur = df[legacy_book_col].fillna("").astype(str).str.strip()
            if actual_col in df.columns:
                act = df[actual_col].fillna("").astype(str).str.strip()
                cur = act.where(act != "", cur)
            if planned_col in df.columns:
                pln = df[planned_col].fillna("").astype(str).str.strip()
                cur = pln.where((cur == "") & (pln != ""), cur)
            df[legacy_book_col] = cur

        for col_name in [f"{label} Start Pg", f"{label} End Pg", f"{label} Minutes"]:
            if col_name not in df.columns:
                df[col_name] = None

    keep_cols = ["Week", "Phase"]
    for label, _required in SESSION_COLS:
        keep_cols.extend([f"{label} Book", f"{label} Start Pg", f"{label} End Pg", f"{label} Minutes"])
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    return df

def compute_pages_read(df_weekly: pd.DataFrame) -> pd.DataFrame:
    df = df_weekly.copy()

    for label, _req in SESSION_COLS:
        start = pd.to_numeric(df[f"{label} Start Pg"], errors="coerce")
        end = pd.to_numeric(df[f"{label} End Pg"], errors="coerce")
        pages = (end - start + 1).clip(lower=0)
        pages = pages.where(start.notna() & end.notna(), other=pd.NA)
        df[f"{label} Pages Read"] = pages

    page_cols = [f"{label} Pages Read" for label, _ in SESSION_COLS]
    min_cols = [f"{label} Minutes" for label, _ in SESSION_COLS]

    pages_df = df[page_cols].apply(pd.to_numeric, errors="coerce")
    mins_df = df[min_cols].apply(pd.to_numeric, errors="coerce")

    df["Week Pages Read"] = pages_df.sum(axis=1, skipna=True)
    df["Week Minutes"] = mins_df.sum(axis=1, skipna=True)

    return df

def book_progress(df_books: pd.DataFrame, df_weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Sum Pages Read by matching book name strings in Weekly Engine session columns.
    """
    dfw = compute_pages_read(df_weekly)

    # Build long-form log: (Book, PagesRead)
    logs = []
    for label, _ in SESSION_COLS:
        b = dfw[f"{label} Book"].fillna("").astype(str)
        p = pd.to_numeric(dfw[f"{label} Pages Read"], errors="coerce").fillna(0)
        logs.append(pd.DataFrame({"Book": b, "PagesRead": p}))
    long = pd.concat(logs, ignore_index=True)
    long = long[long["Book"].str.strip() != ""]

    by_book = long.groupby("Book", as_index=False)["PagesRead"].sum()

    out = df_books.copy()
    out["Total Pages"] = pd.to_numeric(out["Total Pages"], errors="coerce").fillna(0).astype(int)
    out = out.merge(by_book, on="Book", how="left")
    out["PagesRead"] = out["PagesRead"].fillna(0)
    out["% Progress"] = out.apply(
        lambda r: (r["PagesRead"] / r["Total Pages"]) if r["Total Pages"] > 0 else pd.NA, axis=1
    )
    return out[["Book", "Status", "Total Pages", "PagesRead", "% Progress", "Notes"]].sort_values(
        by=["Status", "Book"]
    )

def get_book_options(df_books: pd.DataFrame) -> List[str]:
    opts = df_books["Book"].dropna().astype(str).tolist()
    # Keep unique, preserve order
    seen = set()
    out = []
    for x in opts:
        x = x.strip()
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def upsert_session_row(
    df_weekly: pd.DataFrame,
    week: int,
    session_label: str,
    book: str,
    start_pg: int | None,
    end_pg: int | None,
    minutes: int | None,
    phase: str | None = None,
) -> pd.DataFrame:
    """Update one session cell-set for a given week and return normalized df."""
    df = normalize_weekly_schema(df_weekly.copy())
    if week not in df["Week"].tolist():
        return df
    ix = df.index[df["Week"] == week][0]
    df.at[ix, f"{session_label} Book"] = book or ""
    df.at[ix, f"{session_label} Start Pg"] = start_pg
    df.at[ix, f"{session_label} End Pg"] = end_pg
    df.at[ix, f"{session_label} Minutes"] = minutes
    if phase is not None:
        df.at[ix, "Phase"] = phase
    return normalize_weekly_schema(df)


# -----------------------------
# Gemini helpers (optional)
# -----------------------------
def generate_plan_with_gemini(
    api_key: str,
    model: str,
    weeks: int,
    sessions_per_week: int,
    min_mins: int,
    max_mins: int,
    streams: List[str],
    preferences: str,
) -> pd.DataFrame:
    """
    Returns a weekly_engine-like DataFrame with Week/Phase and session Book/Minutes filled.
    Does NOT fill page ranges (user logs those).
    """
    if genai is None:
        raise RuntimeError("google-genai not installed. pip install google-genai")

    client = genai.Client(api_key=api_key)

    # Ask for JSON to avoid brittle CSV parsing.
    prompt = f"""
Create a {weeks}-week reading plan.

Constraints:
- Sessions per week: {sessions_per_week} (use S1..S{sessions_per_week}, and optionally S5 as a buffer if sessions_per_week==4)
- Minutes per session: {min_mins} to {max_mins}
- User won't read every day; plan should be resilient. Prefer 4 sessions/week with an optional buffer session.
- Nonfiction is modular: allow switching; don't require finishing.

Allowed book streams (use only these names exactly):
{streams}
These should populate the 'Sx Book' fields.

Preferences:
{preferences}

Output STRICT JSON array, length {weeks}, where each element has:
{{
  "Week": <int>,
  "Phase": "<short label>",
  "Sessions": [
    {{"slot":"S1","Book":"<one of allowed names>","Minutes":<int>}},
    ...
  ],
  "Optional": {{"slot":"S5","Book":"<one of allowed names or empty>","Minutes":<int>}}  // optional can be null
}}
"""

    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "temperature": 0.4,
        },
    )

    text = (resp.text or "").strip()

    def extract_json(s: str) -> str:
        fence = re.search(r"```(?:json)?\s*(.*?)\s*```", s, flags=re.DOTALL | re.IGNORECASE)
        if fence:
            return fence.group(1).strip()

        start_candidates = [i for i in [s.find("["), s.find("{")] if i != -1]
        if not start_candidates:
            return s
        start = min(start_candidates)

        end_candidates = [s.rfind("]"), s.rfind("}")]
        end = max(end_candidates)
        if end == -1:
            return s

        return s[start:end + 1].strip()

    json_str = extract_json(text)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        raise ValueError(
            "Gemini did not return valid JSON. "
            "Try again or tighten the prompt.\n"
            f"First 300 chars:\n{json_str[:300]}"
        )

    # Build weekly engine skeleton
    df = default_weekly_engine(weeks=weeks)

    for item in data:
        w = int(item["Week"])
        df.loc[df["Week"] == w, "Phase"] = item.get("Phase", "")

        sessions = item.get("Sessions", [])
        for s in sessions:
            slot = s.get("slot", "")
            if slot not in [f"S{i}" for i in range(1, 6)]:
                continue
            df.loc[df["Week"] == w, f"{slot} Book"] = s.get("Book", "")
            df.loc[df["Week"] == w, f"{slot} Minutes"] = s.get("Minutes", None)

        opt = item.get("Optional", None)
        if isinstance(opt, dict):
            df.loc[df["Week"] == w, "S5 Book"] = opt.get("Book", "")
            df.loc[df["Week"] == w, "S5 Minutes"] = opt.get("Minutes", None)

    return df


# -----------------------------
# Streamlit app
# -----------------------------
st.set_page_config(page_title="Reading Planner", layout="wide")

st.title("Reading Planner (local)")

# Session state init
if "books" not in st.session_state:
    st.session_state.books = default_book_list()
if "weekly" not in st.session_state:
    st.session_state.weekly = default_weekly_engine(weeks=52)
if "_weekly_schema_normalized" not in st.session_state:
    st.session_state.weekly = normalize_weekly_schema(st.session_state.weekly)
    st.session_state._weekly_schema_normalized = True

tabs = st.tabs(["Book list", "Weekly engine", "Summary"])

# ---- Tab 1: Book list
with tabs[0]:
    st.subheader("Book list (wishlist + active)")
    st.caption("Add any book/stream here. Weekly engine book dropdown uses this list.")

    df_books = st.session_state.books.copy()

    edited = st.data_editor(
        df_books,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Status": st.column_config.SelectboxColumn("Status", options=BOOK_STATUSES),
            "Total Pages": st.column_config.NumberColumn("Total Pages", min_value=0, step=1),
        },
    )

    st.session_state.books = edited

    st.divider()
    st.subheader("Optional: Gemini setup (only when generating/editing plan)")
    st.caption("You can ignore this after initial generation. Key is kept only in this session unless you choose otherwise.")

    col1, col2, col3 = st.columns([2, 2, 3])
    with col1:
        api_key = st.text_input("Gemini API key", type="password")
    with col2:
        model = st.text_input("Model", value="gemini-2.5-flash")
    with col3:
        prefs = st.text_area(
            "Plan preferences (used only when generating)",
            value="Design a 12-month phased reading plan. " \
            "Months 1-3: Establish rhythm, finish current GOT book, light thriller. " \
            "Months 4-6: Increase thriller intensity, maintain nonfiction depth. " \
            "Months 7-9: Heavier nonfiction focus, slightly reduce fiction load. " \
            "Months 10-12: Peak narrative momentum (major thriller or GOT push). " \
            "Rules: Avoid repeating identical weekly structures for more than 4 consecutive weeks. " \
            "Gradually adjust session minutes over the year. " \
            "Ensure cognitive balance (no heavy nonfiction immediately after another heavy session). " \
            "Return ONLY valid JSON. No markdown. No commentary. No code fences. " \
            "Round session minutes to the nearest 10 minutes.",
            height=80,
        )

    gen_cols = st.columns([1, 1, 1, 3])
    with gen_cols[0]:
        weeks = st.number_input("Weeks", min_value=4, max_value=104, value=52, step=1)
    with gen_cols[1]:
        sessions_per_week = st.number_input("Sessions/week", min_value=1, max_value=5, value=4, step=1)
    with gen_cols[2]:
        min_mins = st.number_input("Min mins", min_value=5, max_value=120, value=15, step=5)
        max_mins = st.number_input("Max mins", min_value=10, max_value=180, value=60, step=5)
    st.caption("Minutes are per session. Pages are logged by you in Weekly Engine.")

    if st.button("Generate / regenerate Weekly Engine with Gemini"):
        if not api_key:
            st.error("Enter your Gemini API key (or skip this feature).")
        else:
            stream_rows = st.session_state.books[st.session_state.books["Status"] == "Stream"]
            streams = get_book_options(stream_rows)
            if not streams:
                streams = get_book_options(st.session_state.books)
            try:
                df_new = generate_plan_with_gemini(
                    api_key=api_key,
                    model=model,
                    weeks=int(weeks),
                    sessions_per_week=int(sessions_per_week),
                    min_mins=int(min_mins),
                    max_mins=int(max_mins),
                    streams=streams,
                    preferences=prefs,
                )
                st.session_state.weekly = df_new
                st.success("Weekly Engine generated. Go to the Weekly engine tab to edit/log.")
            except Exception as e:
                st.exception(e)

# ---- Tab 2: Weekly engine
with tabs[1]:
    st.subheader("Weekly engine (log sessions)")
    st.caption("Log sessions with controls below. Table is view-only.")

    df_books = st.session_state.books
    book_options = [""] + get_book_options(df_books)
    session_options = [s for s, _ in SESSION_COLS]

    df_weekly = normalize_weekly_schema(st.session_state.weekly.copy())

    with st.form("weekly_log_form", clear_on_submit=False):
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            selected_week = st.selectbox(
                "Week",
                options=df_weekly["Week"].astype(int).tolist(),
                index=0,
            )
            selected_session = st.selectbox("Session", options=session_options, index=0)
        with c2:
            start_pg = st.number_input("Start Pg", min_value=0, step=1, value=0)
            end_pg = st.number_input("End Pg", min_value=0, step=1, value=0)
            minutes = st.number_input("Minutes", min_value=0, step=5, value=20)
        with c3:
            selected_book = st.selectbox("Book", options=book_options, index=0)
            current_phase = df_weekly.loc[df_weekly["Week"] == selected_week, "Phase"].astype(str).iloc[0]
            phase = st.text_input("Phase (optional)", value=current_phase)
        submit_log = st.form_submit_button("Save session log")

    if submit_log:
        if not selected_book.strip():
            st.error("Choose a book before saving.")
        elif end_pg > 0 and start_pg > 0 and end_pg < start_pg:
            st.error("End Pg must be greater than or equal to Start Pg.")
        else:
            start_pg_val = int(start_pg) if start_pg > 0 else None
            end_pg_val = int(end_pg) if end_pg > 0 else None
            minutes_val = int(minutes) if minutes > 0 else None
            updated = upsert_session_row(
                st.session_state.weekly,
                week=int(selected_week),
                session_label=selected_session,
                book=selected_book,
                start_pg=start_pg_val,
                end_pg=end_pg_val,
                minutes=minutes_val,
                phase=phase,
            )
            st.session_state.weekly = updated
            st.success(f"Saved {selected_session} for Week {selected_week}.")

    st.markdown("### Weekly engine (view only)")
    view_df = compute_pages_read(normalize_weekly_schema(st.session_state.weekly.copy()))
    st.dataframe(view_df, use_container_width=True, height=600)

# ---- Tab 3: Summary
with tabs[2]:
    st.subheader("Summary")

    df_prog = book_progress(st.session_state.books, st.session_state.weekly)
    df_prog_view = df_prog.copy()
    df_prog_view["% Progress"] = pd.to_numeric(df_prog_view["% Progress"], errors="coerce") * 100.0
    st.markdown("### Book progress")
    st.dataframe(
        df_prog_view,
        use_container_width=True,
        column_config={
            "% Progress": st.column_config.ProgressColumn(
                "% Progress", format="%.0f%%", min_value=0.0, max_value=100.0
            )
        },
    )

    st.markdown("### Weekly totals")
    dfw = compute_pages_read(st.session_state.weekly)
    weekly_totals = dfw[["Week", "Phase", "Week Pages Read", "Week Minutes"]].copy()
    st.dataframe(weekly_totals, use_container_width=True, height=360)

    # Basic stats
    st.markdown("### Quick stats")
    total_mins = pd.to_numeric(dfw["Week Minutes"], errors="coerce").fillna(0).sum()
    total_pages = pd.to_numeric(dfw["Week Pages Read"], errors="coerce").fillna(0).sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total minutes logged", int(total_mins))
    c2.metric("Total pages logged", int(total_pages))
    c3.metric("Weeks with any reading", int((pd.to_numeric(dfw["Week Minutes"], errors="coerce").fillna(0) > 0).sum()))

    st.divider()
    st.subheader("Export / Import")
    st.caption("Export CSVs for backup; you can extend timeline by generating more weeks via Gemini or manual copy/paste.")

    # Export buttons
    csv_books = st.session_state.books.to_csv(index=False).encode("utf-8")
    csv_weekly = st.session_state.weekly.to_csv(index=False).encode("utf-8")
    st.download_button("Download Book list CSV", csv_books, file_name="book_list.csv", mime="text/csv")
    st.download_button("Download Weekly engine CSV", csv_weekly, file_name="weekly_engine.csv", mime="text/csv")

    st.divider()
    st.subheader("Save to disk (manual)")

    save_dir = "saves"
    os.makedirs(save_dir, exist_ok=True)

    colA, colB = st.columns([1, 3])
    with colA:
        save_clicked = st.button("Save now")

    with colB:
        save_name = st.text_input("Save label (optional)", value="")

    if save_clicked:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        label = "".join(ch for ch in save_name.strip() if ch.isalnum() or ch in ("-", "_"))
        label_part = f"_{label}" if label else ""

        books_path = os.path.join(save_dir, f"book_list{label_part}_{ts}.csv")
        weekly_path = os.path.join(save_dir, f"weekly_engine{label_part}_{ts}.csv")

        st.session_state.books.to_csv(books_path, index=False)
        st.session_state.weekly.to_csv(weekly_path, index=False)

        st.success(f"Saved:\n- {books_path}\n- {weekly_path}")

    # Import buttons
    st.divider()
    st.subheader("Import existing data")

    uploaded_books = st.file_uploader("Upload Book list CSV", type="csv", key="books_upload")
    uploaded_weekly = st.file_uploader("Upload Weekly engine CSV", type="csv", key="weekly_upload")

    if uploaded_books is not None:
        df_books_new = pd.read_csv(uploaded_books)
        st.session_state.books = df_books_new
        st.success("Book list loaded.")

    if uploaded_weekly is not None:
        df_weekly_new = pd.read_csv(uploaded_weekly)
        df_weekly_new = normalize_weekly_schema(df_weekly_new)
        st.session_state.weekly = df_weekly_new
        st.success("Weekly engine loaded.")
