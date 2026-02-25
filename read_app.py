from __future__ import annotations

import os
import re
import json
import time
import random
import datetime as dt
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import streamlit as st
from streamlit.components.v1 import html as st_html

# -----------------------------
# Optional LLM (Gemini)
# -----------------------------
try:
    import google.generativeai as genai  # pip install google-generativeai
except Exception:
    genai = None


# -----------------------------
# App constants / paths
# -----------------------------
APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

LOG_PATH = DATA_DIR / "reading_trainer_log.csv"

DEFAULT_MODEL = "gemini-1.5-flash"  # change if you prefer


# -----------------------------
# Local fallback text bank
# -----------------------------
LOCAL_TEXT_BANK = [
    (
        "The rain arrived quietly, as if it had been waiting behind the clouds for the city to look away. "
        "Streetlights blurred into golden halos, and the pavement shone like dark glass. "
        "A cyclist cut through the intersection, leaving a thin wake of water. "
        "Inside a small café, a barista wiped the counter twice, then stopped, listening to the weather "
        "with the focus of someone trying to remember an old song."
    ),
    (
        "On the third day of the hike, the path narrowed and the trees leaned closer together. "
        "A cold wind moved through the leaves, carrying the scent of pine and stone. "
        "At noon they reached a ridge where the valley opened below them, wide and quiet. "
        "Far away, a river caught the light and looked, for a moment, like a ribbon of metal."
    ),
    (
        "The library had the kind of silence that felt carefully constructed. "
        "Pages turned softly, chairs shifted, pens tapped once and then stopped. "
        "In the corner, a student traced the spine of a book with their thumb, "
        "as if choosing the next sentence mattered more than the next hour."
    ),
    (
        "He didn’t notice the change at first. The conversations were the same, the streets familiar, "
        "the morning routine intact. But the clocks seemed slightly impatient, "
        "and the pauses between thoughts felt shorter. By lunchtime he had the odd sense "
        "that the day was reading him back."
    ),
]


# -----------------------------
# Helpers
# -----------------------------
def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def ensure_log_exists() -> None:
    if not LOG_PATH.exists():
        df = pd.DataFrame(
            columns=[
                "timestamp",
                "technique",
                "duration_min",
                "help_level",
                "target_wpm",
                "words",
                "quiz_score",
                "quiz_total",
                "percent",
                "notes",
            ]
        )
        df.to_csv(LOG_PATH, index=False)


def load_log() -> pd.DataFrame:
    ensure_log_exists()
    try:
        return pd.read_csv(LOG_PATH)
    except Exception:
        return pd.DataFrame()


def append_log(row: Dict[str, Any]) -> None:
    ensure_log_exists()
    df = pd.read_csv(LOG_PATH)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(LOG_PATH, index=False)


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w']+\b", text))


def split_sentences(text: str) -> List[str]:
    # Simple sentence split (good enough for practice text)
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_sentence(sentence: str, target_chunk_words: int) -> List[str]:
    words = re.findall(r"\b[\w']+\b|[^\w\s]", sentence, re.UNICODE)
    # Build chunks by counting word tokens only, but keep punctuation attached
    chunks = []
    cur = []
    w = 0
    for tok in words:
        cur.append(tok)
        if re.match(r"\b[\w']+\b", tok):
            w += 1
        if w >= target_chunk_words:
            chunks.append("".join(_join_tokens(cur)))
            cur = []
            w = 0
    if cur:
        chunks.append("".join(_join_tokens(cur)))
    # Cleanup whitespace
    chunks = [re.sub(r"\s+", " ", c).strip() for c in chunks if c.strip()]
    return chunks


def _join_tokens(tokens: List[str]) -> List[str]:
    """
    Join tokens with spaces where needed.
    We return a list of strings we can ''.join() later.
    """
    out = []
    prev = ""
    for t in tokens:
        if not out:
            out.append(t)
        else:
            # no space before punctuation
            if re.match(r"[.,;:!?)]", t):
                out.append(t)
            elif prev in ["(", "“", '"', "'"]:
                out.append(t)
            else:
                out.append(" " + t)
        prev = t
    return out


def apply_help_to_chunks(chunks: List[str], help_level: int) -> str:
    """
    help_level (0-4):
      4 = strongest: chunk boundaries + bold + subtle separators
      3 = chunk boundaries + bold
      2 = chunk boundaries only
      1 = minimal: occasional boundary markers (every other chunk)
      0 = none: plain sentence
    """
    if help_level <= 0:
        return " ".join(chunks)

    styled = []
    for i, c in enumerate(chunks):
        mark = " | " if help_level >= 2 else " "
        show_marker = True
        if help_level == 1:
            show_marker = (i % 2 == 0)

        if help_level >= 3:
            c_html = f"<span style='font-weight:700'>{st.escape(c)}</span>"
        else:
            c_html = st.escape(c)

        if help_level >= 4:
            c_html = (
                f"<span style='padding:2px 6px; border-radius:8px; "
                f"background: rgba(0,0,0,0.05); display:inline-block; margin:2px 0'>{c_html}</span>"
            )

        styled.append(c_html)
        if show_marker and i != len(chunks) - 1:
            styled.append(f"<span style='opacity:0.35'>{st.escape(mark)}</span>")

    return "".join(styled)


def local_generate_text(kind: str, approx_words: int) -> str:
    # Kind unused for local; simple random stitching + trimming
    base = " ".join(random.sample(LOCAL_TEXT_BANK, k=min(2, len(LOCAL_TEXT_BANK))))
    # Expand if needed
    while word_count(base) < approx_words:
        base += " " + random.choice(LOCAL_TEXT_BANK)
    # Trim to approx words
    words = base.split()
    return " ".join(words[:approx_words])


def llm_generate_text(api_key: str, model: str, kind: str, approx_words: int) -> str:
    if genai is None:
        raise RuntimeError("google-generativeai is not installed, so LLM generation isn't available.")

    genai.configure(api_key=api_key)
    m = genai.GenerativeModel(model)

    # Keep it safe + consistent: fictional, original, neutral
    prompt = f"""
Write an original fictional reading passage for training reading technique "{kind}".
Constraints:
- Language: English
- Tone: engaging but not too complex
- No copyrighted text, no references to real books/characters/brands
- Approximate length: {approx_words} words
- Include a few concrete details (places, objects, actions) so we can test comprehension.
Return ONLY the passage, no headings.
""".strip()

    resp = m.generate_content(prompt)
    text = getattr(resp, "text", "") or ""
    return text.strip()


def local_make_quiz(text: str, n_q: int = 5) -> Dict[str, Any]:
    """
    Simple offline quiz: pick sentences; ask about a detail by masking a word.
    Not perfect, but works without an LLM.
    """
    sents = split_sentences(text)
    sents = [s for s in sents if word_count(s) >= 10]
    if len(sents) < 2:
        sents = split_sentences(" ".join(LOCAL_TEXT_BANK))

    questions = []
    for _ in range(n_q):
        s = random.choice(sents)
        tokens = re.findall(r"\b[\w']+\b", s)
        if len(tokens) < 8:
            continue
        answer = random.choice(tokens[3:-2])
        masked = re.sub(rf"\b{re.escape(answer)}\b", "____", s, count=1)

        # Build options
        distractors = set()
        all_words = re.findall(r"\b[\w']+\b", text)
        while len(distractors) < 3 and len(all_words) > 10:
            w = random.choice(all_words)
            if w.lower() != answer.lower() and len(w) >= 3:
                distractors.add(w)
        options = list(distractors) + [answer]
        random.shuffle(options)

        questions.append(
            {
                "q": f"Fill in the blank (from the passage):\n\n{masked}",
                "options": options,
                "answer": answer,
            }
        )

    if not questions:
        # fallback
        questions = [
            {
                "q": "Was the passage fictional?",
                "options": ["Yes", "No", "Not sure", "It was a poem"],
                "answer": "Yes",
            }
        ]

    return {"questions": questions}


def llm_make_quiz(api_key: str, model: str, text: str, n_q: int = 6) -> Dict[str, Any]:
    if genai is None:
        raise RuntimeError("google-generativeai is not installed, so LLM quiz generation isn't available.")

    genai.configure(api_key=api_key)
    m = genai.GenerativeModel(model)

    prompt = f"""
Create {n_q} multiple-choice comprehension questions about the passage below.
Requirements:
- Questions must be answerable ONLY from the passage.
- 4 options each (A-D).
- Exactly one correct answer.
- Focus on concrete details and causal relations (who/what/where/why), not trivia.
Return valid JSON with this schema:
{{
  "questions": [
    {{
      "q": "...",
      "options": ["...","...","...","..."],
      "answer_index": 0
    }}
  ]
}}
PASSAGE:
\"\"\"{text}\"\"\"
""".strip()

    resp = m.generate_content(prompt)
    raw = (getattr(resp, "text", "") or "").strip()

    # Try to parse JSON robustly
    try:
        # Sometimes models wrap JSON in ```json blocks
        raw2 = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE | re.MULTILINE).strip()
        obj = json.loads(raw2)
        # Basic validation
        qs = obj.get("questions", [])
        cleaned = []
        for q in qs:
            opts = q.get("options", [])
            ai = q.get("answer_index", None)
            if isinstance(opts, list) and len(opts) == 4 and isinstance(ai, int) and 0 <= ai < 4:
                cleaned.append({"q": q.get("q", ""), "options": opts, "answer_index": ai})
        if cleaned:
            return {"questions": cleaned}
    except Exception:
        pass

    # fallback to local if parsing fails
    return local_make_quiz(text, n_q=min(5, n_q))


def compute_passage_for_A(
    api_key: str,
    model: str,
    use_llm: bool,
    approx_words: int,
) -> str:
    if use_llm and api_key:
        return llm_generate_text(api_key, model, kind="Technique A (chunking phrases)", approx_words=approx_words)
    return local_generate_text("A", approx_words)


def compute_passage_for_B(
    api_key: str,
    model: str,
    use_llm: bool,
    approx_words: int,
) -> str:
    if use_llm and api_key:
        return llm_generate_text(api_key, model, kind="Technique B (moving underline)", approx_words=approx_words)
    return local_generate_text("B", approx_words)


def render_pacer(text: str, wpm: int) -> None:
    """
    Displays text with a word-by-word moving underline/highlight driven by JS.
    """
    words = re.findall(r"\b[\w']+\b|[^\w\s]", text, re.UNICODE)
    # Wrap each token in a span
    spans = []
    for i, tok in enumerate(words):
        safe = tok.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        # Put spaces before word tokens/punct appropriately
        if i == 0:
            prefix = ""
        else:
            # no space before punctuation
            if re.match(r"[.,;:!?)]", tok):
                prefix = ""
            elif words[i - 1] in ["(", "“", '"', "'"]:
                prefix = ""
            else:
                prefix = " "
        spans.append(f"{prefix}<span class='tok' data-i='{i}'>{safe}</span>")

    ms_per_word = int(60000 / max(60, wpm))  # clamp minimum
    total_ms = ms_per_word * max(1, len(words))

    html_doc = f"""
<!DOCTYPE html>
<html>
<head>
<style>
  .wrap {{
    font-family: ui-serif, Georgia, "Times New Roman", Times, serif;
    font-size: 22px;
    line-height: 1.55;
    padding: 12px;
    border-radius: 14px;
    background: rgba(0,0,0,0.03);
  }}
  .tok {{
    position: relative;
    padding-bottom: 2px;
    border-bottom: 0px solid transparent;
    transition: background 80ms linear, border-bottom 80ms linear;
    border-radius: 6px;
  }}
  .active {{
    background: rgba(0,0,0,0.10);
    border-bottom: 3px solid rgba(0,0,0,0.35);
  }}
  .meta {{
    font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
    font-size: 13px;
    opacity: 0.75;
    margin: 8px 0 0 2px;
  }}
</style>
</head>
<body>
  <div class="wrap" id="wrap">{''.join(spans)}</div>
  <div class="meta">Pacer: {wpm} wpm · approx duration: {total_ms/1000:.0f}s</div>

<script>
  const toks = Array.from(document.querySelectorAll('.tok'));
  let i = 0;
  function step() {{
    toks.forEach(t => t.classList.remove('active'));
    if (i < toks.length) {{
      toks[i].classList.add('active');
      // keep the active token in view
      toks[i].scrollIntoView({{block: 'center', inline: 'nearest'}});
      i += 1;
      setTimeout(step, {ms_per_word});
    }}
  }}
  // start after a short pause
  setTimeout(step, 350);
</script>
</body>
</html>
""".strip()

    st_html(html_doc, height=420, scrolling=True)


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Reading Trainer (Chunking + Pacer)", layout="wide")

st.title("Reading Trainer: Chunking (A) + Moving Underline (B)")
st.caption("Practice in short, deliberate sessions. Comprehension comes first.")

with st.sidebar:
    st.subheader("Text generation (optional)")
    use_llm = st.toggle("Use LLM (Gemini) for passages/quizzes", value=False)
    api_key = st.text_input("Gemini API key", type="password", help="Stored only in memory for this session.")
    model = st.text_input("Model name", value=DEFAULT_MODEL)
    st.divider()
    st.subheader("Logging")
    st.write(f"Log file: `{LOG_PATH}`")

log_df = load_log()

tabs = st.tabs(["Technique A — Chunking phrases (10 min)", "Technique B — Moving underline (5 min)", "Progress"])


# -----------------------------
# Technique A
# -----------------------------
with tabs[0]:
    st.header("Technique A — Chunking / phrase capture")
    st.write(
        "Goal: move from hearing every word to *seeing phrases*. "
        "You’ll get guided chunk boundaries that you can fade out over time."
    )

    col1, col2, col3 = st.columns([1.0, 1.0, 1.2])

    with col1:
        help_level = st.slider("Visual help level", 0, 4, 3, help="Higher = stronger chunk cues.")
        chunk_words = st.slider("Chunk size (words)", 2, 5, 3)
    with col2:
        duration_min = st.slider("Session duration (minutes)", 5, 15, 10)
        difficulty = st.selectbox("Text difficulty", ["Easy", "Medium", "Hard"], index=1)
    with col3:
        st.markdown(
            "**How to use:**\n"
            "- Read smoothly in phrase units.\n"
            "- Don’t force speed; stay accurate.\n"
            "- After reading, take the quiz.\n"
        )

    approx_wpm = 200
    approx_words = int(approx_wpm * duration_min)

    if st.button("Generate 10-minute exercises", type="primary"):
        passage = compute_passage_for_A(api_key, model, use_llm, approx_words)
        st.session_state["A_passage"] = passage
        st.session_state["A_start_ts"] = time.time()
        st.session_state["A_quiz_done"] = False

        # Create quiz now (so it matches the passage)
        if use_llm and api_key:
            quiz = llm_make_quiz(api_key, model, passage, n_q=6)
        else:
            quiz = local_make_quiz(passage, n_q=6)
        st.session_state["A_quiz"] = quiz

    passage = st.session_state.get("A_passage", "")
    if passage:
        st.subheader("Reading exercises")
        sents = split_sentences(passage)

        # Render each sentence as chunked HTML
        blocks = []
        for s in sents:
            chunks = chunk_sentence(s, target_chunk_words=chunk_words)
            blocks.append(apply_help_to_chunks(chunks, help_level=help_level))

        # Slightly vary presentation depending on difficulty
        font_size = {"Easy": 24, "Medium": 22, "Hard": 20}[difficulty]
        line_height = {"Easy": 1.7, "Medium": 1.6, "Hard": 1.55}[difficulty]

        st.markdown(
            f"""
            <div style="font-family: ui-serif, Georgia, 'Times New Roman', Times, serif;
                        font-size: {font_size}px; line-height: {line_height};
                        padding: 14px 16px; border-radius: 16px;
                        background: rgba(0,0,0,0.03);">
              {"<br><br>".join(blocks)}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.info(
            "Tip: If you notice strong inner narration, slightly soften it by focusing on *meaning per phrase* "
            "instead of sound per word."
        )

        st.subheader("Comprehension check")
        quiz = st.session_state.get("A_quiz", None)
        if quiz and not st.session_state.get("A_quiz_done", False):
            qs = quiz.get("questions", [])
            answers = []
            for idx, q in enumerate(qs):
                st.markdown(f"**Q{idx+1}.** {q['q']}")
                choice = st.radio(
                    label=f"q{idx}",
                    options=q["options"],
                    index=None,
                    horizontal=False,
                    key=f"A_q_{idx}",
                )
                answers.append(choice)

            if st.button("Submit quiz (A)"):
                score = 0
                total = len(qs)
                for idx, q in enumerate(qs):
                    if "answer_index" in q:
                        correct = q["options"][q["answer_index"]]
                    else:
                        correct = q.get("answer", "")
                    if answers[idx] == correct:
                        score += 1

                percent = (score / max(1, total)) * 100.0
                st.session_state["A_quiz_done"] = True
                st.success(f"Score: {score}/{total} ({percent:.0f}%)")

                started = st.session_state.get("A_start_ts", None)
                elapsed_min = None
                if started:
                    elapsed_min = max(1, int((time.time() - started) / 60))

                append_log(
                    {
                        "timestamp": now_iso(),
                        "technique": "A_chunking",
                        "duration_min": elapsed_min or duration_min,
                        "help_level": help_level,
                        "target_wpm": "",
                        "words": word_count(passage),
                        "quiz_score": score,
                        "quiz_total": total,
                        "percent": round(percent, 1),
                        "notes": f"chunk_words={chunk_words}, difficulty={difficulty}, llm={bool(use_llm and api_key)}",
                    }
                )
                st.toast("Logged session ✔")

        elif st.session_state.get("A_quiz_done", False):
            st.write("Quiz completed and logged. Generate a new session when you’re ready.")


# -----------------------------
# Technique B
# -----------------------------
with tabs[1]:
    st.header("Technique B — Moving underline / visual anchor")
    st.write(
        "Goal: reduce word-by-word narration by letting a steady pace pull your eyes forward. "
        "You control the WPM."
    )

    col1, col2 = st.columns([1.1, 1.0])
    with col1:
        wpm = st.slider("Pacer speed (WPM)", 160, 320, 230, step=5)
        duration_min_b = st.slider("Session duration (minutes)", 3, 8, 5)
    with col2:
        st.markdown(
            "**How to use:**\n"
            "- Keep your eyes with the underline/highlight.\n"
            "- If you lose meaning, drop WPM 10–15%.\n"
            "- After reading, take the quiz.\n"
        )

    approx_words_b = int(wpm * duration_min_b)

    if st.button("Generate 5-minute pacer text", type="primary"):
        passage_b = compute_passage_for_B(api_key, model, use_llm, approx_words_b)
        st.session_state["B_passage"] = passage_b
        st.session_state["B_start_ts"] = time.time()
        st.session_state["B_quiz_done"] = False

        if use_llm and api_key:
            quiz_b = llm_make_quiz(api_key, model, passage_b, n_q=6)
        else:
            quiz_b = local_make_quiz(passage_b, n_q=6)
        st.session_state["B_quiz"] = quiz_b

    passage_b = st.session_state.get("B_passage", "")
    if passage_b:
        st.subheader("Pacer")
        render_pacer(passage_b, wpm=wpm)

        st.subheader("Comprehension check")
        quiz_b = st.session_state.get("B_quiz", None)
        if quiz_b and not st.session_state.get("B_quiz_done", False):
            qs = quiz_b.get("questions", [])
            answers = []
            for idx, q in enumerate(qs):
                st.markdown(f"**Q{idx+1}.** {q['q']}")
                choice = st.radio(
                    label=f"bq{idx}",
                    options=q["options"],
                    index=None,
                    horizontal=False,
                    key=f"B_q_{idx}",
                )
                answers.append(choice)

            if st.button("Submit quiz (B)"):
                score = 0
                total = len(qs)
                for idx, q in enumerate(qs):
                    if "answer_index" in q:
                        correct = q["options"][q["answer_index"]]
                    else:
                        correct = q.get("answer", "")
                    if answers[idx] == correct:
                        score += 1

                percent = (score / max(1, total)) * 100.0
                st.session_state["B_quiz_done"] = True
                st.success(f"Score: {score}/{total} ({percent:.0f}%)")

                started = st.session_state.get("B_start_ts", None)
                elapsed_min = None
                if started:
                    elapsed_min = max(1, int((time.time() - started) / 60))

                append_log(
                    {
                        "timestamp": now_iso(),
                        "technique": "B_pacer",
                        "duration_min": elapsed_min or duration_min_b,
                        "help_level": "",
                        "target_wpm": wpm,
                        "words": word_count(passage_b),
                        "quiz_score": score,
                        "quiz_total": total,
                        "percent": round(percent, 1),
                        "notes": f"llm={bool(use_llm and api_key)}",
                    }
                )
                st.toast("Logged session ✔")
        elif st.session_state.get("B_quiz_done", False):
            st.write("Quiz completed and logged. Generate a new session when you’re ready.")


# -----------------------------
# Progress tab
# -----------------------------
with tabs[2]:
    st.header("Progress")
    df = load_log()
    if df.empty:
        st.info("No sessions logged yet. Do a session in Technique A or B first.")
    else:
        # Parse timestamp
        try:
            df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
        except Exception:
            df["timestamp_dt"] = pd.NaT

        tech = st.multiselect(
            "Filter techniques",
            options=sorted(df["technique"].dropna().unique().tolist()),
            default=sorted(df["technique"].dropna().unique().tolist()),
        )
        df2 = df[df["technique"].isin(tech)].copy()

        st.subheader("Recent sessions")
        st.dataframe(
            df2.sort_values("timestamp_dt", ascending=False).head(25),
            use_container_width=True,
        )

        st.subheader("Comprehension score over time")
        # Make a chart-friendly df
        cdf = df2.dropna(subset=["percent"]).copy()
        if not cdf.empty:
            cdf = cdf.sort_values("timestamp_dt")
            chart_df = cdf[["timestamp_dt", "percent", "technique"]].copy()
            chart_df = chart_df.set_index("timestamp_dt")
            st.line_chart(chart_df, y="percent", color="technique")
        else:
            st.info("No valid percent scores to plot yet.")

        st.subheader("Optional: export log")
        st.download_button(
            "Download log CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="reading_trainer_log.csv",
            mime="text/csv",
        )


# Footer note
st.caption("Note: This is training, not speed-reading. If comprehension dips, slow down.")