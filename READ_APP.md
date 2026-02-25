# Reading Trainer App (`read_app.py`)

## Purpose
`read_app.py` is a short-session reading trainer for improving reading speed while protecting comprehension.

It includes two techniques:

1. Technique A: phrase chunking (default ~10 minutes).
2. Technique B: moving underline pacer (default ~5 minutes).

## Run
```bash
streamlit run read_app.py
```

## Optional Gemini Usage
The app can work fully offline with local fallback passages and quizzes.

If you enable Gemini in the sidebar:
- Provide an API key in-session.
- It can generate custom passages and comprehension quizzes.

Dependency notes:
- `read_app.py` uses `google-generativeai`.
- `app.py` uses `google-genai`.
- Both are included in `requirements.txt`.

## Technique A: Chunking
- Adjustable visual help level (`0` to `4`) to progressively reduce guidance.
- Adjustable chunk size (`2` to `5` words).
- Difficulty presets that affect typography and line spacing.
- Ends with a comprehension quiz to verify understanding.

## Technique B: Moving Underline Pacer
- Adjustable pacer speed (`WPM`).
- Adjustable short session duration.
- JavaScript-based moving highlight to pull eyes forward steadily.
- Ends with a comprehension quiz.

## Progress Logging
Sessions are logged to:

`data/reading_trainer_log.csv`

Logged fields include:
- timestamp
- technique
- duration
- help level / target WPM
- word count
- quiz score and percentage
- notes

Use the `Progress` tab to:
- Filter by technique.
- Review recent sessions.
- Track comprehension trend over time.
- Export CSV.

## Practical Use Pattern
1. Start with comfortable speeds and stronger visual support.
2. Keep quiz performance stable before increasing speed or reducing help.
3. Increase WPM gradually (small increments) only when comprehension stays consistent.
4. Prefer frequent short sessions over occasional long sessions.
