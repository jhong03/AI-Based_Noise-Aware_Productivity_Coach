# build_dataset_from_sqlite.py
import os, sqlite3, json, random
from datetime import datetime, date
from collections import Counter, defaultdict

DB_PATH = os.path.join(os.path.expanduser("~"), "Documents", "NoiseLogs", "noise_focus.db")
OUT_DIR = os.path.join(os.getcwd(), "datasets")
OUT_JSONL = os.path.join(OUT_DIR, "coach_dataset.jsonl")
OUT_JSONL_VAL = os.path.join(OUT_DIR, "coach_dataset_val.jsonl")

random.seed(42)

def pct(n, d): return round(100.0 * n / d, 1) if d else 0.0

def fetch_dates(conn):
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT DATE(timestamp) FROM NoiseLog ORDER BY DATE(timestamp) ASC")
    return [r[0] for r in cur.fetchall()]

def summarize_for_date(conn, d):
    cur = conn.cursor()

    # Noise basics
    cur.execute("SELECT COUNT(*), AVG(db_level) FROM NoiseLog WHERE DATE(timestamp)=?", (d,))
    total_logs, avg_db = cur.fetchone()
    total_logs = total_logs or 0
    avg_db = round(avg_db or 0.0, 1)

    cur.execute("""
      SELECT noise_category, COUNT(*)
      FROM NoiseLog
      WHERE DATE(timestamp)=?
      GROUP BY noise_category
    """, (d,))
    cat = dict(cur.fetchall())
    quiet = cat.get("Quiet", 0)
    moderate = cat.get("Moderate", 0)
    noisy = cat.get("Noisy", 0)

    cur.execute("""
      SELECT label, COUNT(*) AS cnt
      FROM NoiseLog
      WHERE DATE(timestamp)=?
      GROUP BY label
      ORDER BY cnt DESC
      LIMIT 1
    """, (d,))
    row = cur.fetchone()
    top_label = row[0] if row else "Unknown"

    # Pomodoro sessions
    cur.execute("""
      SELECT COUNT(*),
             SUM(strftime('%s', COALESCE(end_time, start_time)) - strftime('%s', start_time)) / 60.0
      FROM PomodoroSession
      WHERE DATE(start_time)=? AND status='Completed'
    """, (d,))
    session_count, focus_minutes = cur.fetchone()
    session_count = session_count or 0
    focus_minutes = round(focus_minutes or 0.0)

    # Optional mean session focus_score if available
    cur.execute("""
      SELECT AVG(pr.focus_score)
      FROM PomodoroReport pr
      JOIN PomodoroSession ps ON ps.session_id = pr.session_id
      WHERE DATE(ps.start_time)=?
    """, (d,))
    focus_score = cur.fetchone()[0]
    focus_score = round(float(focus_score), 1) if focus_score is not None else None

    return {
        "date": d,
        "total_logs": total_logs,
        "avg_db": avg_db,
        "quiet_pct": pct(quiet, total_logs),
        "moderate_pct": pct(moderate, total_logs),
        "noisy_pct": pct(noisy, total_logs),
        "top_label": top_label,
        "session_count": session_count,
        "focus_minutes": focus_minutes,
        "focus_score": focus_score,
    }

def make_instruction(s):
    return (
        "Analyze the following daily noise & focus summary and write a concise motivational note:\n"
        f"Date: {s['date']}\n"
        f"Average dB: {s['avg_db']}\n"
        f"Noise mix: Quiet {s['quiet_pct']}% | Moderate {s['moderate_pct']}% | Noisy {s['noisy_pct']}%\n"
        f"Most common sound: {s['top_label']}\n"
        f"Completed Pomodoros: {s['session_count']} (≈ {s['focus_minutes']} minutes)\n"
        + (f"Mean session focus score: {s['focus_score']}\n" if s['focus_score'] is not None else "")
        + "End with one actionable tip."
    )

def synth_response(s):
    # Simple rule-based “coach” text (4–6 sentences) + actionable tip.
    avg = s["avg_db"]
    q, m, n = s["quiet_pct"], s["moderate_pct"], s["noisy_pct"]
    sessions = s["session_count"]
    minutes = s["focus_minutes"]

    mood = []
    if sessions >= 4 or minutes >= 100:
        mood.append("Great stamina today — you put in solid focused time.")
    elif sessions >= 2:
        mood.append("Nice consistency — you got meaningful focus blocks in.")
    else:
        mood.append("Light day for focus — that happens, no stress.")

    if avg < 45:
        noise_line = "The environment skewed quiet, ideal for deep work."
    elif avg < 65:
        noise_line = "Moderate noise dominated, which can be productive for routine tasks."
    else:
        noise_line = "Noise levels trended high, which likely added friction to concentration."

    mix_line = f"Quiet/Moderate/Noisy split was {q}/{m}/{n}%, suggesting your best windows may be in quieter periods."

    if s["top_label"] not in ("Unknown", ""):
        label_line = f"Most frequent sound was '{s['top_label']}', so plan around or mask that pattern."
    else:
        label_line = "There wasn’t a single dominant sound pattern today."

    # Actionable tip variants
    tips = [
        "Block your next deep-work session in the quietest two-hour window you can find tomorrow.",
        "Use a brown-noise track and lower system notifications during your next Pomodoro.",
        "Batch routine tasks into moderate-noise hours; save deep work for quiet windows.",
        "Try noise-isolating headphones and a 5-minute setup ritual before each Pomodoro.",
        "Do a quick environment reset (seat, lighting, hydration) before the first session."
    ]
    tip = random.choice(tips)

    lines = [
        mood[0],
        noise_line,
        mix_line,
        label_line,
        f"You completed {sessions} Pomodoro(s) totaling ~{minutes} minutes.",
        f"Actionable tip: {tip}"
    ]
    # Guarantee 4–6 sentences
    return " ".join(lines[:5]) + " " + lines[5]

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"DB not found at {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    dates = fetch_dates(conn)
    items = []

    for d in dates:
        s = summarize_for_date(conn, d)
        if s["total_logs"] == 0:
            continue
        items.append({
            "instruction": make_instruction(s),
            "response": synth_response(s)
        })

    conn.close()

    if not items:
        print("No data to export. Run your app to collect logs first.")
        return

    # train/val split
    random.shuffle(items)
    k = max(1, int(0.1 * len(items)))
    val = items[:k]
    train = items[k:]

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in train: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(OUT_JSONL_VAL, "w", encoding="utf-8") as f:
        for r in val: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ Wrote {len(train)} train and {len(val)} val records to {OUT_DIR}")

if __name__ == "__main__":
    import json, os, sqlite3, random

    os.makedirs(OUT_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    dates = fetch_dates(conn)
    new_items = []

    # --- Load existing dataset (if any) ---
    existing = []
    existing_dates = set()
    if os.path.exists(OUT_JSONL):
        with open(OUT_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                # crude way to extract date from the text
                for token in row["instruction"].split():
                    if token.count("-") == 2:  # e.g. 2025-10-11
                        existing_dates.add(token)
                        break
                existing.append(row)

    # --- Add only new days ---
    for d in dates:
        if d in existing_dates:
            continue
        s = summarize_for_date(conn, d)
        if s["total_logs"] == 0:
            continue
        new_items.append({"instruction": make_instruction(s),
                          "response": synth_response(s)})

    if new_items:
        print(f"🧩 Found {len(new_items)} new records. Appending...")
        existing.extend(new_items)
        with open(OUT_JSONL, "w", encoding="utf-8") as f:
            for r in existing:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        print("✅ Dataset already up-to-date.")

    conn.close()

