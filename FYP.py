import os
import sqlite3
import time
from datetime import datetime, date
import csv

import numpy as np
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub


# === Set permanent cache directory for TF Hub ===
TFHUB_CACHE = os.path.join(os.path.expanduser("~"), "Documents", "TFHubCache")
os.environ["TFHUB_CACHE_DIR"] = TFHUB_CACHE
os.makedirs(TFHUB_CACHE, exist_ok=True)

# === Setup storage paths ===
BASE_DIR = os.path.join(os.path.expanduser("~"), "Documents", "NoiseLogs")
DB_PATH = os.path.join(BASE_DIR, "noise_focus.db")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# === Load YAMNet model once with cache failsafe ===
YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
try:
    yamnet_model = hub.load(YAMNET_URL)
except ValueError as e:
    print("⚠️ Detected possible corrupt TFHub cache. Clearing and retrying...")
    import shutil
    shutil.rmtree(TFHUB_CACHE, ignore_errors=True)  # delete bad cache
    os.makedirs(TFHUB_CACHE, exist_ok=True)
    yamnet_model = hub.load(YAMNET_URL)

class_map_path = yamnet_model.class_map_path().numpy().decode("utf-8")

# Load class names properly
class_names = []
with tf.io.gfile.GFile(class_map_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        class_names.append(row["display_name"])
print(f"✅ Loaded {len(class_names)} class names")  # should be 521


# === Database setup ===
def init_storage():
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Sessions
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS PomodoroSession (
        session_id INTEGER PRIMARY KEY AUTOINCREMENT,
        start_time DATETIME NOT NULL,
        end_time DATETIME,
        status TEXT CHECK(status IN ('Completed', 'Aborted')) NOT NULL,
        focus_score REAL
    )
    """)

    # Logs
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS NoiseLog (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME NOT NULL,
        db_level REAL NOT NULL,
        noise_category TEXT CHECK(noise_category IN ('Quiet', 'Moderate', 'Noisy')) NOT NULL,
        label TEXT NOT NULL,
        confidence REAL NOT NULL,
        session_id INTEGER,
        FOREIGN KEY (session_id) REFERENCES PomodoroSession(session_id)
    )
    """)

    # Per-session reports
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS PomodoroReport (
        report_id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER NOT NULL,
        avg_db REAL NOT NULL,
        most_common_label TEXT,
        focus_score REAL,
        FOREIGN KEY (session_id) REFERENCES PomodoroSession(session_id)
    )
    """)

    # Daily reports
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS DailyReport (
        report_id INTEGER PRIMARY KEY AUTOINCREMENT,
        date DATE NOT NULL,
        avg_db REAL NOT NULL,
        most_common_label TEXT,
        total_focus_time INTEGER,
        daily_focus_score REAL
    )
    """)

    conn.commit()
    conn.close()
    print(f"✅ Storage initialized at {BASE_DIR}")


# === Noise measurement ===
def get_db_level(duration=0.96, samplerate=16000):
    # Use float32 (lighter, matches TF expectations)
    recording = sd.rec(int(duration * samplerate),
                       samplerate=samplerate,
                       channels=1, dtype='float32')
    sd.wait()
    rms = np.sqrt(np.mean(recording ** 2))
    db = 20 * np.log10(rms + 1e-6)
    db_spl = db + 100
    return recording.flatten(), round(db_spl, 2)


def noise_category(db):
    if db <= 40:
        return "Quiet"
    elif db <= 70:
        return "Moderate"
    else:
        return "Noisy"


def classify_sound(audio_chunk):
    waveform = tf.convert_to_tensor(audio_chunk, dtype=tf.float32)
    scores, _, _ = yamnet_model(waveform)
    mean_scores = np.mean(scores, axis=0)
    top_class = np.argmax(mean_scores)
    return class_names[top_class], float(mean_scores[top_class])


# === Save logs ===
def save_noise_log(db_level, category, label, confidence, session_id=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save to DB
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO NoiseLog (timestamp, db_level, noise_category, label, confidence, session_id)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (timestamp, db_level, category, label, confidence, session_id))
    conn.commit()
    conn.close()

    # Decide which log file to write
    if session_id is not None:
        logfile = os.path.join(LOG_DIR, f"{datetime.now().date()}_pomodoro_logs.txt")
    else:
        logfile = os.path.join(LOG_DIR, f"{datetime.now().date()}_logs.txt")

    # Save to text log
    with open(logfile, "a") as f:
        f.write(f"{timestamp} | {db_level:.2f} dB | {category} | {label} ({confidence:.2f})\n")

    print(f"💾 {db_level:.2f} dB | {category} | {label} ({confidence:.2f}) -> {os.path.basename(logfile)}")



# === Reporting ===
def generate_pomodoro_report(session_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT AVG(db_level) FROM NoiseLog WHERE session_id=?", (session_id,))
    avg_db = cursor.fetchone()[0] or 0

    cursor.execute("""
        SELECT label, COUNT(*) as cnt
        FROM NoiseLog
        WHERE session_id=?
        GROUP BY label
        ORDER BY cnt DESC
        LIMIT 1
    """, (session_id,))
    result = cursor.fetchone()
    most_common_label = result[0] if result else "Unknown"

    focus_score = 0.0  # TODO: replace with MLP later

    cursor.execute("""
        INSERT INTO PomodoroReport (session_id, avg_db, most_common_label, focus_score)
        VALUES (?, ?, ?, ?)
    """, (session_id, avg_db, most_common_label, focus_score))

    conn.commit()
    conn.close()
    print(f"📊 PomodoroReport generated for session {session_id}")


def generate_daily_report():
    today = date.today()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT AVG(db_level) FROM NoiseLog WHERE DATE(timestamp)=?", (today,))
    avg_db = cursor.fetchone()[0] or 0

    cursor.execute("""
        SELECT label, COUNT(*) as cnt
        FROM NoiseLog
        WHERE DATE(timestamp)=?
        GROUP BY label
        ORDER BY cnt DESC
        LIMIT 1
    """, (today,))
    result = cursor.fetchone()
    most_common_label = result[0] if result else "Unknown"

    cursor.execute("""
        SELECT SUM(strftime('%s', end_time) - strftime('%s', start_time)) / 60
        FROM PomodoroSession
        WHERE DATE(start_time)=? AND status='Completed'
    """, (today,))
    total_focus_time = cursor.fetchone()[0] or 0

    daily_focus_score = 0.0  # TODO: replace with MLP later

    cursor.execute("""
        INSERT INTO DailyReport (date, avg_db, most_common_label, total_focus_time, daily_focus_score)
        VALUES (?, ?, ?, ?, ?)
    """, (today, avg_db, most_common_label, total_focus_time, daily_focus_score))

    conn.commit()
    conn.close()
    print(f"📊 DailyReport generated for {today}")


# === Monitoring loop ===
def continuous_monitor(run_time=10, session_id=None):
    start = time.time()
    while (time.time() - start) < run_time:
        audio_chunk, db = get_db_level()
        category = noise_category(db)
        label, confidence = classify_sound(audio_chunk)

        save_noise_log(db, category, label, confidence, session_id)

        # tiny sleep prevents Windows buffer crash
        time.sleep(0.01)


# === Main entry ===
if __name__ == "__main__":
    init_storage()

    pomodoro_mode = True  # or False depending on what you run

    session_id = None
    if pomodoro_mode:
        # Start Pomodoro session
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO PomodoroSession (start_time, status)
            VALUES (?, ?)
        """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Completed"))
        session_id = cursor.lastrowid
        conn.commit()
        conn.close()

    print("🎤 Running Noise Monitor (10s test)...")
    continuous_monitor(run_time=10, session_id=session_id)

    if pomodoro_mode:
        # End Pomodoro session
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE PomodoroSession
            SET end_time=?, status=?
            WHERE session_id=?
        """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Completed", session_id))
        conn.commit()
        conn.close()

        # Generate reports
        generate_pomodoro_report(session_id)

    # Always generate daily report
    generate_daily_report()
    print("✅ Finished")