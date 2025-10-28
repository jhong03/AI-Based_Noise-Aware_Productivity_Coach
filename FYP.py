import os
import sys
import sqlite3
import time
from datetime import datetime, date, timedelta, timezone
import csv
import numpy as np
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub
import psutil
import gc
import threading
from collections import deque

# =============================
# === TIMEZONE HELPERS (NEW) ==
# =============================

def now_local():
    """Return an aware datetime in local timezone."""
    return datetime.now().astimezone()

def now_iso_local():
    """ISO 8601 with offset, e.g., 2025-10-13T19:15:02+08:00"""
    return now_local().isoformat(timespec="seconds")

def today_local_bounds():
    """
    Return (start, end) ISO strings for the current local day:
    [YYYY-mm-ddT00:00:00+08:00, next day 00:00:00+08:00)
    """
    now = now_local()
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    return start.isoformat(timespec="seconds"), end.isoformat(timespec="seconds")

# =============================
# === PERFORMANCE MODULES ===
# =============================

def memory_guard(threshold_mb: int = 2000, check_interval_sec: int = 5):
    """Monitor and limit memory usage for TensorFlow, Torch, and Python GC."""
    proc = psutil.Process()

def hard_reset_tf():
    """
    Flush TensorFlow + GPU cache safely (one-time action, not a loop).
    """
    try:
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("✅ TensorFlow + GPU cache flushed")
    except Exception as e:
        print(f"⚠️ TF reset error: {e}")


class MemoryGuard:
    """
    Runs in the background:
    - Monitors RAM usage of the whole Python process
    - If memory exceeds threshold_mb → clears TF / Torch caches
    """

    def __init__(self, threshold_mb=5000, check_interval_sec=5):
        self.threshold_mb = threshold_mb
        self.check_interval_sec = check_interval_sec
        self.proc = psutil.Process(os.getpid())
        self.active = True

    def start(self):
        t = threading.Thread(target=self._loop, daemon=True, name="MemoryGuard")
        t.start()

    def _loop(self):
        tick = 0
        while self.active:
            try:
                mem_mb = self.proc.memory_info().rss / (1024 * 1024)

                if mem_mb > self.threshold_mb:
                    print(f"⚠️ Memory high ({mem_mb:.1f} MB) — performing cleanup...")
                    self.cleanup()
                elif tick % 10 == 0:
                    print(f"🧠 Memory OK: {mem_mb:.1f} MB")

                tick += 1
                time.sleep(self.check_interval_sec)

            except Exception as e:
                print(f"⚠️ MemoryGuard error: {e}")
                time.sleep(self.check_interval_sec)

    def cleanup(self):
        try:
            tf.keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()
        except:
            pass

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

        gc.collect()
        print("✅ Memory cleanup done.")

    def stop(self):
        self.active = False


class AudioRingBuffer:
    def __init__(self, max_chunks: int = 10):
        self._buf = deque(maxlen=max_chunks)
        self._lock = threading.Lock()

    def push(self, chunk: np.ndarray):
        with self._lock:
            self._buf.append(chunk.copy())

    def get_concatenated(self) -> np.ndarray:
        with self._lock:
            if not self._buf:
                return np.array([], dtype=np.float32)
            return np.concatenate(list(self._buf), axis=0)

    def clear(self):
        with self._lock:
            self._buf.clear()

# =============================
# === SETUP AND MODEL LOAD ===
# =============================

try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

TFHUB_CACHE = os.path.join(os.path.expanduser("~"), "Documents", "TFHubCache")
os.environ["TFHUB_CACHE_DIR"] = TFHUB_CACHE
os.makedirs(TFHUB_CACHE, exist_ok=True)

BASE_DIR = os.path.join(os.path.expanduser("~"), "Documents", "NoiseLogs")
DB_PATH = os.path.join(BASE_DIR, "noise_focus.db")
LOG_DIR = os.path.join(BASE_DIR, "logs")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
try:
    yamnet_model = hub.load(YAMNET_URL)
except ValueError:
    print("⚠️ Detected possible corrupt TFHub cache. Clearing and retrying...")
    import shutil
    shutil.rmtree(TFHUB_CACHE, ignore_errors=True)
    os.makedirs(TFHUB_CACHE, exist_ok=True)
    yamnet_model = hub.load(YAMNET_URL)

class_map_path = yamnet_model.class_map_path().numpy().decode("utf-8")

# --- Compiled inference wrapper to prevent retracing & memory bloat ---
@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32, name="waveform")])
def _predict_yamnet(waveform):
    # YAMNet SavedModel signature does NOT accept 'training'
    return yamnet_model(waveform)

# Optional: one-time warmup to build the graph (16 kHz * 0.96 s = 15360 samples)
try:
    _ = _predict_yamnet(tf.zeros([15360], dtype=tf.float32))
except Exception as _warmup_e:
    # Non-fatal; continue without warmup if device/runtime complains
    pass

class_names = []
with tf.io.gfile.GFile(class_map_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        class_names.append(row["display_name"])
print(f"✅ Loaded {len(class_names)} class names")

memory_guard(threshold_mb=1500, check_interval_sec=5)
audio_buffer = AudioRingBuffer(max_chunks=10)

# =============================
# === DATABASE INITIALIZATION ===
# =============================

def init_storage():
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS PomodoroSession (
        session_id INTEGER PRIMARY KEY AUTOINCREMENT,
        start_time TEXT NOT NULL,  -- store ISO 8601 with offset
        end_time   TEXT,
        status TEXT CHECK(status IN ('Completed', 'Aborted', 'Running')) NOT NULL,
        focus_score REAL
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS NoiseLog (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,   -- ISO 8601 with offset
        db_level REAL NOT NULL,
        noise_category TEXT CHECK(noise_category IN ('Quiet', 'Moderate', 'Noisy')) NOT NULL,
        label TEXT NOT NULL,
        confidence REAL NOT NULL,
        session_id INTEGER,
        FOREIGN KEY (session_id) REFERENCES PomodoroSession(session_id)
    )
    """)
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
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS DailyReport (
        report_id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,  -- keep as YYYY-MM-DD (local)
        avg_db REAL NOT NULL,
        most_common_label TEXT,
        total_focus_time INTEGER,
        daily_focus_score REAL
    )
    """)
    conn.commit()
    conn.close()
    print(f"✅ Storage initialized at {BASE_DIR}")

def init_reports_table():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS ReportHistory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,      -- YYYY-MM-DD (local)
            summary TEXT,
            ai_report TEXT,
            created_at TEXT          -- ISO 8601 with offset (do NOT rely on CURRENT_TIMESTAMP/UTC)
        )
    """)
    conn.commit()
    conn.close()
    print("✅ ReportHistory table ready.")

# =============================
# === NOISE CAPTURE + CLASSIFY ===
# =============================

def get_db_level(duration=0.96, samplerate=16000):
    try:
        recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
        sd.wait()
        rms = np.sqrt(np.mean(np.square(recording)))
        db = 20 * np.log10(rms + 1e-6)
        db_spl = db + 100
        return recording.flatten(), round(db_spl, 2)
    except Exception as e:
        print(f"⚠️ Mic capture error: {e}")
        return np.zeros(int(duration * samplerate)), 0.0

def noise_category(db):
    thresholds = {range(0, 41): "Quiet", range(41, 71): "Moderate"}
    for r, label in thresholds.items():
        if int(db) in r:
            return label
    return "Noisy"

def classify_sound(audio_chunk):
    """Run YAMNet classification with a compiled graph (no retracing, no 'training' kwarg)."""
    try:
        # ensure float32 tensor of shape [N]
        waveform = tf.convert_to_tensor(audio_chunk, dtype=tf.float32)
        # call compiled function
        scores, _, _ = _predict_yamnet(waveform)
        # average over time frames (scores is [frames, 521])
        mean_scores = tf.reduce_mean(scores, axis=0).numpy()
        top_class = int(np.argmax(mean_scores))
        return class_names[top_class], float(mean_scores[top_class])
    except Exception as e:
        print(f"⚠️ classify_sound error: {e}")
        return "Unknown", 0.0


# =============================
# === DATABASE LOGGING ===
# =============================

def save_noise_log(db_level, category, label, confidence, session_id=None):
    timestamp = now_iso_local()  # <<< CHANGED
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO NoiseLog (timestamp, db_level, noise_category, label, confidence, session_id)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (timestamp, db_level, category, label, confidence, session_id))
    conn.commit()
    conn.close()

    logfile = os.path.join(LOG_DIR, f"{date.today()}_pomodoro_logs.txt" if session_id else f"{date.today()}_logs.txt")
    with open(logfile, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} | {db_level:.2f} dB | {category} | {label} ({confidence:.2f})\n")

    print(f"💾 {db_level:.2f} dB | {category} | {label} ({confidence:.2f}) -> {os.path.basename(logfile)}")

# =============================
# === REPORTING ===
# =============================

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

    focus_score = float(max(0, 100 - avg_db))

    cursor.execute("""
        INSERT INTO PomodoroReport (session_id, avg_db, most_common_label, focus_score)
        VALUES (?, ?, ?, ?)
    """, (session_id, avg_db, most_common_label, focus_score))

    conn.commit()
    conn.close()

    outfile = os.path.join(REPORT_DIR, f"pomodoro_report_{session_id}.txt")
    with open(outfile, "w", encoding="utf-8") as f:
        f.write(f"🧠 Pomodoro Session Report\n")
        f.write(f"Session ID: {session_id}\n")
        f.write(f"Average dB: {avg_db:.2f}\n")
        f.write(f"Most Common Label: {most_common_label}\n")
        f.write(f"Focus Score: {focus_score:.2f}\n")
        f.write(f"Generated at: {now_iso_local()}\n")

    print(f"📝 Saved pomodoro report -> {outfile}")

def generate_daily_report():
    """Generate a daily noise summary with proper timezone handling."""
    # Compute local-day boundaries (start and end of today)
    start_iso, end_iso = today_local_bounds()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # --- Average dB (local-day filter)
    cursor.execute("""
        SELECT AVG(db_level) FROM NoiseLog
        WHERE timestamp >= ? AND timestamp < ?
    """, (start_iso, end_iso))
    avg_db = cursor.fetchone()[0] or 0

    # --- Most common label
    cursor.execute("""
        SELECT label, COUNT(*) as cnt FROM NoiseLog
        WHERE timestamp >= ? AND timestamp < ?
        GROUP BY label
        ORDER BY cnt DESC
        LIMIT 1
    """, (start_iso, end_iso))
    result = cursor.fetchone()
    most_common_label = result[0] if result else "Unknown"

    # --- Calculate total Pomodoro focus time (using proper timezone parsing)
    cursor.execute("""
        SELECT start_time, end_time FROM PomodoroSession
        WHERE start_time >= ?AND start_time < ?AND status='Completed'
    """, (start_iso, end_iso))
    rows = cursor.fetchall()

    total_focus_time = 0.0
    for start_str, end_str in rows:
        if start_str and end_str:
            try:
                start_dt = datetime.fromisoformat(start_str)
                end_dt = datetime.fromisoformat(end_str)
                diff_min = (end_dt - start_dt).total_seconds() / 60.0
                total_focus_time += diff_min
            except Exception as e:
                print(f"⚠️ Error parsing times: {e}")

    daily_focus_score = 0.0

    # --- Store daily record (date only, local)
    local_yyyy_mm_dd = now_local().date().isoformat()
    cursor.execute("""
        INSERT INTO DailyReport (date, avg_db, most_common_label, total_focus_time, daily_focus_score)
        VALUES (?, ?, ?, ?, ?)
    """, (local_yyyy_mm_dd, avg_db, most_common_label, total_focus_time, daily_focus_score))

    conn.commit()
    conn.close()

    # --- Save readable text report
    outfile = os.path.join(REPORT_DIR, f"daily_report_{local_yyyy_mm_dd}.txt")
    with open(outfile, "w", encoding="utf-8") as f:
        f.write(f"📅 Daily Report for {local_yyyy_mm_dd}\n")
        f.write(f"Average dB: {avg_db:.2f}\n")
        f.write(f"Most Common Label: {most_common_label}\n")
        f.write(f"Total Focus Time: {total_focus_time:.1f} minutes\n")
        f.write(f"Daily Focus Score: {daily_focus_score:.2f}\n")
        f.write(f"Generated at: {now_iso_local()}\n")

    print(f"📝 Saved daily report -> {outfile}")

def unload_ai_report_model(model=None):
    """
    Frees memory used by the AI report-generation model
    (e.g., TinyLlama, Mistral, Zephyr, etc.).
    Call this right after report generation.
    """
    try:
        print("🧹 Unloading AI report-generation model...")
        if model is not None:
            del model
        # Clear potential framework caches
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            tf.keras.backend.clear_session()
        except Exception:
            pass
        gc.collect()
        print("✅ AI report-generation model unloaded.")
    except Exception as e:
        print(f"⚠️ Failed to unload report model: {e}")

# =============================
# === CONTINUOUS MONITOR ===
# =============================

def continuous_monitor(run_time=10, session_id=None):
    start = time.time()
    while (time.time() - start) < run_time:
        audio_chunk, db = get_db_level()
        audio_buffer.push(audio_chunk)
        window = audio_buffer.get_concatenated()
        if len(window) > 0:
            category = noise_category(db)
            label, confidence = classify_sound(window)
            save_noise_log(db, category, label, confidence, session_id)
        time.sleep(0.05)

# =============================
# === MAIN APP ===
# =============================

if __name__ == "__main__":
    init_storage()
    init_reports_table()

    pomodoro_mode = True
    session_id = None

    if pomodoro_mode:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO PomodoroSession (start_time, status)
            VALUES (?, ?)
        """, (now_iso_local(), "Completed"))  # <<< CHANGED
        session_id = cursor.lastrowid
        conn.commit()
        conn.close()

    print("🎤 Running Noise Monitor (10s test)...")
    continuous_monitor(run_time=10, session_id=session_id)

    if pomodoro_mode:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE PomodoroSession
            SET end_time=?, status=?
            WHERE session_id=?
        """, (now_iso_local(), "Completed", session_id))  # <<< CHANGED
        conn.commit()
        conn.close()
        generate_pomodoro_report(session_id)

    generate_daily_report()
    print("✅ Finished")
