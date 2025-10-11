# background_retrain.py
import os, threading, subprocess, time
from datetime import date

DATA_BUILDER = "build_dataset_from_sqlite.py"
TRAIN_SCRIPT = "train_local_chatbot.py"
LAST_FILE = "last_train_date.txt"

def _run_subprocess(cmd):
    """Run a Python script silently."""
    try:
        subprocess.run(["python", cmd],
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL,
                       check=True)
    except Exception as e:
        print(f"⚠️ Background retrain error: {e}")

def background_retrain(interval_hours=24):
    """Run once per day in a background thread."""
    def loop():
        while True:
            today = str(date.today())
            # Has this day's training been done?
            if not os.path.exists(LAST_FILE) or open(LAST_FILE).read().strip() != today:
                print("🧩 Updating dataset and retraining in background...")
                _run_subprocess(DATA_BUILDER)
                _run_subprocess(TRAIN_SCRIPT)
                with open(LAST_FILE, "w") as f:
                    f.write(today)
                print("✅ Background model update complete.")
            # Sleep for given interval before checking again
            time.sleep(interval_hours * 3600)
    threading.Thread(target=loop, daemon=True).start()
