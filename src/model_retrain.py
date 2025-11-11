# background_retrain.py
import os
import threading
import time
from datetime import date

# Optional: only if you want to reset TF after attempting retrain
try:
    from FYP import hard_reset_tf
except ImportError:
    hard_reset_tf = None

LAST_FILE = "last_train_date.txt"

def background_retrain(interval_hours=24):
    """
    Disabled heavy retraining safely.
    This function now only checks daily, logs, and skips to avoid memory spikes.
    """
    def loop():
        while True:
            today = str(date.today())

            # ❌ Retraining intentionally disabled to prevent memory overload
            print(f"⚠️ Background retraining skipped for {today} (disabled to prevent memory leak).")

            # ✅ Still update last_run file to avoid spam
            with open(LAST_FILE, "w") as f:
                f.write(today)

            # ✅ Optional: clear any residual GPU/TF memory
            if hard_reset_tf:
                try:
                    hard_reset_tf()
                    print("🧹 TensorFlow/PyTorch cache flushed after skipped retrain.")
                except Exception as e:
                    print(f"⚠️ Error during cleanup: {e}")

            time.sleep(interval_hours * 3600)

    threading.Thread(target=loop, daemon=True).start()
