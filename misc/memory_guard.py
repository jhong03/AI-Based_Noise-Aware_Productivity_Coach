# memory_guard.py
# Soft guard that trims TF, Torch, Python GC, and Windows working set when near cap.

import os
import time
import gc
import threading
import psutil

def _trim_tf_torch():
    try:
        import tensorflow as tf
        try:
            tf.keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()
        except Exception:
            pass
    except Exception:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

def _trim_working_set_windows():
    if os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        GetCurrentProcess = kernel32.GetCurrentProcess
        GetCurrentProcess.restype = wintypes.HANDLE

        EmptyWorkingSet = psapi.EmptyWorkingSet
        EmptyWorkingSet.argtypes = [wintypes.HANDLE]
        EmptyWorkingSet.restype  = wintypes.BOOL

        hProc = GetCurrentProcess()
        EmptyWorkingSet(hProc)
    except Exception:
        pass
# memory_guard.py

_current_threshold = 1500  # Default 1.5 GB

def memory_guard(threshold_mb=1500, check_interval_sec=5):
    global _current_threshold
    _current_threshold = threshold_mb

def increase_memory_cap(new_threshold_mb):
    global _current_threshold
    print(f"⤴ Temporary memory cap set to {new_threshold_mb} MB")
    _current_threshold = new_threshold_mb

def restore_memory_cap():
    global _current_threshold
    print("⤵ Memory cap restored to 1500 MB")
    _current_threshold = 1500


def start_memory_guard(threshold_mb: int = 4800, check_interval_s: int = 5, verbose_heartbeat_s: int = 60):
    """
    Start a background guard that keeps the *current process* memory below threshold.
    This complements the OS hard cap (if present) and avoids stalls.
    """
    proc = psutil.Process(os.getpid())

    def _loop():
        last_heartbeat = 0.0
        while True:
            try:
                mem_mb = proc.memory_info().rss / (1024 * 1024)
                now = time.time()

                if mem_mb > threshold_mb:
                    print(f"⚠️ Memory guard: {mem_mb:.1f} MB > {threshold_mb} MB — trimming caches")
                    _trim_tf_torch()
                    gc.collect()
                    _trim_working_set_windows()
                    mem_mb_after = proc.memory_info().rss / (1024 * 1024)
                    print(f"✅ Trim complete: {mem_mb_after:.1f} MB")

                elif now - last_heartbeat > verbose_heartbeat_s:
                    print(f"🧠 Memory OK: {mem_mb:.1f} MB")
                    last_heartbeat = now

                time.sleep(check_interval_s)
            except Exception:
                time.sleep(check_interval_s)

    t = threading.Thread(target=_loop, name="MemoryGuard", daemon=True)
    t.start()
    return t
