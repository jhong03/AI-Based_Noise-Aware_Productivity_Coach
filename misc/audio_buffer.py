# audio_buffer.py
from collections import deque
import numpy as np
import threading

class AudioRingBuffer:
    """
    Thread-safe fixed-length ring buffer for audio frames (numpy arrays).
    Stores the most recent N chunks to avoid unbounded memory growth.
    """
    def __init__(self, max_chunks: int = 10):
        self._buf = deque(maxlen=max_chunks)
        self._lock = threading.Lock()

    def push(self, chunk: np.ndarray):
        with self._lock:
            self._buf.append(chunk.copy() if isinstance(chunk, np.ndarray) else np.array(chunk, dtype=np.float32))

    def get_concatenated(self) -> np.ndarray:
        with self._lock:
            if not self._buf:
                return np.array([], dtype=np.float32)
            return np.concatenate(list(self._buf), axis=0)

    def clear(self):
        with self._lock:
            self._buf.clear()
