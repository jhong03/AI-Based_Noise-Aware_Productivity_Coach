# Noise-Aware Productivity Coach (Local, No API Keys)

A desktop coach that **listens to ambient sound**, **classifies noise** (via YAMNet), logs to SQLite, and generates **motivational daily reports** with a **locally fine-tuned LLM** (no internet, no API keys).

## ✨ Features
- Real-time mic monitor + Pomodoro timer (Tkinter GUI)
- YAMNet-based sound classification (TensorFlow Hub)
- Daily & session reports in SQLite
- **Local fine-tuned chatbot** (Transformers + LoRA) — no keys
- Automatic background re-training from your daily logs

## 🖥️ Install
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt

# Install torch for your system if needed:
# CPU: pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
# CUDA 12.1 (example): pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
