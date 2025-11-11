# 🎧 Noise-Aware Productivity Coach

An intelligent desktop app that analyzes environmental noise using TensorFlow YAMNet
and generates AI-based productivity feedback using TinyLlama.

## 🚀 Features
- Real-time noise classification via YAMNet  
- Local AI productivity report generator (TinyLlama)  
- Pomodoro session tracking  
- Daily/Session reports saved in SQLite  
- Tkinter GUI with visual analytics  

## 🧠 Tech Stack
- Python 3.10+
- TensorFlow Hub (YAMNet)
- PyTorch + Transformers (TinyLlama)
- Tkinter GUI
- SQLite

## ⚙️ Setup
```bash
git clone https://github.com/<your-username>/NoiseAwareCoach.git
cd NoiseAwareCoach
pip install -r requirements.txt
python gui_app.py

## 📦 Model Download

The TinyLlama model is **not included** due to GitHub’s 100 MB file limit.

Download it once from [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)

