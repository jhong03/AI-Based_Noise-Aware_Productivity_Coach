AI-Based Noise-Aware Productivity Coach

An intelligent desktop app that analyzes environmental noise levels in real-time using YAMNet and provides feedback to help users maintain focus and productivity.

Features
Real-time environmental sound detection using TensorFlow YAMNet

Noise awareness feedback via GUI

Local SQLite-based logging of productivity sessions

Visual analytics for historical sound patterns

Tech Stack
Python 3.10+

TensorFlow Hub (YAMNet)

Tkinter GUI

SQLite for data logging

Matplotlib for visualization

Setup
git clone https://github.com/<your-username>/AI-NoiseAware-Coach.git
cd AI-NoiseAware-Coach
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

▶️ Run the App
python gui_app.py


or use the batch file:

run_app.bat
