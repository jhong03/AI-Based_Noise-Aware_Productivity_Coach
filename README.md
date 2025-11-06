# 🎧 AI-Based Noise-Aware Productivity Coach

An intelligent desktop app that analyzes environmental noise levels in real time using **YAMNet** and provides feedback to help users maintain focus and productivity.

---

## 🚀 Features
- 🎤 Real-time environmental sound detection using TensorFlow YAMNet  
- 🧠 AI-driven noise awareness feedback via GUI  
- 💾 Local SQLite-based logging of productivity sessions  
- 📊 Visual analytics dashboard for historical sound patterns  
- ⏱️ Two operation modes: Passive and Pomodoro  

---

## 🧰 Tech Stack
- **Python 3.10+**
- **TensorFlow Hub (YAMNet)**
- **Tkinter GUI**
- **SQLite** (for local data storage)
- **Matplotlib** (for visualization)

---

## ⚙️ Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/AI-Based_Noise-Aware_Productivity_Coach.git
cd AI-Based_Noise-Aware_Productivity_Coach

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
