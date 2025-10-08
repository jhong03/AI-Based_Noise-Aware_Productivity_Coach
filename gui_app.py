import tkinter as tk
from tkinter import messagebox, ttk
import threading
import time
import sqlite3
from datetime import datetime
import os

# === import your backend code ===
from FYP import init_storage, get_db_level, noise_category, classify_sound, save_noise_log

# Initialize storage once
init_storage()


# === Local DB connection helper ===
def get_connection():
    """Return a new SQLite connection to the same DB used by FYP.py."""
    db_path = os.path.join(os.path.expanduser("~"), "Documents", "NoiseLogs", "noise_focus.db")
    return sqlite3.connect(db_path, timeout=10, check_same_thread=False)


# === Tkinter App ===
class NoiseAwareApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Noise-Aware Productivity Coach")
        self.geometry("600x400")
        self.resizable(False, False)

        # Theme configuration
        self.themes = {
            "light": {
                "bg": "#f0f0f0",
                "fg": "#1e1e1e",
                "button_bg": "#ffffff",
                "button_fg": "#1e1e1e",
                "button_active_bg": "#e0e0e0",
                "entry_bg": "#ffffff",
                "entry_fg": "#1e1e1e",
                "accent": "#4a90e2",
                "progress_trough": "#d9d9d9",
            },
            "dark": {
                "bg": "#1f2933",
                "fg": "#f5f5f5",
                "button_bg": "#323f4b",
                "button_fg": "#f5f5f5",
                "button_active_bg": "#3e4c59",
                "entry_bg": "#3e4c59",
                "entry_fg": "#f5f5f5",
                "accent": "#9f7aea",
                "progress_trough": "#52606d",
            },
        }
        self.current_theme = "light"
        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            # Fallback to default theme if clam is not available
            self.style.theme_use("default")

        # Recording state
        self.recording_active = True
        self.app_running = True  # flag to stop threads safely
        self.session_id = None   # Pomodoro session ID tracking

        # Handle window close safely
        self.protocol("WM_DELETE_WINDOW", self.safe_quit)

        # Start passive monitoring in background
        threading.Thread(target=self.passive_monitor, daemon=True).start()

        # Container for frames (pages)
        self.container = tk.Frame(self)
        self.container.pack(side="top", fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (MainMenu, PomodoroPage, ReportPage, SettingsPage):
            frame = F(self.container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.apply_theme()
        self.show_frame(MainMenu)

    def show_frame(self, page):
        frame = self.frames[page]
        frame.tkraise()

    def passive_monitor(self):
        """Continuously record until app closes or muted."""
        while self.app_running:
            if self.recording_active:
                audio_chunk, db = get_db_level()
                category = noise_category(db)
                label, confidence = classify_sound(audio_chunk)
                save_noise_log(db, category, label, confidence, session_id=self.session_id)
            else:
                time.sleep(1)  # pause monitoring when muted

    def start_pomodoro_session(self):
        """Start a new Pomodoro session (DB insert)."""
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO PomodoroSession (start_time, status)
            VALUES (?, ?)
        """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Running"))
        self.session_id = cursor.lastrowid
        conn.commit()
        conn.close()
        print(f"▶️ Pomodoro session started (ID={self.session_id})")

    def end_pomodoro_session(self, status="Completed"):
        """End current Pomodoro session."""
        if not self.session_id:
            return
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE PomodoroSession
            SET end_time=?, status=?
            WHERE session_id=?
        """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), status, self.session_id))
        conn.commit()
        conn.close()
        print(f"⏹ Pomodoro session {self.session_id} ended as {status}")
        self.session_id = None

    def safe_quit(self):
        """Stop threads and exit safely."""
        if messagebox.askokcancel("Quit", "Are you sure you want to exit?"):
            self.app_running = False
            self.destroy()

    def set_theme(self, theme_name):
        """Update the current theme and refresh UI colors."""
        if theme_name not in self.themes:
            return
        if theme_name == self.current_theme:
            return
        self.current_theme = theme_name
        self.apply_theme()

    def apply_theme(self):
        """Apply the current theme colors to the entire UI."""
        colors = self.themes[self.current_theme]
        self.configure(bg=colors["bg"])
        self.container.configure(bg=colors["bg"])
        self.style.configure(
            "Theme.Horizontal.TProgressbar",
            background=colors["accent"],
            troughcolor=colors["progress_trough"],
            bordercolor=colors["bg"],
            lightcolor=colors["accent"],
            darkcolor=colors["accent"],
        )

        for frame in self.frames.values():
            self._apply_theme_to_widget(frame, colors)
            if hasattr(frame, "on_theme_applied"):
                frame.on_theme_applied(colors)

    def _apply_theme_to_widget(self, widget, colors):
        """Recursively apply theme colors to widgets within frames."""
        if isinstance(widget, (tk.Frame, tk.LabelFrame, tk.Toplevel)):
            widget.configure(bg=colors["bg"])

        if isinstance(widget, tk.Label):
            widget.configure(bg=colors["bg"], fg=colors["fg"])
        elif isinstance(widget, tk.Button):
            widget.configure(
                bg=colors["button_bg"],
                fg=colors["button_fg"],
                activebackground=colors["button_active_bg"],
                activeforeground=colors["button_fg"],
                highlightbackground=colors["bg"],
            )
        elif isinstance(widget, tk.Radiobutton):
            widget.configure(
                bg=colors["bg"],
                fg=colors["fg"],
                activebackground=colors["button_active_bg"],
                selectcolor=colors["bg"],
                highlightbackground=colors["bg"],
            )
        elif isinstance(widget, tk.Entry):
            widget.configure(
                bg=colors["entry_bg"],
                fg=colors["entry_fg"],
                insertbackground=colors["fg"],
                highlightbackground=colors["bg"],
            )
        elif isinstance(widget, ttk.Progressbar):
            widget.configure(style="Theme.Horizontal.TProgressbar")

        for child in widget.winfo_children():
            self._apply_theme_to_widget(child, colors)


# === Main Menu Page ===
class MainMenu(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)

        self.controller = controller
        self.mic_test_running = False  # mic test state

        # === Top Navigation Bar (icons) ===
        report_icon = tk.Label(self, text="📑", font=("Arial", 16), cursor="hand2")
        report_icon.place(relx=0.05, rely=0.05, anchor="nw")
        report_icon.bind("<Button-1>", lambda e: controller.show_frame(ReportPage))

        settings_icon = tk.Label(self, text="⚙️", font=("Arial", 16), cursor="hand2")
        settings_icon.place(relx=0.95, rely=0.05, anchor="ne")
        settings_icon.bind("<Button-1>", lambda e: controller.show_frame(SettingsPage))

        # === Date & Time ===
        self.datetime_label = tk.Label(self, text="", font=("Arial", 12))
        self.datetime_label.pack(pady=5)
        self.update_datetime()

        # === Title ===
        tk.Label(self, text="NOISE-AWARE PRODUCTIVITY COACH",
                 font=("Arial", 16, "bold")).pack(pady=10)

        # === Buttons ===
        tk.Button(self, text="Start Pomodoro Session",
                  command=lambda: controller.show_frame(PomodoroPage)).pack(pady=5)

        # Mic Test button
        self.mic_test_btn = tk.Button(self, text="🎤 Start Mic Test",
                                      command=self.toggle_mic_test)
        self.mic_test_btn.pack(pady=5)

        # Progress bar (hidden until test starts)
        self.progress = ttk.Progressbar(self, orient="horizontal",
                                        length=300, mode="determinate", maximum=100)
        self.progress.pack(pady=5)
        self.progress.pack_forget()

        # Speaker toggle button
        self.speaker_btn = tk.Button(self, text="🔊", font=("Arial", 14),
                                     command=self.toggle_recording)
        self.speaker_btn.pack(pady=5)

        # Quit button
        tk.Button(self, text="❌ Quit", fg="red",
                  command=controller.safe_quit).pack(pady=10)

        # === Memo ===
        tk.Label(self, text="Memo:").pack()
        self.memo_entry = tk.Entry(self, width=40)
        self.memo_entry.pack(pady=5)

        # === Tip of the Day ===
        tk.Label(self, text="Tip of the Day: Stay Hydrated!",
                 font=("Arial", 10, "italic")).pack(pady=10)

    def update_datetime(self):
        now = datetime.now().strftime("%b %d, %Y %I:%M %p")
        self.datetime_label.config(text=now)
        self.after(1000, self.update_datetime)

    def toggle_mic_test(self):
        """Toggle microphone intensity testing."""
        if not self.mic_test_running:
            self.mic_test_running = True
            self.mic_test_btn.config(text="🛑 Stop Mic Test")
            self.progress.pack()
            self.update_mic_bar()
        else:
            self.mic_test_running = False
            self.mic_test_btn.config(text="🎤 Start Mic Test")
            self.progress.pack_forget()

    def update_mic_bar(self):
        if not self.mic_test_running:
            return
        audio_chunk, db = get_db_level()
        intensity = min(max(db, 0), 100)
        self.progress["value"] = intensity
        self.after(50, self.update_mic_bar)

    def toggle_recording(self):
        """Toggle passive monitoring ON/OFF."""
        self.controller.recording_active = not self.controller.recording_active
        self.speaker_btn.config(text="🔊" if self.controller.recording_active else "🔇")


# === Pomodoro Page ===
class PomodoroPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.running = False
        self.remaining = 0

        tk.Label(self, text="Pomodoro Timer", font=("Arial", 14, "bold")).pack(pady=10)

        self.timer_label = tk.Label(self, text="00:00", font=("Arial", 28))
        self.timer_label.pack(pady=10)

        tk.Button(self, text="25 min Session",
                  command=lambda: self.start_pomodoro(25 * 60, is_work=True)).pack(pady=5)
        tk.Button(self, text="5 min Break",
                  command=lambda: self.start_pomodoro(5 * 60, is_work=False)).pack(pady=5)
        tk.Button(self, text="15 min Break",
                  command=lambda: self.start_pomodoro(15 * 60, is_work=False)).pack(pady=5)

        tk.Button(self, text="Reset", command=self.reset_timer).pack(pady=10)
        tk.Button(self, text="Back", command=lambda: controller.show_frame(MainMenu)).pack(pady=5)

    def start_pomodoro(self, seconds, is_work=True):
        if self.running:
            return
        self.running = True
        self.remaining = seconds

        if is_work:
            self.controller.start_pomodoro_session()

        self.update_timer(is_work)

    def update_timer(self, is_work):
        if not self.running:
            return
        mins, secs = divmod(self.remaining, 60)
        self.timer_label.config(text=f"{mins:02d}:{secs:02d}")
        if self.remaining > 0:
            self.remaining -= 1
            self.after(1000, lambda: self.update_timer(is_work))
        else:
            self.running = False
            if is_work:
                self.controller.end_pomodoro_session(status="Completed")
                messagebox.showinfo("Pomodoro", "Session completed! Take a break.")
            else:
                messagebox.showinfo("Break", "Break time is over!")

    def reset_timer(self):
        if self.running:
            self.running = False
            self.controller.end_pomodoro_session(status="Aborted")
        self.timer_label.config(text="00:00")


# === Report Page (placeholder) ===
class ReportPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        tk.Label(self, text="Report Page (Coming Soon)", font=("Arial", 14)).pack(pady=20)
        tk.Button(self, text="Back", command=lambda: controller.show_frame(MainMenu)).pack()


# === Settings Page ===
class SettingsPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        tk.Label(self, text="Settings", font=("Arial", 16, "bold")).pack(pady=(20, 10))
        tk.Label(self, text="Customize how the Noise-Aware Productivity Coach looks.",
                 font=("Arial", 10)).pack(pady=(0, 20))

        self.theme_var = tk.StringVar(value=controller.current_theme)

        theme_section = tk.Frame(self)
        theme_section.pack(pady=10, padx=20, fill="x")

        tk.Label(theme_section, text="Color Theme", font=("Arial", 12, "bold")).pack(anchor="w")
        tk.Label(theme_section,
                 text="Choose between light and dark appearances to match your environment.",
                 wraplength=400, justify="left").pack(anchor="w", pady=(0, 10))

        options_frame = tk.Frame(theme_section)
        options_frame.pack(anchor="w")

        tk.Radiobutton(options_frame, text="Light", value="light",
                       variable=self.theme_var, command=self.change_theme).pack(anchor="w", pady=2)
        tk.Radiobutton(options_frame, text="Dark", value="dark",
                       variable=self.theme_var, command=self.change_theme).pack(anchor="w", pady=2)

        self.preview_label = tk.Label(
            self,
            text="Preview: Productive focus starts with the right lighting!",
            font=("Arial", 11, "italic"),
            wraplength=420,
            justify="center",
        )
        self.preview_label.pack(pady=25, padx=20)

        tk.Button(self, text="Back", command=lambda: controller.show_frame(MainMenu)).pack(pady=10)

    def change_theme(self):
        self.controller.set_theme(self.theme_var.get())

    def on_theme_applied(self, _colors):
        """Keep the selection synced with the controller state."""
        if self.theme_var.get() != self.controller.current_theme:
            self.theme_var.set(self.controller.current_theme)


# === Run App ===
if __name__ == "__main__":
    app = NoiseAwareApp()
    app.mainloop()
