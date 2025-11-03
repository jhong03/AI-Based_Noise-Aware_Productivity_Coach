import tkinter as tk
from tkinter import messagebox, ttk
import threading
import time
import sqlite3
from datetime import datetime, date
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
from tkcalendar import DateEntry
from FYP import memory_guard, MemoryGuard
import sys
from io import StringIO

mem_guard = MemoryGuard(threshold_mb=5000, check_interval_sec=5)
mem_guard.start()
# === import your backend code ===
from FYP import (
    init_storage,
    init_reports_table,
    get_db_level,
    noise_category,
    classify_sound,
    save_noise_log,
    today_local_bounds,
    DB_PATH,
)
# ⛔️ DO NOT import the AI generator here anymore (it loads the model at startup)
# from ai_report_generator_local import generate_ai_report

# Initialize storage once
init_storage()
init_reports_table()
from background_retrain import background_retrain

# start background retraining once app launches
background_retrain(interval_hours=24)

# === Local DB connection helper ===
def get_connection():
    """Return a new SQLite connection to the same DB used by FYP.py."""
    return sqlite3.connect(DB_PATH, timeout=10, check_same_thread=False)


# === Tkinter App ===
class NoiseAwareApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Noise-Aware Productivity Coach")
        self.geometry("1100x700")
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
        self.pomodoro_alert_enabled = True  # toggleable Pomodoro completion alerts
        self.mic_sensitivity = tk.IntVar(value=50)  # microphone sensitivity level
        self._mic_sensitivity_value = self.mic_sensitivity.get()
        self.mic_sensitivity.trace_add("write", self._cache_mic_sensitivity)
        self.preferred_name = ""

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
        for F in (MainMenu, PomodoroPage, ReportPage, DetailedReportPage, SettingsPage):
            frame = F(self.container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.apply_theme()
        self.show_frame(MainMenu)

    def show_frame(self, page):
        frame = self.frames[page]
        if hasattr(frame, "on_show"):
            frame.on_show()
        frame.tkraise()

    def passive_monitor(self):
        """Continuously record until app closes or muted."""
        while self.app_running:
            if self.recording_active:
                sensitivity = self.get_mic_sensitivity()
                audio_chunk, db = get_db_level(sensitivity=sensitivity)
                category = noise_category(db)
                label, confidence = classify_sound(audio_chunk)
                save_noise_log(db, category, label, confidence, session_id=self.session_id)
                # Small sleep to avoid pegging a CPU core

                poll_interval = 0.05 if self.recording_active else 1.0
                # When active but quiet (e.g., low RMS), you can stretch to 0.2 s:
                if db < 35:
                    poll_interval = 0.2
                time.sleep(poll_interval)
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
        """Force-terminate all threads, models, and subprocesses to free memory completely."""
        if messagebox.askokcancel("Quit", "Are you sure you want to exit?"):
            print("🧹 Cleaning up background tasks...")

            # Stop custom threads
            self.app_running = False
            self.recording_active = False

            import gc, os, signal, psutil, time

            # Give threads a moment to stop gracefully
            time.sleep(0.5)

            # 🧠 Try unloading models (AI + TF + Torch)
            try:
                import torch, gc
                torch.cuda.empty_cache()
                gc.collect()
                print("✅ Cleared PyTorch GPU cache and memory.")
            except Exception as e:
                print(f"⚠️ Torch cleanup skipped: {e}")

            try:
                import tensorflow as tf
                tf.keras.backend.clear_session()
                print("✅ TensorFlow session cleared.")
            except Exception as e:
                print(f"⚠️ TensorFlow cleanup skipped: {e}")

            try:
                import torch
                torch.cuda.empty_cache()
                print("✅ PyTorch cache cleared.")
            except Exception:
                pass

            gc.collect()

            # 🧩 Kill all background threads and subprocesses for this PID
            pid = os.getpid()
            process = psutil.Process(pid)
            for child in process.children(recursive=True):
                try:
                    child.terminate()
                except Exception:
                    pass

            # Give them 1s to terminate nicely
            gone, alive = psutil.wait_procs(process.children(recursive=True), timeout=1)
            for p in alive:
                try:
                    p.kill()
                except Exception:
                    pass

            # Close Tk
            try:
                self.destroy()
            except Exception:
                pass

            print("💀 Terminating Python process completely.")
            os.kill(pid, signal.SIGTERM)

    def set_theme(self, theme_name):
        """Update the current theme and refresh UI colors."""
        if theme_name not in self.themes:
            return
        if theme_name == self.current_theme:
            return
        self.current_theme = theme_name
        self.apply_theme()

    def set_pomodoro_alert_enabled(self, enabled):
        """Toggle the Pomodoro completion alert sound."""
        self.pomodoro_alert_enabled = bool(enabled)

    def set_preferred_name(self, name):
        """Persist the trimmed preferred name for downstream use."""
        self.preferred_name = (name or "").strip()

    def get_preferred_name(self):
        """Return the sanitized preferred name (empty string if unset)."""
        return getattr(self, "preferred_name", "").strip()

    def set_mic_sensitivity(self, value):
        """Update the microphone sensitivity (0-100)."""
        try:
            numeric_value = int(float(value))
        except (TypeError, ValueError):
            return
        numeric_value = max(0, min(100, numeric_value))
        if self.mic_sensitivity.get() != numeric_value:
            self.mic_sensitivity.set(numeric_value)

    def get_mic_sensitivity(self):
        """Return the cached microphone sensitivity value for background threads."""
        return getattr(self, "_mic_sensitivity_value", 50)

    def _cache_mic_sensitivity(self, *_):
        """Cache IntVar value so worker threads can read without touching Tk state."""
        try:
            self._mic_sensitivity_value = int(self.mic_sensitivity.get())
        except tk.TclError:
            # Tk may be shutting down; keep the last known value.
            pass

    def adjust_mic_sensitivity(self, delta):
        """Incrementally adjust microphone sensitivity."""
        self.set_mic_sensitivity(self.mic_sensitivity.get() + delta)

    def play_pomodoro_alert(self):
        """Play the default system alert sound for Pomodoro completions."""
        try:
            self.bell()
        except tk.TclError:
            # Fallback: silently ignore if the platform cannot play the bell.
            pass

    def apply_theme(self):
        """Apply the current theme colors to the entire UI."""
        colors = self.themes[self.current_theme]
        self.configure(bg=colors["bg"])
        self.container.configure(bg=colors["bg"])
        self.style.configure(
            "Theme.Horizontal.TProgressbar",
            background=colors["accent"],
            troughcolor=colors["progress_trrough"] if "progress_trrough" in colors else colors["progress_trough"],
            bordercolor=colors["bg"],
            lightcolor=colors["accent"],
            darkcolor=colors["accent"],
        )

        for frame in self.frames.values():
            self._apply_theme_to_widget(frame, colors)
            if hasattr(frame, "on_theme_applied"):
                frame.on_theme_applied(colors)

    def _apply_theme_to_widget(self, widget, colors):
        """Recursively apply theme colors to widgets within frames, skipping special widgets that don't support 'bg'."""
        from tkcalendar import DateEntry
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        # skip widgets that shouldn't be themed recursively
        skip_types = (DateEntry, FigureCanvasTkAgg)
        if isinstance(widget, skip_types):
            return

        # --- Base background containers ---
        if isinstance(widget, (tk.Frame, tk.LabelFrame, tk.Toplevel)):
            widget.configure(bg=colors["bg"])

        if isinstance(widget, tk.Canvas):
            widget.configure(bg=colors["bg"], highlightbackground=colors["bg"])

        # --- Standard widgets ---
        try:
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
            elif isinstance(widget, tk.Checkbutton):
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
            elif isinstance(widget, tk.Text):
                widget.configure(
                    bg=colors["entry_bg"],
                    fg=colors["entry_fg"],
                    insertbackground=colors["fg"],
                    highlightbackground=colors["bg"],
                )
            elif isinstance(widget, tk.Scale):
                widget.configure(
                    bg=colors["bg"],
                    fg=colors["fg"],
                    highlightbackground=colors["bg"],
                    troughcolor=colors.get("progress_trough", colors["bg"]),
                    activebackground=colors["accent"],
                )
            elif isinstance(widget, ttk.Progressbar):
                widget.configure(style="Theme.Horizontal.TProgressbar")
        except Exception:
            # some third-party widgets (e.g. tkcalendar internals) don't accept bg/fg
            pass

        # --- Recurse into children safely ---
        for child in widget.winfo_children():
            self._apply_theme_to_widget(child, colors)

    def show_log_window(self):
        if hasattr(self, "log_window") and self.log_window.winfo_exists():
            self.log_window.lift()
            return

        self.log_window = tk.Toplevel(self)
        self.log_window.title("Live System Logs")
        self.log_window.geometry("650x450")
        self.log_window.configure(bg=self.themes[self.current_theme]["bg"])

        # Text area styled like a terminal
        self.log_text_area = tk.Text(
            self.log_window,
            wrap="word",
            state="disabled",
            bg="#1e1e1e",
            fg="#00ff66",
            insertbackground="#00ff66",
        )
        self.log_text_area.pack(fill="both", expand=True)

        scrollbar = tk.Scrollbar(self.log_window, command=self.log_text_area.yview)
        scrollbar.pack(side="right", fill="y")
        self.log_text_area.config(yscrollcommand=scrollbar.set)

        # ✅ Redirect prints from this point onwards
        self.redirect_print_to_gui()

    class DualOutput:
        def __init__(self, widget):
            self.widget = widget
            self.console = sys.__stdout__

        def write(self, text):
            try:
                if self.widget and self.widget.winfo_exists():
                    self.widget.config(state="normal")
                    self.widget.insert(tk.END, text)
                    self.widget.see(tk.END)
                    self.widget.config(state="disabled")
            except:  # GUI closed → just write to terminal
                pass

            # Always still print to terminal
            self.console.write(text)

        def flush(self):
            self.console.flush()

    # ✅ 3) Bind print() output to the live log window
    def redirect_print_to_gui(self):
        if hasattr(self, "log_text_area"):
            sys.stdout = self.DualOutput(self.log_text_area)


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

        # Terminal / Log icon
        log_icon = tk.Label(self, text="🖥️", font=("Arial", 16), cursor="hand2")
        log_icon.place(relx=0.95, rely=0.95, anchor="sw")  # adjust position
        log_icon.bind("<Button-1>", lambda e: self.controller.show_log_window())

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

        # === Preferred Name Entry ===
        name_frame = tk.Frame(self)
        name_frame.pack(pady=(5, 0))
        tk.Label(name_frame, text="Preferred Name:").pack(side="left", padx=(0, 5))
        self.name_var = tk.StringVar(value=controller.get_preferred_name())
        self.name_entry = tk.Entry(name_frame, textvariable=self.name_var, width=25)
        self.name_entry.pack(side="left")
        self.name_var.trace_add(
            "write", lambda *_: self.controller.set_preferred_name(self.name_var.get())
        )

        # === Buttons ===
        tk.Button(self, text="Start Pomodoro Session",
                  command=self.navigate_to_pomodoro).pack(pady=5)

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

    def navigate_to_pomodoro(self):
        """Ensure a preferred name is provided before opening the Pomodoro page."""
        name = self.controller.get_preferred_name()
        if not name:
            messagebox.showwarning(
                "Preferred Name Required",
                "Please enter your preferred name so we can tailor your Pomodoro support."
            )
            self.name_entry.focus_set()
            return
        if self.name_var.get() != name:
            self.name_var.set(name)
        self.controller.show_frame(PomodoroPage)

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
        audio_chunk, db = get_db_level(sensitivity=self.controller.mic_sensitivity.get())
        intensity = min(max(db, 0), 100)
        self.progress["value"] = intensity
        self.after(50, self.update_mic_bar)

    def on_show(self):
        """Refresh the preferred name entry whenever the page becomes visible."""
        current_name = self.controller.get_preferred_name()
        if self.name_var.get() != current_name:
            self.name_var.set(current_name)

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
                if self.controller.pomodoro_alert_enabled:
                    self.controller.play_pomodoro_alert()
                messagebox.showinfo("Pomodoro", "Session completed! Take a break.")
            else:
                messagebox.showinfo("Break", "Break time is over!")

    def reset_timer(self):
        if self.running:
            self.running = False
            self.controller.end_pomodoro_session(status="Aborted")
        self.timer_label.config(text="00:00")


# === Report Page (Optimized with Loading Spinner and Faster AI Report) ===
class ReportPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.spinner_running = False

        # --- Navigation ---
        nav_frame = tk.Frame(self)
        nav_frame.pack(fill="x", padx=15, pady=(15, 0))
        tk.Button(
            nav_frame,
            text="⬅ Back to Main Menu",
            command=lambda: controller.show_frame(MainMenu),
        ).pack(anchor="w")

        tk.Label(self, text="Daily Productivity Insights", font=("Arial", 14, "bold")).pack(pady=(10, 10))

        # --- Summary Box ---
        self.summary_box = tk.Text(self, height=8, width=70, wrap="word", state="disabled")
        self.summary_box.pack(padx=20, pady=5)

        # --- Generate Button ---
        self.generate_button = tk.Button(self, text="✨ Generate AI Report", command=self.generate_report)
        self.generate_button.pack(pady=10)

        # --- Spinner / Status Label ---
        self.status_label = tk.Label(self, text="", font=("Arial", 10, "italic"), fg="#888")
        self.status_label.pack(pady=(0, 5))

        # --- AI Report Box ---
        self.ai_report_box = tk.Text(self, height=10, width=70, wrap="word", state="disabled")
        self.ai_report_box.pack(padx=20, pady=(5, 15))

        tk.Button(self, text="📊 View Detailed Report",
                  command=lambda: controller.show_frame(DetailedReportPage)).pack(pady=5)

        tk.Button(
            self,
            text="⬅ Back to Main Menu",
            command=lambda: controller.show_frame(MainMenu),
        ).pack(pady=(10, 20))

    # --- On Page Show ---
    def on_show(self):
        self.refresh_summary()

    def refresh_summary(self):
        summary_text = self.build_daily_summary()
        self.summary_box.config(state="normal")
        self.summary_box.delete("1.0", tk.END)
        if summary_text:
            self.summary_box.insert(tk.END, summary_text)
        else:
            self.summary_box.insert(tk.END, "No noise activity recorded yet for today. Start a session to collect data!")
        self.summary_box.config(state="disabled")

    # --- Build Daily Summary ---
    def build_daily_summary(self):
        try:
            conn = get_connection()
            cursor = conn.cursor()
            today = date.today()
            start_iso, end_iso = today_local_bounds()

            # Total logs and average dB
            cursor.execute(
                """
                SELECT COUNT(*), AVG(db_level)
                FROM NoiseLog
                WHERE timestamp >= ? AND timestamp < ?
                """,
                (start_iso, end_iso),
            )
            total_logs, avg_db = cursor.fetchone()
            if not total_logs:
                conn.close()
                return ""

            avg_db = avg_db or 0

            # Noise category counts
            cursor.execute("""
                SELECT noise_category, COUNT(*) FROM NoiseLog
                WHERE timestamp >= ? AND timestamp < ?
                GROUP BY noise_category
            """, (start_iso, end_iso))
            category_counts = {row[0]: row[1] for row in cursor.fetchall()}

            # Most frequent sound label
            cursor.execute("""
                SELECT label, COUNT(*) FROM NoiseLog
                WHERE timestamp >= ? AND timestamp < ?
                GROUP BY label ORDER BY COUNT(*) DESC LIMIT 1
            """, (start_iso, end_iso))
            row = cursor.fetchone()
            top_label = row[0] if row else "Unknown"

            # Completed Pomodoro sessions + duration
            cursor.execute("""
                SELECT COUNT(*),
                       SUM(strftime('%s', COALESCE(end_time, start_time)) - strftime('%s', start_time)) / 60.0
                FROM PomodoroSession
                WHERE start_time >= ? AND start_time < ? AND status='Completed'
            """, (start_iso, end_iso))
            session_count, focus_minutes = cursor.fetchone()
            focus_minutes = focus_minutes or 0

            conn.close()

            quiet = category_counts.get("Quiet", 0)
            moderate = category_counts.get("Moderate", 0)
            noisy = category_counts.get("Noisy", 0)

            def pct(x):
                return (x / total_logs) * 100 if total_logs else 0

            return "\n".join([
                f"Date: {today.strftime('%b %d, %Y')}",
                f"Total noise logs: {total_logs}",
                f"Average sound level: {avg_db:.1f} dB SPL",
                f"Noise mix → Quiet {pct(quiet):.0f}% | Moderate {pct(moderate):.0f}% | Noisy {pct(noisy):.0f}%",
                f"Most common sound: {top_label}",
                f"Completed Pomodoros: {session_count or 0} (≈ {focus_minutes:.0f} min focused work)"
            ])
        except sqlite3.Error as e:
            return f"❌ DB Error: {e}"

    # --- Report Generation with Progress & Lazy model load/unload ---
    def generate_report(self):
        summary_text = self.build_daily_summary()
        if not summary_text:
            messagebox.showinfo("AI Report", "No data available for today.")
            return

        preferred_name = self.controller.get_preferred_name()
        if not preferred_name:
            messagebox.showwarning(
                "Preferred Name Required",
                "Please enter your preferred name on the main menu so the AI report can speak to you directly."
            )
            return

        # progress bar UI
        self.progress = ttk.Progressbar(self, orient="horizontal", length=350, mode="determinate", maximum=100)
        self.progress.pack(pady=5)
        self.status_label.config(text="Initializing...")

        def on_progress(percent, message):
            self.progress["value"] = percent
            self.status_label.config(text=message)
            self.update_idletasks()

        def background_task():
            try:
                # ✅ Use subprocess-safe generator (no manual unload)
                from ai_report_generator_local import generate_ai_report

                try:
                    report = generate_ai_report(
                        summary_text,
                        preferred_name=preferred_name,
                        progress_callback=on_progress
                    )
                except TypeError:
                    # In case callback isn't supported
                    report = generate_ai_report(summary_text, preferred_name=preferred_name)

                # ✅ Save to local DB
                conn = get_connection()
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO ReportHistory (date, summary, ai_report) VALUES (?, ?, ?)",
                    (datetime.now().strftime("%Y-%m-%d"), summary_text, report)
                )
                conn.commit()
                conn.close()

                # ✅ Update GUI safely on main thread
                self.after(0, lambda: self.display_ai_report(report))
                self.after(0, lambda: self.status_label.config(text="✅ Report Ready!"))

            except Exception as e:
                self.after(0, lambda: self.display_ai_report(f"⚠️ Error generating report:\n{e}"))
                self.after(0, lambda: self.status_label.config(text="⚠️ Failed."))
            finally:
                self.after(0, lambda: self.progress.destroy())
                self.after(0, lambda: self.generate_button.config(state="normal"))

        # disable button to prevent double clicks
        self.generate_button.config(state="disabled")
        threading.Thread(target=background_task, daemon=True).start()

    # --- Display Final Report ---
    def display_ai_report(self, text):
        self.ai_report_box.config(state="normal")
        self.ai_report_box.delete("1.0", tk.END)
        self.ai_report_box.insert(tk.END, text)
        self.ai_report_box.config(state="disabled")
        self.generate_button.config(state="normal")


# === Settings Page ===
class SettingsPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        # === Scrollable container ===
        self.columnconfigure(0, weight=1)
        self._canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=scrollbar.set)

        self._canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        content = tk.Frame(self._canvas)
        content_id = self._canvas.create_window((0, 0), window=content, anchor="nw")

        def _update_scrollregion(_):
            self._canvas.configure(scrollregion=self._canvas.bbox("all"))

        def _resize_content(event):
            self._canvas.itemconfigure(content_id, width=event.width)

        content.bind("<Configure>", _update_scrollregion)
        self._canvas.bind("<Configure>", _resize_content)

        content.bind("<Enter>", lambda _e: self._bind_mousewheel())
        content.bind("<Leave>", lambda _e: self._unbind_mousewheel())

        tk.Label(content, text="Settings", font=("Arial", 16, "bold")).pack(pady=(20, 10))
        tk.Label(content, text="Customize how the Noise-Aware Productivity Coach looks.",
                 font=("Arial", 10)).pack(pady=(0, 20))

        self.theme_var = tk.StringVar(value=controller.current_theme)

        theme_section = tk.Frame(content)
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

        mic_section = tk.Frame(content)
        mic_section.pack(pady=10, padx=20, fill="x")

        tk.Label(mic_section, text="Microphone Sensitivity", font=("Arial", 12, "bold")).pack(anchor="w")
        tk.Label(
            mic_section,
            text="Adjust how sensitive the microphone is when monitoring noise.",
            wraplength=400,
            justify="left",
        ).pack(anchor="w", pady=(0, 10))

        slider_frame = tk.Frame(mic_section)
        slider_frame.pack(fill="x", pady=(0, 5))

        minus_button = tk.Button(slider_frame, text="-", width=3,
                                 command=lambda: self.adjust_sensitivity(-1))
        minus_button.pack(side="left", padx=(0, 5))

        self.sensitivity_scale = tk.Scale(
            slider_frame,
            from_=0,
            to=100,
            orient="horizontal",
            variable=controller.mic_sensitivity,
            command=self.on_sensitivity_change,
            length=260,
        )
        self.sensitivity_scale.pack(side="left", fill="x", expand=True)

        plus_button = tk.Button(slider_frame, text="+", width=3,
                                command=lambda: self.adjust_sensitivity(1))
        plus_button.pack(side="left", padx=(5, 0))

        self.sensitivity_value_label = tk.Label(mic_section, font=("Arial", 10, "italic"))
        self.sensitivity_value_label.pack(anchor="w")

        self._update_sensitivity_display()

        alert_section = tk.Frame(content)
        alert_section.pack(pady=10, padx=20, fill="x")

        tk.Label(alert_section, text="Pomodoro Alerts", font=("Arial", 12, "bold")).pack(anchor="w")
        tk.Label(
            alert_section,
            text="Play a sound when a Pomodoro work session finishes.",
            wraplength=400,
            justify="left",
        ).pack(anchor="w", pady=(0, 10))

        self.alert_var = tk.BooleanVar(value=controller.pomodoro_alert_enabled)
        tk.Checkbutton(
            alert_section,
            text="Enable completion sound",
            variable=self.alert_var,
            command=self.toggle_alert,
        ).pack(anchor="w", pady=2)

        tk.Button(alert_section, text="Preview Sound", command=controller.play_pomodoro_alert).pack(anchor="w", pady=(5, 0))

        self.preview_label = tk.Label(
            content,
            text="Preview: Productive focus starts with the right lighting!",
            font=("Arial", 11, "italic"),
            wraplength=420,
            justify="center",
        )
        self.preview_label.pack(pady=25, padx=20)

        tk.Button(content, text="Back", command=lambda: controller.show_frame(MainMenu)).pack(pady=10)

    def change_theme(self):
        self.controller.set_theme(self.theme_var.get())

    def toggle_alert(self):
        self.controller.set_pomodoro_alert_enabled(self.alert_var.get())

    def on_show(self):
        self._update_sensitivity_display()

    def on_theme_applied(self, _colors):
        """Keep the selection synced with the controller state."""
        if self.theme_var.get() != self.controller.current_theme:
            self.theme_var.set(self.controller.current_theme)
        self._update_sensitivity_display()

    def on_sensitivity_change(self, value):
        self.controller.set_mic_sensitivity(value)
        self._update_sensitivity_display()

    def adjust_sensitivity(self, delta):
        self.controller.adjust_mic_sensitivity(delta)
        self._update_sensitivity_display()

    def _update_sensitivity_display(self):
        value = self.controller.mic_sensitivity.get()
        self.sensitivity_value_label.config(text=f"Current level: {value}")
        if int(self.sensitivity_scale.get()) != value:
            self.sensitivity_scale.set(value)

    def _bind_mousewheel(self):
        self._canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self._canvas.bind_all("<Button-4>", self._on_mousewheel)
        self._canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _unbind_mousewheel(self):
        self._canvas.unbind_all("<MouseWheel>")
        self._canvas.unbind_all("<Button-4>")
        self._canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event):
        if event.num == 4:
            delta = -1
        elif event.num == 5:
            delta = 1
        else:
            delta = -int(event.delta / 120)
        self._canvas.yview_scroll(delta, "units")


# === Detailed Report Page (Tabbed View: AI Reports + Analytics) ===
class DetailedReportPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        colors = controller.themes[controller.current_theme]
        self._chart_range_initialized = False

        tk.Label(self, text="📊 Detailed Productivity Insights", font=("Arial", 16, "bold"),
                 bg=colors["bg"], fg=colors["fg"]).pack(pady=10)

        tk.Button(self, text="⬅ Back to Reports",
                  command=lambda: controller.show_frame(ReportPage)).pack(pady=5)

        # --- Tab control ---
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=20, pady=10)

        # ---------------- TAB 1 : AI Report History ----------------
        tab_reports = tk.Frame(notebook, bg=colors["bg"])
        notebook.add(tab_reports, text="📄 AI Reports")

        # Date filter
        filter_frame = tk.Frame(tab_reports, bg=colors["bg"])
        filter_frame.pack(pady=10)
        tk.Label(filter_frame, text="Select Date:", bg=colors["bg"], fg=colors["fg"]).pack(side="left", padx=5)
        self.date_entry = DateEntry(filter_frame, width=12, background="darkblue",
                                    foreground="white", borderwidth=2, date_pattern="yyyy-mm-dd")
        self.date_entry.pack(side="left", padx=5)
        tk.Button(filter_frame, text="🔍 Load Reports", command=self.load_reports).pack(side="left", padx=8)

        # Scrollable report viewer
        text_frame = tk.Frame(tab_reports, bg=colors["bg"])
        text_frame.pack(fill="both", expand=True, padx=20, pady=10)
        self.text_area = tk.Text(text_frame, wrap="word", font=("Segoe UI", 10),
                                 bg=colors["entry_bg"], fg=colors["entry_fg"])
        self.text_area.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(text_frame, command=self.text_area.yview)
        scrollbar.pack(side="right", fill="y")
        self.text_area.config(yscrollcommand=scrollbar.set)

        # ---------------- TAB 2 : Visual Analytics ----------------
        tab_charts = tk.Frame(notebook, bg=colors["bg"])
        notebook.add(tab_charts, text="📈 Visual Analytics")

        # Date range selectors
        range_frame = tk.Frame(tab_charts, bg=colors["bg"])
        range_frame.pack(pady=10)
        tk.Label(range_frame, text="Start:", bg=colors["bg"], fg=colors["fg"]).grid(row=0, column=0, padx=5)
        self.start_date = DateEntry(range_frame, width=12, background="darkblue",
                                    foreground="white", borderwidth=2, date_pattern="yyyy-mm-dd")
        self.start_date.grid(row=0, column=1, padx=5)
        tk.Label(range_frame, text="End:", bg=colors["bg"], fg=colors["fg"]).grid(row=0, column=2, padx=5)
        self.end_date = DateEntry(range_frame, width=12, background="darkblue",
                                  foreground="white", borderwidth=2, date_pattern="yyyy-mm-dd")
        self.end_date.grid(row=0, column=3, padx=5)
        tk.Button(range_frame, text="📅 Apply Filter", command=self.load_charts).grid(row=0, column=4, padx=10)

        self.chart_frame = tk.Frame(tab_charts, bg=colors["bg"])
        self.chart_frame.pack(fill="both", expand=True, padx=20, pady=10)

    def on_show(self):
        """Auto-refresh analytics when the page is displayed."""
        self.load_charts()

    # ---------- TAB 1 : Load AI Reports ----------
    def load_reports(self):
        selected_date = self.date_entry.get_date().strftime("%Y-%m-%d")
        self.text_area.delete(1.0, tk.END)
        if not os.path.exists(DB_PATH):
            self.text_area.insert(tk.END, f"⚠️ Database not found at:\n{DB_PATH}")
            return

        conn = None
        try:
            conn = get_connection()
            c = conn.cursor()
            c.execute("""
                SELECT date, summary, ai_report, created_at
                FROM ReportHistory
                WHERE date = ?
                ORDER BY created_at DESC
            """, (selected_date,))
            rows = c.fetchall()
        except Exception as e:
            self.text_area.insert(tk.END, f"⚠️ Error loading reports:\n{e}")
            return
        finally:
            if conn is not None:
                conn.close()

        if not rows:
            self.text_area.insert(tk.END, f"No reports found for {selected_date}.\n")
            return

        for row in rows:
            date_str, summary, ai_report, created_at = row
            self.text_area.insert(tk.END, f"📅 Date: {date_str}\n🕒 Generated: {created_at}\n\n")
            self.text_area.insert(tk.END, f"📋 Summary:\n{summary}\n\n")
            self.text_area.insert(tk.END, f"💬 AI Report:\n{ai_report}\n")
            self.text_area.insert(tk.END, "-"*90 + "\n\n")

    # ---------- Chart Helpers ----------
    def _plot_noise_trend(self, ax, df_logs, df_sessions):
        if df_logs.empty:
            ax.text(0.5, 0.5, "No readings available", ha="center", va="center", fontsize=11)
            ax.set_axis_off()
            return

        logs = df_logs.sort_values("timestamp").set_index("timestamp")
        window = "15min"
        category_levels = (
            logs.groupby([pd.Grouper(freq=window), "noise_category"])["db_level"]
            .mean()
            .unstack(fill_value=0)
        )
        category_order = ["Quiet", "Moderate", "Noisy"]
        category_levels = category_levels.reindex(columns=category_order, fill_value=0)
        x = category_levels.index

        if x.empty:
            ax.text(0.5, 0.5, "Not enough data for trend", ha="center", va="center", fontsize=11)
            ax.set_axis_off()
            return

        palette = {"Quiet": "#57cc99", "Moderate": "#ffd166", "Noisy": "#ef476f"}
        stack_values = [category_levels[col].to_numpy() for col in category_order]
        ax.stackplot(
            x,
            stack_values,
            labels=[f"{col} avg dB" for col in category_order],
            colors=[palette[col] for col in category_order],
            alpha=0.55,
        )

        legend_handles = [
            mpatches.Patch(color=palette[col], alpha=0.55, label=f"{col} avg dB")
            for col in category_order
        ]

        avg_db = logs["db_level"].resample(window).mean().reindex(x)
        avg_db = avg_db.interpolate("time")
        ax.plot(x, avg_db, color="#26547C", linewidth=2.2, label="Overall avg dB")
        legend_handles.append(mpatches.Patch(color="#26547C", label="Overall avg dB"))

        if not df_sessions.empty and "start_time" in df_sessions:
            completed_block = False
            other_block = False
            for _, session in df_sessions.iterrows():
                start_time = session.get("start_time")
                if pd.isna(start_time):
                    continue
                end_time = session.get("end_time")
                if pd.isna(end_time):
                    end_time = start_time + pd.Timedelta(minutes=25)
                if str(session.get("status", "")).lower() == "completed":
                    color = "#118ab2"
                    completed_block = True
                else:
                    color = "#073b4c"
                    other_block = True
                ax.axvspan(start_time, end_time, color=color, alpha=0.12)

            if completed_block:
                legend_handles.append(mpatches.Patch(color="#118ab2", alpha=0.2, label="Completed focus block"))
            if other_block:
                legend_handles.append(mpatches.Patch(color="#073b4c", alpha=0.15, label="Active/Interrupted block"))

        peak_time = avg_db.idxmax()
        if pd.notna(peak_time):
            peak_value = avg_db.loc[peak_time]
            ax.annotate(
                f"Peak {peak_value:.1f} dB",
                xy=(peak_time, peak_value),
                xytext=(10, 20),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="#26547C"),
                fontsize=9,
                color="#26547C",
            )

        ax.set_title("Stacked Noise Profile & Focus Windows", fontsize=12, fontweight="bold")
        ax.set_ylabel("Average dB (15 min bins)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, which="major", axis="y", linestyle="--", alpha=0.3)
        if legend_handles:
            ax.legend(handles=legend_handles, loc="upper left", ncol=2, fontsize=9)

    def _plot_interruption_heatmap(self, ax, df_logs):
        if df_logs.empty:
            ax.text(0.5, 0.5, "No noise activity to map", ha="center", va="center", fontsize=11)
            ax.set_axis_off()
            return

        logs = df_logs.copy()
        logs["hour"] = logs["timestamp"].dt.hour
        logs["weekday"] = logs["timestamp"].dt.day_name()
        logs["noise_category"] = logs["noise_category"].fillna("Unknown")

        noisy_mask = logs["noise_category"].str.lower() == "noisy"
        focus_events = logs[noisy_mask]
        intensity_label = "Noisy event count"

        if focus_events.empty and "db_level" in logs:
            valid_db = logs["db_level"].dropna()
            if not valid_db.empty:
                threshold = np.percentile(valid_db, 80)
                focus_events = logs[logs["db_level"] >= threshold]
                intensity_label = "High dB event count"

        if focus_events.empty:
            ax.text(0.5, 0.5, "No intense noise patterns detected", ha="center", va="center", fontsize=11)
            ax.set_axis_off()
            return

        heatmap = (
            focus_events.groupby(["weekday", "hour"])
            .size()
            .unstack(fill_value=0)
            .reindex([
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ],
                     fill_value=0)
        )

        im = ax.imshow(heatmap.values, aspect="auto", cmap="YlOrRd")
        ax.set_title("Interruption Frequency Heatmap", fontsize=12, fontweight="bold")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Weekday")
        ax.set_xticks(range(len(heatmap.columns)))
        ax.set_xticklabels(heatmap.columns, rotation=0)
        ax.set_yticks(range(len(heatmap.index)))
        ax.set_yticklabels(heatmap.index)

        max_idx = np.unravel_index(np.argmax(heatmap.values), heatmap.values.shape)
        max_value = heatmap.values[max_idx]
        if max_value > 0:
            y, x = max_idx
            ax.annotate(
                f"Peak: {int(max_value)}",
                xy=(x, y),
                xytext=(5, -12),
                textcoords="offset points",
                color="#b83227",
                fontsize=9,
                fontweight="bold",
            )

        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(intensity_label)

    def _plot_noise_completion_strip(self, ax, df_logs, df_sessions):
        if df_sessions.empty:
            ax.text(0.5, 0.5, "No Pomodoro sessions yet", ha="center", va="center", fontsize=11)
            ax.set_axis_off()
            return

        if "session_id" not in df_logs:
            ax.text(0.5, 0.5, "Session linkage missing in logs", ha="center", va="center", fontsize=11)
            ax.set_axis_off()
            return

        session_logs = df_logs.dropna(subset=["session_id", "db_level"]).copy()
        session_logs["session_id"] = pd.to_numeric(session_logs["session_id"], errors="coerce")
        session_logs.dropna(subset=["session_id"], inplace=True)
        if session_logs.empty:
            ax.text(0.5, 0.5, "No noise readings captured during sessions", ha="center", va="center", fontsize=11)
            ax.set_axis_off()
            return

        session_logs["session_id"] = session_logs["session_id"].astype(int)
        df_sessions = df_sessions.dropna(subset=["session_id"]).copy()
        df_sessions["session_id"] = pd.to_numeric(df_sessions["session_id"], errors="coerce")
        df_sessions.dropna(subset=["session_id"], inplace=True)
        if df_sessions.empty:
            ax.text(0.5, 0.5, "Session metadata unavailable", ha="center", va="center", fontsize=11)
            ax.set_axis_off()
            return

        df_sessions["session_id"] = df_sessions["session_id"].astype(int)
        avg_db_by_session = (
            session_logs.groupby("session_id")["db_level"].mean().reset_index(name="avg_db")
        )
        merged = avg_db_by_session.merge(
            df_sessions[["session_id", "status"]], on="session_id", how="inner"
        )

        if merged.empty:
            ax.text(0.5, 0.5, "No completed sessions with noise data", ha="center", va="center", fontsize=11)
            ax.set_axis_off()
            return

        merged["status"] = merged["status"].fillna("Unknown")
        statuses = merged["status"].unique()
        colors = {
            "Completed": "#06d6a0",
            "Running": "#118ab2",
            "Interrupted": "#ffd166",
            "Cancelled": "#ef476f",
            "Abandoned": "#ef476f",
            "Unknown": "#8d99ae",
        }

        legend_handles = []
        for idx, status in enumerate(statuses):
            values = merged.loc[merged["status"] == status, "avg_db"].to_numpy()
            x_positions = np.full_like(values, idx, dtype=float)
            jitter = np.random.uniform(-0.18, 0.18, size=len(values))
            color = colors.get(status, "#577590")
            ax.scatter(
                x_positions + jitter,
                values,
                color=color,
                alpha=0.75,
                edgecolor="#1f1f1f",
                linewidth=0.4,
                label=status,
            )
            legend_handles.append(mpatches.Patch(color=color, label=status))

        overall_mean = merged["avg_db"].mean()
        ax.axhline(overall_mean, color="#073b4c", linestyle="--", linewidth=1, alpha=0.6)
        ax.annotate(
            f"Mean session dB: {overall_mean:.1f}",
            xy=(0.02, overall_mean),
            xycoords=("axes fraction", "data"),
            textcoords="offset points",
            xytext=(5, -12),
            color="#073b4c",
            fontsize=9,
        )

        ax.set_title("Noise vs. Session Outcome", fontsize=12, fontweight="bold")
        ax.set_ylabel("Average dB per Session")
        ax.set_xticks(range(len(statuses)))
        ax.set_xticklabels(statuses, rotation=20)
        ax.grid(True, axis="y", linestyle=":", alpha=0.3)
        ax.legend(handles=legend_handles, title="Session Status", loc="upper right")

    # ---------- TAB 2 : Load Visual Charts ----------
    def load_charts(self):
        for widget in self.chart_frame.winfo_children():
            widget.destroy()

        conn = None
        try:
            conn = get_connection()
            df_logs = pd.read_sql_query("SELECT * FROM NoiseLog", conn)
            df_sessions = pd.read_sql_query("SELECT * FROM PomodoroSession", conn)
        except Exception as e:
            tk.Label(self.chart_frame, text=f"⚠️ Failed to load data: {e}",
                     font=("Arial", 12)).pack(pady=30)
            return
        finally:
            if conn is not None:
                conn.close()

        if df_logs.empty:
            tk.Label(self.chart_frame, text="No noise data available yet.",
                     font=("Arial", 12)).pack(pady=30)
            return

        # === Robust timestamp parsing with UTC normalization ===
        local_tz = datetime.now().astimezone().tzinfo
        df_logs["timestamp"] = pd.to_datetime(df_logs["timestamp"], errors="coerce", utc=True)
        df_logs = df_logs.dropna(subset=["timestamp"])
        df_logs["timestamp"] = (
            df_logs["timestamp"].dt.tz_convert(local_tz).dt.tz_localize(None)
        )  # make tz-naive in local time for plotting

        if df_logs.empty:
            tk.Label(self.chart_frame, text="No valid timestamps available for plotting.",
                     font=("Arial", 12)).pack(pady=30)
            return

        # Auto-adjust the visible range the first time charts are loaded
        min_ts = df_logs["timestamp"].min()
        max_ts = df_logs["timestamp"].max()
        min_date = min_ts.date()
        max_date = max_ts.date()

        if not hasattr(self, "_chart_range_initialized") or not self._chart_range_initialized:
            self.start_date.set_date(min_date)
            self.end_date.set_date(max_date)
            self._chart_range_initialized = True

        # Ensure start <= end for filtering while keeping the user's selection intact
        start_date_value = self.start_date.get_date()
        end_date_value = self.end_date.get_date()
        if start_date_value <= end_date_value:
            range_start = start_date_value
            range_end = end_date_value
        else:
            range_start = end_date_value
            range_end = start_date_value

        start = datetime.combine(range_start, datetime.min.time())
        end = datetime.combine(range_end, datetime.min.time()) + pd.Timedelta(days=1)
        df_logs = df_logs[(df_logs["timestamp"] >= start) & (df_logs["timestamp"] < end)]

        if not df_sessions.empty:
            df_sessions["start_time"] = pd.to_datetime(
                df_sessions["start_time"], errors="coerce", utc=True
            )
            df_sessions["end_time"] = pd.to_datetime(
                df_sessions["end_time"], errors="coerce", utc=True
            )
            df_sessions = df_sessions.dropna(subset=["start_time"])
            df_sessions["start_time"] = (
                df_sessions["start_time"].dt.tz_convert(local_tz).dt.tz_localize(None)
            )
            df_sessions["end_time"] = (
                df_sessions["end_time"].dt.tz_convert(local_tz).dt.tz_localize(None)
            )
            df_sessions = df_sessions[(df_sessions["start_time"] >= start) & (df_sessions["start_time"] < end)]
            df_sessions.sort_values("start_time", inplace=True)

        if df_logs.empty:
            tk.Label(self.chart_frame, text="No records for this date range.",
                     font=("Arial", 12)).pack(pady=30)
            return

        df_logs["db_level"] = pd.to_numeric(df_logs["db_level"], errors="coerce")
        if "confidence" in df_logs:
            df_logs["confidence"] = pd.to_numeric(df_logs["confidence"], errors="coerce")
        df_logs.dropna(subset=["db_level"], inplace=True)

        if df_logs.empty:
            tk.Label(self.chart_frame, text="No valid dB readings for this date range.",
                     font=("Arial", 12)).pack(pady=30)
            return

        fig = plt.figure(figsize=(12, 7))
        gs = fig.add_gridspec(2, 2, height_ratios=[2.4, 1.8])
        ax_trend = fig.add_subplot(gs[0, :])
        ax_heat = fig.add_subplot(gs[1, 0])
        ax_strip = fig.add_subplot(gs[1, 1])

        self._plot_noise_trend(ax_trend, df_logs, df_sessions)
        self._plot_interruption_heatmap(ax_heat, df_logs)
        self._plot_noise_completion_strip(ax_strip, df_logs, df_sessions)

        display_end = (end - pd.Timedelta(seconds=1)).date()
        fig.suptitle(
            f"Noise & Focus Analytics ({start.date()} → {display_end})",
            fontsize=14,
            fontweight="bold",
        )
        fig.subplots_adjust(top=0.9, hspace=0.45, wspace=0.28)

        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)


# === Run App ===
if __name__ == "__main__":
    app = NoiseAwareApp()
    app.mainloop()
