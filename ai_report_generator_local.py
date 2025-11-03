"""
Optimized TinyLlama local report generator (Subprocess-Safe)
------------------------------------------------------------
• Fast & lightweight (<= 4 GB RAM during generation)
• Runs model in an isolated subprocess — full memory release after each report
• Compatible with GUI auto-progress callbacks
• Generates concise, motivational coaching summaries
"""

import os
import gc
import re
import multiprocessing
from typing import Optional, Callable

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)


# =========================
# Environment + Config
# =========================
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Prevent TensorFlow GPU reserve
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

LOCAL_MODEL_PATH = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

DEFAULT_GEN_CFG = GenerationConfig(
    max_new_tokens=900,
    temperature=0.8,
    top_p=0.95,
    repetition_penalty=1.05,
    presence_penalty=0.2,
    do_sample=True,
    return_full_text=False
)

STOP_STRINGS = ["\nUser:", "\nAssistant:", "\nCoach:", "\nSystem:"]


# =========================
# Utilities
# =========================
def _device_hint():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def _apply_stop_strings(text: str) -> str:
    for s in STOP_STRINGS:
        if s in text:
            text = text.split(s)[0]
    return text.strip()


def _clean_spaces(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _parse_summary_metrics(summary_text: str) -> dict:
    """Extract useful numbers and descriptors from the raw summary text."""
    metrics = {}
    summary = summary_text or ""

    avg_match = re.search(r"Average[^0-9]*([0-9]+(?:\.[0-9]+)?)\s*dB", summary, re.IGNORECASE)
    if avg_match:
        metrics["average_db"] = avg_match.group(1)

    avg_label_match = re.search(r"Average[^()]*\(([^)]+)\)", summary, re.IGNORECASE)
    if avg_label_match:
        metrics["average_label"] = avg_label_match.group(1)

    most_common_match = re.search(r"Most common:\s*([\w\s-]+)", summary, re.IGNORECASE)
    if most_common_match:
        metrics["most_common_noise"] = most_common_match.group(1).strip()

    quiet_match = re.search(r"Quiet(?:est)?(?: window| period)?:\s*([^\n]+)", summary, re.IGNORECASE)
    if quiet_match:
        metrics["quiet_window"] = quiet_match.group(1).strip()

    loud_match = re.search(r"(?:Peak|Loudest)(?: window| period| level)?:\s*([^\n]+)", summary, re.IGNORECASE)
    if loud_match:
        metrics["loud_window"] = loud_match.group(1).strip()

    pom_match = re.search(r"Pomodoros:\s*([0-9]+)(?:\s*\(([^)]+)\))?", summary, re.IGNORECASE)
    if pom_match:
        metrics["pomodoros"] = pom_match.group(1)
        if pom_match.group(2):
            metrics["pomodoro_detail"] = pom_match.group(2).strip()

    focus_min_match = re.search(r"([0-9]+)\s*min\s*focused", summary, re.IGNORECASE)
    if focus_min_match:
        metrics["focused_minutes"] = focus_min_match.group(1)

    return metrics


def _build_fallback_response(summary_text: str, preferred_name: str) -> str:
    metrics = _parse_summary_metrics(summary_text)

    user_suffix = f" for you, {preferred_name}" if preferred_name else ""

    if metrics.get("average_db"):
        label_part = f" ({metrics['average_label']})" if metrics.get("average_label") else ""
        summary_line = (
            f"Today's noise environment hovered around {metrics['average_db']} dB{label_part}{user_suffix}, "
            "offering a stable backdrop for focus."
        )
    else:
        summary_line = f"Today's noise environment was balanced overall{user_suffix}."

    bullets = []

    if metrics.get("most_common_noise"):
        bullets.append(
            f"• Dominant noise source: {metrics['most_common_noise']}. Plan deep-focus work when this source is naturally lower, and batch collaborative tasks when it's active."
        )
    else:
        bullets.append(
            "• Protect your quiet hours by silencing notifications and closing non-essential tabs during priority work blocks."
        )

    if metrics.get("quiet_window"):
        bullets.append(
            f"• Leverage the quieter window ({metrics['quiet_window']}) for high-cognition work and schedule reminders to start your toughest task then."
        )
    else:
        bullets.append(
            "• Identify a repeatable quiet window in your day and guard it with calendar blocks for deep work."
        )

    if metrics.get("loud_window"):
        bullets.append(
            f"• During louder periods ({metrics['loud_window']}), shift to lighter tasks, collaborative check-ins, or use noise-masking audio to stay composed."
        )
    else:
        bullets.append(
            "• Keep a ready playlist or noise buffer for surprise spikes so interruptions don't derail your momentum."
        )

    if metrics.get("pomodoros") or metrics.get("focused_minutes"):
        focus_detail = []
        if metrics.get("pomodoros"):
            focus_detail.append(f"{metrics['pomodoros']} Pomodoros")
        if metrics.get("pomodoro_detail"):
            focus_detail.append(metrics['pomodoro_detail'])
        elif metrics.get("focused_minutes"):
            focus_detail.append(f"{metrics['focused_minutes']} min focused work")

        focus_text = " and ".join(focus_detail)
        bullets.append(
            f"• Sustain that {focus_text}; pair each session with a brief reset so the routine stays energizing."
        )
    else:
        bullets.append(
            "• Use brief breaks every 50–60 minutes to refresh attention and reinforce consistent performance."
        )

    bullets.append(
        "• Track which environments feel most productive, note what made them work, and keep reinforcing those routines so progress compounds each session."
    )

    return "\n".join([summary_line, *bullets])


def _unload_model(model=None):
    """Internal memory cleanup helper."""
    try:
        print("🧹 Cleaning TinyLlama model from memory...")
        if model is not None:
            model.to("cpu")
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("✅ TinyLlama model memory released.")
    except Exception as e:
        print(f"⚠️ Cleanup error: {e}")


# =========================
# Internal Worker (runs in subprocess)
# =========================
def _generate_in_subprocess(summary_text: str, preferred_name: str, queue: multiprocessing.Queue):
    """Worker that loads TinyLlama, generates text, then exits (freeing memory)."""
    preferred_name = (preferred_name or "").strip()
    model = None
    tokenizer = None
    try:
        dev, dtype = _device_hint()
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_PATH,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto" if dev != "cpu" else None,
        )
        model.eval()

        system_prompt = (
            "You are a local noise-aware productivity coach. "
            "Analyze focus and noise data, highlight key patterns, and provide concise, motivational advice. "
            "Avoid greetings or sign-offs. Keep tone professional and natural. "
            "Avoid generating responses like your own thoughts — respond as if directly advising the user 1-to-1."
        )
        if preferred_name:
            system_prompt += f" Always address the user by the name {preferred_name}."
        content_guide = (
            "Structure your answer as:\n"
            "• 1 summary line about today’s environment\n"
            "• 3–5 bullet points covering noise patterns, improvements, and focus actions.\n"
            "• Include actionable productivity suggestions.\n"
            "• Give encouragement based on trends in the noise data."
        )

        prompt = (
            f"{system_prompt}\n\n{content_guide}\n\n"
            f"Today's Summary:\n{summary_text.strip()}\n\n"
            "Now write the complete response in the requested structure. "
            "Make it motivational and actionable with focus improvement tips based on the data. "
            "Avoid describing yourself; speak directly to the user.\n"
            "Respond directly:\n"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(dev)

        output_ids = model.generate(
            **inputs,
            max_new_tokens=DEFAULT_GEN_CFG.max_new_tokens,
            temperature=DEFAULT_GEN_CFG.temperature,
            top_p=DEFAULT_GEN_CFG.top_p,
            repetition_penalty=DEFAULT_GEN_CFG.repetition_penalty,
            do_sample=DEFAULT_GEN_CFG.do_sample,
        )

        generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        text = _apply_stop_strings(text)
        text = _clean_spaces(text)

        if len(text) < 50 or "•" not in text:
            text = _build_fallback_response(summary_text, preferred_name)

        if text.count("•") < 3:
            text += (
                "\n• Plan one dedicated silent Pomodoro to rebuild focus.\n"
                "• Review noise trends weekly to optimize your environment.\n"
                "• Reward yourself after sustained focus sessions to reinforce momentum."
            )

        queue.put(text)

    except Exception as e:
        queue.put(f"⚠️ Report generation error: {e}")
    finally:
        _unload_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        torch.cuda.empty_cache()


# =========================
# Public API
# =========================
def generate_ai_report(summary_text: str,
                       preferred_name: str = "",
                       progress_callback: Optional[Callable[[int, str], None]] = None) -> str:
    """
    Generates AI report safely in a subprocess to ensure full memory release.
    The GUI can call this normally — no changes needed.
    """
    if progress_callback:
        progress_callback(10, "Spawning TinyLlama subprocess...")

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_generate_in_subprocess,
        args=(summary_text, preferred_name, queue)
    )
    process.start()

    if progress_callback:
        progress_callback(50, "TinyLlama generating report...")

    process.join()
    result = queue.get() if not queue.empty() else "⚠️ No report generated."
    process.close()

    if progress_callback:
        progress_callback(100, "Done.")
    print("🧹 Subprocess exited — memory fully released.")

    return result


# =========================
# Manual test
# =========================
if __name__ == "__main__":
    sample = (
        "Date: Oct 13, 2025\n"
        "Total logs: 520\n"
        "Average: 46.7 dB SPL (Moderate)\n"
        "Most common: Speech\n"
        "Pomodoros: 3 (75 min focused work)"
    )
    print(generate_ai_report(sample))
