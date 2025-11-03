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
            if preferred_name:
                text = (
                    f"Today's noise environment was balanced overall for you, {preferred_name}.\n"
                    "• Protect your quiet hours by keeping notifications off during focus periods.\n"
                    "• Use brief breaks to refresh attention and maintain consistency.\n"
                    "• Reflect on which environments feel most productive and repeat them.\n"
                    "• Keep reinforcing your routine — progress compounds with each session."
                )
            else:
                text = (
                    "Today's noise environment was balanced overall.\n"
                    "• Protect your quiet hours by keeping notifications off during focus periods.\n"
                    "• Use brief breaks to refresh attention and maintain consistency.\n"
                    "• Reflect on which environments feel most productive and repeat them.\n"
                    "• Keep reinforcing your routine — progress compounds with each session."
                )

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
