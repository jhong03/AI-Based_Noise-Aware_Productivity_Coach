# AI Report Generator (Explanatory Summary + Insights)
# ----------------------------------------------------
# Features:
# - Summary line expanded into short explanatory paragraph
# - Bullet points provide actionable insights only
# - High-variation output with style modes and random seeds
# - Memory-safe subprocess architecture
# - Fallback ensures longer, structured reports

import os
import gc
import re
import multiprocessing
import random
from typing import Optional, Callable

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# =========================
# Environment + Model Path
# =========================
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

LOCAL_MODEL_PATH = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

DEFAULT_GEN_CFG = GenerationConfig(
    max_new_tokens=1200,
    temperature=1.1,
    top_p=0.92,
    top_k=40,
    repetition_penalty=1.08,
    do_sample=True,
    return_full_text=False
)

# =========================
# Utilities
# =========================

def _device_hint():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32

def _clean_spaces(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def _extract_first_int(text: str):
    if not text:
        return None
    m = re.search(r"([0-9]+)", text)
    return int(m.group(1)) if m else None

def _parse_summary_metrics(summary_text: str) -> dict:
    metrics = {}
    s = summary_text or ""

    avg = re.search(r"Average[^0-9]*([0-9]+(?:\.[0-9]+)?)", s)
    if avg:
        metrics["average_db"] = avg.group(1)

    label = re.search(r"Average[^()]*\(([^)]+)\)", s)
    if label:
        metrics["average_label"] = label.group(1)

    mc = re.search(r"Most common:\s*([^\n]+)", s)
    if mc:
        metrics["most_common_noise"] = mc.group(1).strip()

    q = re.search(r"Quiet(?:est)?[^:]*:\s*([^\n]+)", s)
    if q:
        metrics["quiet_window"] = q.group(1).strip()

    l = re.search(r"(?:Peak|Loudest)[^:]*:\s*([^\n]+)", s)
    if l:
        metrics["loud_window"] = l.group(1).strip()

    pom = re.search(r"Pomodoros:\s*([0-9]+)(?:\s*\(([^)]+)\))?", s)
    if pom:
        metrics["pomodoros"] = pom.group(1)
        if pom.group(2):
            metrics["pomodoro_detail"] = pom.group(2)

    fm = re.search(r"([0-9]+)\s*min\s*focused", s)
    if fm:
        metrics["focused_minutes"] = fm.group(1)

    return metrics

def _soft_fallback(summary_text: str, preferred_name: str) -> str:
    metrics = _parse_summary_metrics(summary_text)
    name_suffix = f", {preferred_name}" if preferred_name else ""

    # Expanded explanatory summary
    summary_line = (
        f"Today's environment showed steady patterns{name_suffix}. "
        "Overall, the sound levels were moderate with identifiable quiet periods and occasional peaks. "
        "These patterns provide useful cues to organize focus sessions effectively."
    )

    bullets = []

    # Noise-based bullet
    noise_options = [
        "Anchor difficult tasks in naturally stable sound windows.",
        "Notice how recurring sounds influence your energy; schedule high-focus work accordingly.",
        "Use predictable quiet periods for tasks requiring concentration."
    ]
    bullets.append("• " + random.choice(noise_options))

    # Focus-based bullet
    focus_options = [
        "Use short resets between tasks to clear mental residue.",
        "Start the day with one strong focus block to build momentum.",
        "Track your most productive intervals and replicate them tomorrow."
    ]
    bullets.append("• " + random.choice(focus_options))

    # Improvement action
    improvement_options = [
        "Shift admin work into high-noise moments to preserve focus energy.",
        "Protect at least one quiet window with a schedule block.",
        "Experiment with light sound-masking during louder periods."
    ]
    bullets.append("• " + random.choice(improvement_options))

    # Additional guidance
    bullets.append("• Extend one focus block into a longer streak when energy feels stable.")
    bullets.append("• Note one behaviour that helped today and intentionally repeat it tomorrow.")

    return "\n".join([summary_line, *bullets])

# =========================
# Internal Subprocess Worker
# =========================

def _generate_in_subprocess(summary_text: str, preferred_name: str, queue: multiprocessing.Queue):
    model = None
    tokenizer = None

    try:
        dev, dtype = _device_hint()

        style_mode = random.choice([
            "Analytical", "Motivational", "Technical", "Friendly", "Minimalist"
        ])

        style_seed = random.choice([
            "Focus on how micro-shifts influenced your clarity today.",
            "Notice the emotional tone of the environment.",
            "Momentum built during your stronger intervals.",
            "Your pacing created natural focus arcs.",
            "Small cues guided your most productive windows.",
        ])

        random_noise_token = os.urandom(4).hex()

        # Load model + tokenizer
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_PATH,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto" if dev != "cpu" else None,
        )
        model.eval()

        # =========================
        # FINAL SYSTEM PROMPT
        # =========================
        system_prompt = (
            "You are a noise-aware productivity coach. Your task is to produce a clean, explanatory daily insight report. "
            "Start with a short paragraph summarizing today's noise patterns and their impact on focus. "
            "Then provide 3–5 bullet points of actionable guidance or insights. "
            "Use the preferred_name only once in the summary line if provided. "
            "Avoid greetings, sign-offs, or any mention of being an AI. "
            "Keep phrasing varied, natural, and human-like."
        )

        content_guide = (
            f"Preferred name (use once): {preferred_name if preferred_name else 'None'}.\n"
            f"Style mode: {style_mode}.\n"
            f"Subtle stylistic idea: '{style_seed}'.\n"
            "Incorporate any detectable environmental patterns (quiet windows, peaks, averages, common sources, etc.). "
            "Keep bullets concise and distinct. Avoid repeating the summary content."
        )

        prompt = (
            f"{system_prompt}\n\n"
            f"{content_guide}\n\n"
            f"Today's Summary:\n{summary_text}\n\n"
            "Generate the report now."
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(dev)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=DEFAULT_GEN_CFG.max_new_tokens,
            temperature=DEFAULT_GEN_CFG.temperature,
            top_p=DEFAULT_GEN_CFG.top_p,
            top_k=DEFAULT_GEN_CFG.top_k,
            repetition_penalty=DEFAULT_GEN_CFG.repetition_penalty,
            do_sample=True,
        )

        generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        text = _clean_spaces(text)

        # =========================
        # Fallback if generation too short
        # =========================
        if len(text) < 150:
            text = _soft_fallback(summary_text, preferred_name)

        if text.count("•") < 3:
            text += (
                "\n• Use a structured focus cycle to build consistency."
                "\n• Add one small environmental adjustment tomorrow (lighting, desk setup, etc.)."
            )

        queue.put(text)

    except Exception as e:
        queue.put(f"⚠️ Report generation error: {e}")

    finally:
        if model is not None:
            model.to("cpu")
            del model
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

    if progress_callback:
        progress_callback(10, "Starting TinyLlama subprocess...")

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_generate_in_subprocess,
        args=(summary_text, preferred_name, queue)
    )
    process.start()

    if progress_callback:
        progress_callback(50, "Generating report...")

    process.join()
    result = queue.get() if not queue.empty() else "⚠️ No report generated."
    process.close()

    if progress_callback:
        progress_callback(100, "Done.")

    return result

# =========================
# Manual Test
# =========================
if __name__ == "__main__":
    sample = (
        "Date: Oct 13, 2025\n"
        "Total logs: 520\n"
        "Average: 46.7 dB SPL (Moderate)\n"
        "Most common: Speech\n"
        "Pomodoros: 3 (75 min focused work)"
    )

    print(generate_ai_report(sample, preferred_name="Jing Han"))
