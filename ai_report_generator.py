# ai_report_generator.py
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# === Handle older/newer error classes gracefully ===
try:
    from huggingface_hub import InferenceEndpointError as HFError
except ImportError:
    try:
        from huggingface_hub import InferenceAPIError as HFError
    except ImportError:
        class HFError(Exception):
            pass

# === Load environment ===
load_dotenv()

_ENV_VAR_CANDIDATES = (
    "API_KEY",
    "HUGGINGFACEHUB_API_TOKEN",
    "HUGGING_FACE_API_KEY",
)

def _load_hf_api_key():
    for var_name in _ENV_VAR_CANDIDATES:
        value = os.getenv(var_name)
        if value:
            return value.strip()
    return None

HF_API_KEY = _load_hf_api_key()

# === Conversational models (chat-based) ===
PRIMARY_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
FALLBACK_MODELS = [
    "HuggingFaceH4/zephyr-7b-beta",
    "tiiuae/falcon-7b-instruct",
    "google/gemma-2b-it",
]

# === Helper ===
def _chat_with_model(model_name: str, prompt: str, client: InferenceClient):
    """Try generating a conversational-style reply."""
    try:
        print(f"🧠 Trying chat model: {model_name}")
        response = client.chat_completion(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a friendly productivity coach."},
                {
                    "role": "user",
                    "content": (
                        "Analyze the following daily noise and focus summary, "
                        "and write a concise motivational report (4–6 sentences):\n\n"
                        f"{prompt}\n\n"
                        "Use a warm, supportive tone and end with one actionable tip."
                    ),
                },
            ],
            max_tokens=300,
            temperature=0.7,
        )
        if response and response.choices and response.choices[0].message:
            return response.choices[0].message["content"].strip()
    except HFError as e:
        print(f"⚠️ Inference error for {model_name}: {e}")
    except Exception as e:
        print(f"⚠️ Unexpected error for {model_name}: {e}")
    return None

# === Main ===
def generate_ai_report(summary_text: str) -> str:
    """Generate a motivational daily productivity report via chat-based inference."""
    if not HF_API_KEY:
        env_hint = ", ".join(_ENV_VAR_CANDIDATES)
        return f"❌ Missing Hugging Face API key. Set one of: {env_hint}"

    client = InferenceClient(token=HF_API_KEY)

    for model in [PRIMARY_MODEL] + FALLBACK_MODELS:
        result = _chat_with_model(model, summary_text, client)
        if result:
            return result

    return "⚠️ All models failed. Try regenerating your token or check internet access."

# === Quick local test ===
if __name__ == "__main__":
    demo_summary = """
    Date: Oct 10, 2025
    Average sound level: 63.5 dB SPL
    Noise mix: Quiet 30% | Moderate 50% | Noisy 20%
    Completed Pomodoros: 3 (≈ 75 minutes of focused work)
    """
    print(generate_ai_report(demo_summary))
