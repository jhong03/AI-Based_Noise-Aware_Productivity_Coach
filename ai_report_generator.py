# ai_report_generator.py
import os
import requests
import json

# ======================
# CONFIGURATION
# ======================
# These are the canonical Hugging Face environment variable names. If a deployment
# needs a custom alias, append it to this tuple instead of replacing the existing
# entries so the defaults continue to work out of the box.
_ENV_VAR_CANDIDATES = (
    "HF_API_KEY",
    "HUGGINGFACEHUB_API_TOKEN",
    "HUGGING_FACE_API_KEY",
)


def _load_hf_api_key():
    """Return the first available Hugging Face API key from known environment variables."""
    for var_name in _ENV_VAR_CANDIDATES:
        value = os.getenv(var_name)
        if value:
            return value.strip()
    return None


HF_API_KEY = _load_hf_api_key()
API_URL = "https://api-inference.huggingface.co/models/google/gemma-2b"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"} if HF_API_KEY else {}


def generate_ai_report(summary_text: str) -> str:
    """
    Generate a human-like productivity report based on summarized noise data.
    Uses Hugging Face 'google/gemma-2b' (free public model).
    """
    try:
        if not HF_API_KEY:
            env_hint = ", ".join(_ENV_VAR_CANDIDATES)
            return (
                "❌ Missing Hugging Face API key. Set one of the following environment "
                f"variables and try again: {env_hint}."
            )

        # Compose prompt
        prompt = (
            "You are a friendly productivity coach. "
            "Analyze the following daily noise and focus summary, "
            "and write a short, human-like report with motivational suggestions:\n\n"
            f"{summary_text}\n\n"
            "The report should be in a conversational tone, around 4–6 sentences."
        )

        payload = {"inputs": prompt, "parameters": {"max_new_tokens": 300, "temperature": 0.7}}
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
        response.raise_for_status()

        data = response.json()
        # Handle possible variations in Hugging Face responses
        if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
            return data[0]["generated_text"]
        elif isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]
        else:
            return "⚠️ No valid response received from the AI model."

    except requests.exceptions.HTTPError as http_err:
        status_code = http_err.response.status_code if http_err.response else ""
        if status_code == 401:
            return (
                "❌ Authentication failed with Hugging Face (401 Unauthorized). "
                "Please verify that your API key is correct and has access to the model."
            )
        return f"❌ Network error while contacting Hugging Face: {http_err}"
    except requests.exceptions.RequestException as e:
        return f"❌ Network error while contacting Hugging Face: {e}"
    except json.JSONDecodeError:
        return "❌ Error decoding response from Hugging Face API."
    except Exception as e:
        return f"❌ Unexpected error: {e}"
