# ai_report_generator.py
import os
import requests
import json

# ======================
# CONFIGURATION
# ======================
HF_API_KEY = os.getenv("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")  # Must start with "hf_"
API_URL = "https://api-inference.huggingface.co/models/google/gemma-2b"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}


def generate_ai_report(summary_text: str) -> str:
    """
    Generate a human-like productivity report based on summarized noise data.
    Uses Hugging Face 'google/gemma-2b' (free public model).
    """
    try:
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

    except requests.exceptions.RequestException as e:
        return f"❌ Network error while contacting Hugging Face: {e}"
    except json.JSONDecodeError:
        return "❌ Error decoding response from Hugging Face API."
    except Exception as e:
        return f"❌ Unexpected error: {e}"
