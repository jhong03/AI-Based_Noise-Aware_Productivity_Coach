# ai_report_generator_local.py
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

MODEL_PATH = "coach_model"  # folder you just trained

print("🧠 Loading local model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to("cpu")

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

SYSTEM_PROMPT = (
    "You are a friendly productivity coach. "
    "Analyze the user's daily noise and focus summary, "
    "then write 4–6 warm, motivational sentences ending with one to three actionable tip based on the quality of focus."
)

def generate_ai_report(summary_text: str) -> str:
    prompt = f"{SYSTEM_PROMPT}\n\nSummary:\n{summary_text}\n\nCoach:"
    result = generator(prompt, max_new_tokens=200, temperature=0.7, do_sample=True)
    text = result[0]["generated_text"]
    # trim everything before the "Coach:" marker
    return text.split("Coach:")[-1].strip()

def refresh_model():
    global model, tokenizer, generator
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to("cpu")
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    print("🔄 Reloaded updated coach model.")
