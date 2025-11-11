# train_local_chatbot.py
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

BASE_MODEL = os.environ.get("BASE_MODEL", "microsoft/phi-2")   # good on mid GPUs; for CPU, try "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_TRAIN = "datasets/coach_dataset.jsonl"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map=None,      # force full load to a single device
    trust_remote_code=False
).to("cpu")               # explicitly move everything to CPU


# Optional: 8-bit for speed/VRAM if bitsandbytes installed
USE_8BIT = os.environ.get("USE_8BIT", "0") == "1"
if USE_8BIT:
    from transformers import BitsAndBytesConfig
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        load_in_8bit=True,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        trust_remote_code=False
    )
    model = prepare_model_for_kbit_training(model)

lora_cfg = LoraConfig(
    r=8, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]  # broad coverage; safe for Phi/Mistral/Gemma
)
model = get_peft_model(model, lora_cfg)
model.to("cpu")   # or .to("cuda") if you have GPU

ds = load_dataset("json", data_files=DATA_TRAIN)["train"]

def format_example(q, a):
    return f"User: {q}\nAssistant: {a}"

def tok(batch):
    texts = [format_example(q, a) for q, a in zip(batch["instruction"], batch["response"])]
    out = tokenizer(
        texts, max_length=512, truncation=True, padding="max_length", return_tensors=None
    )
    out["labels"] = out["input_ids"].copy()
    return out

tok_ds = ds.map(tok, batched=True, remove_columns=ds.column_names)

args = TrainingArguments(
    output_dir="coach_model",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="epoch",
    bf16=False,
    fp16=False,  # let bf16 handle if supported
)

trainer = Trainer(model=model, args=args, train_dataset=tok_ds)
trainer.train()

# Save merged LoRA or just adapters. For inference we can keep the adapter path.
model.save_pretrained("coach_model")
tokenizer.save_pretrained("coach_model")
print("✅ Fine-tune complete -> coach_model/")
