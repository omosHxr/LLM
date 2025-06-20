# run.py
import torch
import os
import time
from model.transformer import MiniTransformer
from tokenizer.tokenizer import CharTokenizer
from config import *

tokenizer = CharTokenizer()
vocab_size = len(tokenizer.stoi)
os.makedirs("logs", exist_ok=True)

def init_model(vocab_size):
    model = MiniTransformer(vocab_size, MODEL_DIM, NUM_HEADS, NUM_LAYERS, SEQ_LENGTH).to(DEVICE)
    if os.path.exists(MODEL_SAVE_PATH):
        try:
            state_dict = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            print(f"[⚠️] Model mismatch: {e}")
            exit()
    else:
        print(f"[❗] Model not found at {MODEL_SAVE_PATH}. Run training first.")
        exit()
    model.eval()
    return model

model = init_model(vocab_size)

# ─── Generator with Unlimited Output ───
def generate(prompt, max_tokens=1000):
    prefix = f"Q: {prompt.strip()}\nA:"
    input_ids = tokenizer.encode(prefix)

    if not input_ids:
        return "[⚠️] Prompt too short or tokenization failed."

    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)
    generated = input_tensor

    repeated_threshold = 3
    recent_words = []

    for _ in range(max_tokens):
        context = generated[:, -SEQ_LENGTH:]
        out = model(context)
        next_token = torch.argmax(out[:, -1, :], dim=-1).unsqueeze(0)
        generated = torch.cat((generated, next_token), dim=1)

        decoded = tokenizer.decode(generated[0].tolist()).replace(prefix, "").strip()

        # Detect looped or low-entropy output
        words = decoded.split()
        if words:
            word = words[-1].lower()
            recent_words.append(word)
            if recent_words.count(word) > repeated_threshold:
                print("⚠️ [Loop detected] Switching to web response.")
                return tokenizer.web_lookup(prompt)

        # Optional: break if end token appears (if defined)
        if tokenizer.itos[next_token.item()] in ['<END>', '\n\n']:
            break

    result = tokenizer.decode(generated[0].tolist()).replace(prefix, "").strip()
    return result or "[⚠️] No output generated."

# ─── Logger ───
def log(prompt, response):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    with open(f"logs/session_{timestamp}.txt", "a", encoding="utf-8") as f:
        f.write(f"\n🧠 Prompt: {prompt}\n📡 Response:\n{response}\n")

# ─── Main Loop ───
if __name__ == "__main__":
    print("🎯 darkART AI Ops Online. Unlimited Mode Active. Type your prompt...\n")
    while True:
        try:
            prompt = input("🧠 Prompt: ").strip()
            if not prompt:
                continue
            response = generate(prompt)
            print(f"\n📡 darkART says:\n{response}\n")
            log(prompt, response)
        except KeyboardInterrupt:
            print("\n🛑 Session ended.")
            break
        except Exception as e:
            print(f"[❌] Runtime error: {e}")
