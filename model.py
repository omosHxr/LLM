# model.py

import torch
import os
import requests
from bs4 import BeautifulSoup
from tokenizer.tokenizer import CharTokenizer
from model.transformer import MiniTransformer
from config import *

class SmartResponder:
    def __init__(self):
        self.tokenizer = CharTokenizer()
        self.vocab_size = len(self.tokenizer.stoi)
        self.model = MiniTransformer(
            self.vocab_size, MODEL_DIM, NUM_HEADS, NUM_LAYERS, SEQ_LENGTH
        ).to(DEVICE)

        if os.path.exists(MODEL_SAVE_PATH):
            try:
                self.model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
                self.model.eval()
            except Exception as e:
                print(f"[⚠️] Failed to load model checkpoint: {e}")
                exit()
        else:
            print("[❗] No model checkpoint found. Train the model first.")
            exit()

    def generate_response(self, prompt, length=150):
        prefix = "Q: " + prompt.strip() + "\nA:"
        input_ids = torch.tensor([self.tokenizer.encode(prefix)], dtype=torch.long).to(DEVICE)

        if input_ids.shape[1] > SEQ_LENGTH:
            input_ids = input_ids[:, -SEQ_LENGTH:]

        try:
            for _ in range(length):
                out = self.model(input_ids[:, -SEQ_LENGTH:])
                next_token = torch.argmax(out[:, -1, :], dim=-1).unsqueeze(0)
                input_ids = torch.cat((input_ids, next_token), dim=1)

            output = self.tokenizer.decode(input_ids[0].tolist()).replace(prefix, "").strip()

            # Fallback to web if poor output
            if output.lower().count(prompt.lower().split()[0]) > 3 or output.lower() in ["dane", "unknown", "none"]:
                return self.web_lookup(prompt)

            return output
        except Exception as e:
            return f"[❌] Failed to generate response: {str(e)}"

    def web_lookup(self, query):
        result = self.search_google(query)
        if not result:
            result = self.search_bing(query)
        if not result:
            result = self.search_ahmia(query)
        return result or "No useful information found."

    def search_google(self, query):
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(f"https://www.google.com/search?q={query}", headers=headers, timeout=5)
            soup = BeautifulSoup(resp.text, "html.parser")
            snippets = [span.get_text() for span in soup.find_all("span") if span.get_text()]
            return "\n".join(snippets[:5])
        except Exception:
            return None

    def search_bing(self, query):
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(f"https://www.bing.com/search?q={query}", headers=headers, timeout=5)
            soup = BeautifulSoup(resp.text, "html.parser")
            snippets = [p.get_text() for p in soup.find_all("p")]
            return "\n".join(snippets[:5])
        except Exception:
            return None

    def search_ahmia(self, query):
        try:
            resp = requests.get(f"https://ahmia.fi/search/?q={query}", timeout=7)
            soup = BeautifulSoup(resp.text, "html.parser")
            results = soup.find_all("div", class_="result")
            return "\n".join(result.get_text().strip() for result in results[:3])
        except Exception:
            return None
