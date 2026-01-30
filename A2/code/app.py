
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, render_template, request
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
VOCAB_PATH = os.path.join(BASE_DIR, "saved_lstm_lm", "vocab.pt")
MODEL_PATH = os.path.join(BASE_DIR, "saved_lstm_lm", "model.pt")

app = Flask(__name__)

def tokenize(s: str):
    s = s.lower()
    return re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+|[^\w\s]", s)

class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.emb(x)
        out, hidden = self.lstm(x, hidden)
        out = self.drop(out)
        logits = self.fc(out)
        return logits

def sample_next_token(probs, top_k=50, top_p=0.9):
    probs = probs.squeeze(0)

    if top_k is not None and top_k > 0:
        k = min(top_k, probs.numel())
        v, ix = torch.topk(probs, k)
        mask = torch.zeros_like(probs)
        mask[ix] = probs[ix]
        probs = mask / (mask.sum() + 1e-12)

    if top_p is not None and 0 < top_p < 1:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=0)
        cutoff = cumulative > top_p
        cutoff[0] = False
        sorted_probs[cutoff] = 0
        probs = torch.zeros_like(probs)
        probs[sorted_idx] = sorted_probs
        probs = probs / (probs.sum() + 1e-12)

    return torch.multinomial(probs, 1).item()

@torch.no_grad()
def generate_text(model, prompt, stoi, itos, max_new_tokens, temperature, top_k, top_p, device):
    model.eval()
    tokens = tokenize(prompt)
    unk = stoi.get("<unk>", 1)

    ids = [stoi.get(t, unk) for t in tokens]
    if len(ids) == 0:
        ids = [unk]

    idx = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        logits = model(idx)
        next_logits = logits[:, -1, :] / max(temperature, 1e-6)
        probs = F.softmax(next_logits, dim=-1)
        next_id = sample_next_token(probs, top_k=top_k, top_p=top_p)
        idx = torch.cat([idx, torch.tensor([[next_id]], device=device)], dim=1)

    return " ".join(itos[i] for i in idx[0].tolist())

device = "cuda" if torch.cuda.is_available() else "cpu"

vocab = torch.load(VOCAB_PATH, map_location="cpu")
itos = vocab["itos"]
stoi = vocab["stoi"]

vocab_size = len(itos)

embed_dim = 256
hidden_dim = 512
num_layers = 2
dropout = 0.2

model = LSTMLanguageModel(vocab_size, embed_dim, hidden_dim, num_layers, dropout).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

@app.route("/", methods=["GET", "POST"])
def index():
    output = ""
    prompt = "once upon a time"
    max_new_tokens = 80
    temperature = 0.8
    top_k = 50
    top_p = 0.9

    if request.method == "POST":
        prompt = request.form["prompt"]
        max_new_tokens = int(request.form["max_new_tokens"])
        temperature = float(request.form["temperature"])
        top_k = int(request.form["top_k"])
        top_p = float(request.form["top_p"])

        output = generate_text(
            model, prompt, stoi, itos,
            max_new_tokens, temperature, top_k, top_p, device
        )

    return render_template(
        "index.html",
        output=output,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

if __name__ == "__main__":
    app.run(debug=True)
