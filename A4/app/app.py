
from flask import Flask, request, jsonify, render_template
import torch
import numpy as np

from model_def import load_sbert

app = Flask(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOKENIZER, MODEL, ID2LABEL, MAX_LEN = load_sbert("models/sbert_softmax_snli_meta.json", device=DEVICE)

def predict_nli(premise: str, hypothesis: str):
    a = TOKENIZER(premise, truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
    b = TOKENIZER(hypothesis, truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")

    a_ids = a["input_ids"].to(DEVICE)
    a_mask = a["attention_mask"].to(DEVICE)
    b_ids = b["input_ids"].to(DEVICE)
    b_mask = b["attention_mask"].to(DEVICE)

    with torch.no_grad():
        logits = MODEL(a_ids, a_mask, b_ids, b_mask)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        pred = int(np.argmax(probs))

    return {"label": ID2LABEL[pred], "probs": probs.tolist()}

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    premise = (data.get("premise") or "").strip()
    hypothesis = (data.get("hypothesis") or "").strip()

    if not premise or not hypothesis:
        return jsonify({"error": "Please provide both premise and hypothesis."}), 400

    return jsonify(predict_nli(premise, hypothesis))

if __name__ == "__main__":
    # Run: python app/app.py
    app.run(host="0.0.0.0", port=5000, debug=True)
