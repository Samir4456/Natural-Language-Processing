from flask import Flask, request, render_template_string
import re
import torch
import torch.nn.functional as F

DATA_PATH = "search_data.pt"
app = Flask(__name__)

TEMPLATE = """
<!doctype html>
<html>
  <head>
    <title>NLP Search Engine</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 40px; }
      input[type=text] { width: 420px; padding: 10px; font-size: 16px; }
      select { padding: 10px; font-size: 16px; margin-left: 8px; }
      button { padding: 10px 16px; font-size: 16px; margin-left: 8px; }
      .result { margin-top: 14px; padding: 10px; border: 1px solid #ddd; border-radius: 8px; }
      .meta { color: #555; font-size: 14px; margin-top: 6px; }
      .score { color: #555; font-size: 14px; }
    </style>
  </head>
  <body>
    <h2>Search Similar Context (Dot Product)</h2>

    <form method="GET" action="/">
      <input type="text" name="q" placeholder="Type your query..." value="{{q|default('')}}" />

      <select name="m">
        {% for key, label in model_options %}
          <option value="{{key}}" {% if key == model_selected %}selected{% endif %}>{{label}}</option>
        {% endfor %}
      </select>

      <button type="submit">Search</button>
    </form>

    {% if results is not none %}
      <div class="meta">
        Model: <b>{{ model_selected }}</b>
      </div>

      <h3>Top 10 Results</h3>
      {% if results|length == 0 %}
        <p>No results (empty query).</p>
      {% endif %}

      {% for r in results %}
        <div class="result">
          <div class="score">Score: {{ "%.3f"|format(r.score) }}</div>
          <div>{{ r.text }}</div>
        </div>
      {% endfor %}
    {% endif %}
  </body>
</html>
"""

def tokenize_query(q: str):
    q = q.lower()
    return re.findall(r"[a-z]+(?:'[a-z]+)?", q)

DATA = torch.load(DATA_PATH, map_location="cpu")
stoi = DATA["stoi"]
UNK_ID = DATA["UNK_ID"]
contexts_text = DATA["contexts_text"]
MODEL_STORE = DATA["models"]  # dict: skipgram/neg/glove

MODEL_OPTIONS = [
    ("skipgram", "Skipgram"),
    ("neg", "Skipgram (NEG)"),
    ("glove", "GloVe"),
]

def query_to_vec(query: str, Wn):
    toks = tokenize_query(query)
    if len(toks) == 0:
        return None
    ids = [stoi.get(w, UNK_ID) for w in toks]
    v = Wn[ids].mean(dim=0)
    v = F.normalize(v, p=2, dim=0)
    return v

def search_topk(query: str, model_key: str, k=10):
    model_key = model_key if model_key in MODEL_STORE else "neg"
    Wn = MODEL_STORE[model_key]["Wn"]
    context_vecs = MODEL_STORE[model_key]["context_vecs"]

    qv = query_to_vec(query, Wn)
    if qv is None:
        return []

    scores = torch.mv(context_vecs, qv)
    vals, idxs = torch.topk(scores, k=min(k, scores.numel()))

    out = []
    for s, idx in zip(vals.tolist(), idxs.tolist()):
        out.append(type("R", (), {"score": float(s), "text": contexts_text[idx]}))
    return out

@app.route("/", methods=["GET"])
def index():
    q = request.args.get("q", "").strip()
    m = request.args.get("m", "neg").strip()

    results = None
    if q != "":
        results = search_topk(q, model_key=m, k=10)
    elif "q" in request.args:
        results = []

    return render_template_string(
        TEMPLATE,
        q=q,
        results=results,
        model_selected=m if m in MODEL_STORE else "neg",
        model_options=MODEL_OPTIONS
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
