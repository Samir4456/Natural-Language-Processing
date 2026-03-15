from pathlib import Path

from dash import Dash, html, dcc, Input, Output, State
from utils import (
    load_contextual_chunks,
    load_embedding_model,
    encode_texts,
    build_faiss_index,
    load_generator,
    answer_question
)


# LOAD RESOURCES

BASE_DIR = Path(__file__).resolve().parent.parent
CONTEXTUAL_CHUNKS_PATH = BASE_DIR / "outputs" / "task2" / "contextual_chunks.json"

print("Loading contextual chunks...")
chunks = load_contextual_chunks(str(CONTEXTUAL_CHUNKS_PATH))

print("Loading embedding model...")
embedding_model = load_embedding_model()

print("Encoding chunk texts...")
chunk_texts = [chunk["text"] for chunk in chunks]
embeddings = encode_texts(chunk_texts, embedding_model)

print("Building FAISS index...")
index = build_faiss_index(embeddings)

print("Loading generator...")
tokenizer, text_generator = load_generator()

print("System ready.")


# DASH APP

app = Dash(__name__)
app.title = "Chapter 9 QA Chatbot"

PAGE_STYLE = {
    "minHeight": "100vh",
    "background": "linear-gradient(135deg, #0f172a 0%, #111827 100%)",
    "padding": "32px",
    "fontFamily": "Inter, Arial, sans-serif",
    "color": "#e5e7eb"
}

CONTAINER_STYLE = {
    "maxWidth": "1100px",
    "margin": "0 auto"
}

HEADER_CARD_STYLE = {
    "background": "rgba(255,255,255,0.06)",
    "backdropFilter": "blur(10px)",
    "border": "1px solid rgba(255,255,255,0.08)",
    "borderRadius": "24px",
    "padding": "28px",
    "marginBottom": "24px",
    "boxShadow": "0 10px 30px rgba(0,0,0,0.25)"
}

MAIN_CARD_STYLE = {
    "background": "rgba(255,255,255,0.05)",
    "backdropFilter": "blur(10px)",
    "border": "1px solid rgba(255,255,255,0.08)",
    "borderRadius": "24px",
    "padding": "24px",
    "boxShadow": "0 10px 30px rgba(0,0,0,0.25)"
}

INPUT_STYLE = {
    "width": "100%",
    "height": "120px",
    "padding": "16px",
    "fontSize": "16px",
    "borderRadius": "16px",
    "border": "1px solid #334155",
    "backgroundColor": "#0f172a",
    "color": "#f8fafc",
    "outline": "none",
    "resize": "none"
}

BUTTON_STYLE = {
    "padding": "12px 24px",
    "fontSize": "16px",
    "fontWeight": "600",
    "borderRadius": "14px",
    "border": "none",
    "cursor": "pointer",
    "background": "linear-gradient(135deg, #7c3aed 0%, #2563eb 100%)",
    "color": "white",
    "boxShadow": "0 8px 20px rgba(37,99,235,0.35)"
}

ANSWER_STYLE = {
    "padding": "18px",
    "background": "linear-gradient(135deg, rgba(37,99,235,0.18), rgba(124,58,237,0.18))",
    "border": "1px solid rgba(96,165,250,0.25)",
    "borderRadius": "18px",
    "marginBottom": "22px",
    "whiteSpace": "pre-wrap",
    "lineHeight": "1.7",
    "fontSize": "16px",
    "color": "#f8fafc"
}

USER_BUBBLE_STYLE = {
    "background": "linear-gradient(135deg, #2563eb, #1d4ed8)",
    "color": "white",
    "padding": "14px 18px",
    "borderRadius": "18px 18px 4px 18px",
    "maxWidth": "80%",
    "marginLeft": "auto",
    "marginBottom": "18px",
    "boxShadow": "0 8px 18px rgba(37,99,235,0.25)"
}

BOT_BUBBLE_STYLE = {
    "background": "rgba(255,255,255,0.06)",
    "color": "#f8fafc",
    "padding": "18px",
    "borderRadius": "18px 18px 18px 4px",
    "maxWidth": "100%",
    "marginRight": "auto",
    "marginBottom": "22px",
    "border": "1px solid rgba(255,255,255,0.08)"
}

CHUNK_CARD_STYLE = {
    "background": "rgba(15,23,42,0.9)",
    "border": "1px solid rgba(255,255,255,0.08)",
    "borderRadius": "16px",
    "padding": "16px",
    "marginBottom": "14px",
    "color": "#e5e7eb"
}

app.layout = html.Div(
    style=PAGE_STYLE,
    children=[
        html.Div(
            style=CONTAINER_STYLE,
            children=[
                html.Div(
                    style=HEADER_CARD_STYLE,
                    children=[
                        html.Div(
                            style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "gap": "20px", "flexWrap": "wrap"},
                            children=[
                                html.Div(
                                    children=[
                                        html.H1(
                                            "Chapter 9 QA Chatbot",
                                            style={
                                                "margin": "0 0 10px 0",
                                                "fontSize": "44px",
                                                "fontWeight": "800",
                                                "color": "#f8fafc"
                                            }
                                        ),
                                        html.P(
                                            "Contextual Retrieval Web Application for NLP Assignment",
                                            style={
                                                "margin": "0",
                                                "fontSize": "18px",
                                                "color": "#cbd5e1"
                                            }
                                        )
                                    ]
                                ),
                                html.Div(
                                    style={
                                        "padding": "10px 16px",
                                        "borderRadius": "999px",
                                        "background": "rgba(34,197,94,0.14)",
                                        "border": "1px solid rgba(34,197,94,0.3)",
                                        "color": "#86efac",
                                        "fontWeight": "700"
                                    },
                                    children="Contextual Retrieval Active"
                                )
                            ]
                        ),
                        html.Hr(style={"borderColor": "rgba(255,255,255,0.08)", "margin": "20px 0"}),
                        html.Div(
                            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "18px"},
                            children=[
                                html.Div(
                                    style={
                                        "background": "rgba(255,255,255,0.04)",
                                        "borderRadius": "18px",
                                        "padding": "18px",
                                        "border": "1px solid rgba(255,255,255,0.06)"
                                    },
                                    children=[
                                        html.H4("What this app does", style={"marginTop": "0", "color": "#f8fafc"}),
                                        html.P("Ask questions about Chapter 9 and get answers generated from retrieved contextual chunks only.",
                                               style={"marginBottom": "0", "color": "#cbd5e1", "lineHeight": "1.7"})
                                    ]
                                ),
                                html.Div(
                                    style={
                                        "background": "rgba(255,255,255,0.04)",
                                        "borderRadius": "18px",
                                        "padding": "18px",
                                        "border": "1px solid rgba(255,255,255,0.06)"
                                    },
                                    children=[
                                        html.H4("Example questions", style={"marginTop": "0", "color": "#f8fafc"}),
                                        html.Ul(
                                            [
                                                html.Li("What is the goal of instruction tuning?"),
                                                html.Li("What is preference alignment?"),
                                                html.Li("What is the key insight of Direct Preference Optimization?")
                                            ],
                                            style={"marginBottom": "0", "color": "#cbd5e1", "lineHeight": "1.8"}
                                        )
                                    ]
                                )
                            ]
                        )
                    ]
                ),

                html.Div(
                    style=MAIN_CARD_STYLE,
                    children=[
                        html.Div(
                            style={"marginBottom": "18px"},
                            children=[
                                html.Label(
                                    "Ask your question",
                                    style={
                                        "display": "block",
                                        "marginBottom": "10px",
                                        "fontSize": "18px",
                                        "fontWeight": "700",
                                        "color": "#f8fafc"
                                    }
                                ),
                                dcc.Textarea(
                                    id="question-input",
                                    placeholder="Type your question about Chapter 9 here...",
                                    style=INPUT_STYLE
                                )
                            ]
                        ),

                        html.Div(
                            style={"marginBottom": "22px"},
                            children=[
                                html.Div(
                                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"},
                                    children=[
                                        html.Label(
                                            "Top-k retrieved chunks",
                                            style={
                                                "fontSize": "16px",
                                                "fontWeight": "600",
                                                "color": "#e2e8f0"
                                            }
                                        ),
                                        html.Div(
                                            id="topk-label",
                                            style={
                                                "padding": "6px 12px",
                                                "borderRadius": "999px",
                                                "backgroundColor": "rgba(255,255,255,0.08)",
                                                "border": "1px solid rgba(255,255,255,0.08)",
                                                "fontWeight": "700",
                                                "color": "#f8fafc"
                                            }
                                        )
                                    ]
                                ),
                                dcc.Slider(
                                    id="topk-slider",
                                    min=1,
                                    max=5,
                                    step=1,
                                    value=2,
                                    marks={i: {"label": str(i), "style": {"color": "#cbd5e1"}} for i in range(1, 6)}
                                )
                            ]
                        ),

                        html.Div(
                            style={"display": "flex", "gap": "12px", "marginBottom": "28px"},
                            children=[
                                html.Button("Ask", id="ask-button", n_clicks=0, style=BUTTON_STYLE),
                                html.Button(
                                    "Clear",
                                    id="clear-button",
                                    n_clicks=0,
                                    style={
                                        "padding": "12px 24px",
                                        "fontSize": "16px",
                                        "fontWeight": "600",
                                        "borderRadius": "14px",
                                        "border": "1px solid rgba(255,255,255,0.12)",
                                        "cursor": "pointer",
                                        "background": "rgba(255,255,255,0.04)",
                                        "color": "#f8fafc"
                                    }
                                )
                            ]
                        ),

                        dcc.Store(id="chat-store", data=[]),

                        dcc.Loading(
                            id="loading",
                            type="circle",
                            color="#8b5cf6",
                            children=html.Div(id="chat-area")
                        )
                    ]
                )
            ]
        )
    ]
)


def build_chunk_card(chunk: dict):
    return html.Details(
        style=CHUNK_CARD_STYLE,
        children=[
            html.Summary(
                f"Chunk {chunk['chunk_id']} | Paragraph {chunk['source_paragraph_id']} | Score {chunk['score']:.4f}",
                style={
                    "cursor": "pointer",
                    "fontWeight": "700",
                    "color": "#93c5fd",
                    "marginBottom": "10px"
                }
            ),
            html.Div(
                children=[
                    html.P("Context Prefix", style={"fontWeight": "700", "marginBottom": "6px", "color": "#f8fafc"}),
                    html.P(chunk.get("context_prefix", ""), style={"color": "#cbd5e1", "lineHeight": "1.7"}),
                    html.P("Original Chunk Text", style={"fontWeight": "700", "marginBottom": "6px", "color": "#f8fafc", "marginTop": "14px"}),
                    html.P(chunk.get("original_text", chunk["text"]), style={"color": "#cbd5e1", "lineHeight": "1.7"})
                ]
            )
        ]
    )


def render_chat(chat_history):
    if not chat_history:
        return html.Div(
            style={
                "padding": "28px",
                "borderRadius": "18px",
                "border": "1px dashed rgba(255,255,255,0.15)",
                "textAlign": "center",
                "color": "#94a3b8",
                "background": "rgba(255,255,255,0.02)"
            },
            children=[
                html.H3("No messages yet", style={"color": "#e2e8f0"}),
                html.P("Ask a question above to start chatting with the Chapter 9 assistant.")
            ]
        )

    rendered = []
    for item in chat_history:
        rendered.append(
            html.Div(
                style=USER_BUBBLE_STYLE,
                children=item["question"]
            )
        )

        rendered.append(
            html.Div(
                style=BOT_BUBBLE_STYLE,
                children=[
                    html.H3("Answer", style={"marginTop": "0", "color": "#f8fafc"}),
                    html.Div(item["answer"], style=ANSWER_STYLE),
                    html.H3("Source Chunks Used", style={"color": "#f8fafc"}),
                    html.Div([build_chunk_card(chunk) for chunk in item["retrieved_chunks"]])
                ]
            )
        )

    return html.Div(rendered)


@app.callback(
    Output("topk-label", "children"),
    Input("topk-slider", "value")
)
def update_topk_label(value):
    return f"{value} chunks"


@app.callback(
    Output("chat-store", "data"),
    Input("ask-button", "n_clicks"),
    Input("clear-button", "n_clicks"),
    State("question-input", "value"),
    State("topk-slider", "value"),
    State("chat-store", "data"),
    prevent_initial_call=True
)
def update_chat(ask_clicks, clear_clicks, question, top_k, chat_history):
    from dash import ctx

    if not ctx.triggered_id:
        return chat_history

    if ctx.triggered_id == "clear-button":
        return []

    if not question or not question.strip():
        return chat_history

    result = answer_question(
        question=question.strip(),
        chunks=chunks,
        index=index,
        embedding_model=embedding_model,
        tokenizer=tokenizer,
        text_generator=text_generator,
        top_k=top_k
    )

    chat_history = chat_history or []
    chat_history.append(result)
    return chat_history


@app.callback(
    Output("chat-area", "children"),
    Input("chat-store", "data")
)
def update_chat_area(chat_history):
    return render_chat(chat_history)


if __name__ == "__main__":
    app.run(debug=True)