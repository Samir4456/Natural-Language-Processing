import json
from typing import List, Dict

import numpy as np
import faiss
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
TOP_K = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_contextual_chunks(json_path: str) -> List[Dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)


def encode_texts(
    texts: List[str],
    embedding_model: SentenceTransformer,
    batch_size: int = 32
) -> np.ndarray:
    embeddings = embedding_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embeddings.astype("float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def retrieve_top_k(
    query: str,
    chunks: List[Dict],
    index: faiss.Index,
    embedding_model: SentenceTransformer,
    top_k: int = TOP_K
) -> List[Dict]:
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        chunk = chunks[int(idx)]
        results.append({
            "rank": rank,
            "score": float(score),
            "chunk_id": chunk["chunk_id"],
            "source_paragraph_id": chunk["source_paragraph_id"],
            "context_prefix": chunk.get("context_prefix", ""),
            "original_text": chunk.get("original_text", chunk["text"]),
            "text": chunk["text"]
        })
    return results


def load_generator():
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    return tokenizer, text_generator


def build_contextual_rag_prompt(question: str, retrieved_chunks: List[Dict]) -> str:
    context_blocks = []
    for item in retrieved_chunks:
        context_blocks.append(
            f"[Chunk {item['chunk_id']} | Paragraph {item['source_paragraph_id']}]\n{item.get('original_text', item['text'])}"
        )

    context_text = "\n\n".join(context_blocks)

    prompt = f"""
You are answering a question about a textbook chapter.

Rules:
1. Use ONLY the retrieved context below.
2. Do NOT use outside knowledge.
3. Keep the answer concise.
4. Answer in 1-2 sentences.

Question:
{question}

Retrieved Context:
{context_text}

Answer:
"""
    return prompt.strip()


def generate_answer(
    prompt: str,
    tokenizer,
    text_generator,
    max_new_tokens: int = 60
) -> str:
    messages = [
        {"role": "system", "content": "You are a careful academic assistant."},
        {"role": "user", "content": prompt}
    ]

    text_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    outputs = text_generator(
        text_input,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_full_text=False
    )

    return outputs[0]["generated_text"].strip()


def answer_question(
    question: str,
    chunks: List[Dict],
    index: faiss.Index,
    embedding_model: SentenceTransformer,
    tokenizer,
    text_generator,
    top_k: int = TOP_K
) -> Dict:
    retrieved = retrieve_top_k(
        query=question,
        chunks=chunks,
        index=index,
        embedding_model=embedding_model,
        top_k=top_k
    )

    prompt = build_contextual_rag_prompt(question, retrieved)
    answer = generate_answer(prompt, tokenizer, text_generator)

    return {
        "question": question,
        "answer": answer,
        "retrieved_chunks": retrieved
    }