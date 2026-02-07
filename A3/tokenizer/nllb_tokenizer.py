# tokenizer/nllb_tokenizer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer

@dataclass(frozen=True)
class NLLBLangs:
    ne: str = "npi_Deva"   # Nepali (Devanagari) :contentReference[oaicite:3]{index=3}
    en: str = "eng_Latn"   # English (Latin) :contentReference[oaicite:4]{index=4}

class NLLBMTTokenizer:
    """
    Wrapper around NLLB-200 tokenizer to:
      - set source language
      - build training inputs + labels (teacher forcing)
      - decode outputs
    """
    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        src_lang: str = NLLBLangs.ne,
        tgt_lang: str = NLLBLangs.en,
        max_length: int = 128,
    ):
        self.model_name = model_name
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_length = max_length

        # AutoTokenizer loads NLLB tokenizer (slow or fast depending availability). :contentReference[oaicite:5]{index=5}
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.tok.src_lang = src_lang  # important for NLLB

        self.pad_id = self.tok.pad_token_id
        self.eos_id = self.tok.eos_token_id

        # For generation, NLLB expects a forced BOS token = target language id
        self.forced_bos_token_id = self.tok.convert_tokens_to_ids(tgt_lang)

    def encode_source(self, src_texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Returns input_ids, attention_mask for encoder input.
        """
        batch = self.tok(
            src_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        return batch

    def encode_target_as_labels(self, tgt_texts: List[str]) -> torch.Tensor:
        """
        Returns labels tensor for training (decoder target).
        Pads are set to -100 so loss ignores them.
        """
        with self.tok.as_target_tokenizer():
            self.tok.src_lang = self.tgt_lang  # some versions rely on this context

            tgt = self.tok(
                tgt_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )["input_ids"]

        labels = tgt.clone()
        labels[labels == self.pad_id] = -100
        return labels

    def build_batch(self, src_texts: List[str], tgt_texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Creates model-ready dict: input_ids, attention_mask, labels.
        """
        enc = self.encode_source(src_texts)
        labels = self.encode_target_as_labels(tgt_texts)
        enc["labels"] = labels
        return enc

    def decode(self, ids: torch.Tensor) -> List[str]:
        """
        ids: [B, T] or [T]
        """
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        return self.tok.batch_decode(ids, skip_special_tokens=True)
