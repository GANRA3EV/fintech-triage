"""Локальный русский кодировщик Hugging Face с закреплённой ревизией."""
from functools import lru_cache
import numpy as np

MODEL_NAME = "cointegrated/rubert-tiny2"
MODEL_REVISION = "e8ed3b0c8bbf4fb6984c3de043bf7d2f4e5969ae"


class LocalEncoder:
    def __init__(self, name=MODEL_NAME, revision=MODEL_REVISION):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(name, revision=revision, trust_remote_code=False,
                                             use_safetensors=True).to("cpu").eval()

    def encode(self, texts):
        if not texts:
            return np.empty((0, self.model.config.hidden_size))
        encoded = self.tokenizer(texts, padding=True, truncation=True, max_length=512,
                                 stride=64, return_overflowing_tokens=True, return_tensors="pt")
        mapping = encoded.pop("overflow_to_sample_mapping").numpy()
        vectors = []
        with self.torch.inference_mode():
            for start in range(0, len(mapping), 16):
                batch = {key: value[start:start + 16] for key, value in encoded.items()}
                hidden = self.model(**batch).last_hidden_state[:, 0, :]
                vectors.append(self.torch.nn.functional.normalize(hidden).cpu().numpy())
        windows = np.concatenate(vectors)
        result = np.stack([windows[mapping == index].mean(axis=0) for index in range(len(texts))])
        return result / np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)


@lru_cache(maxsize=2)
def get_encoder(name=MODEL_NAME, revision=MODEL_REVISION):
    return LocalEncoder(name, revision)
