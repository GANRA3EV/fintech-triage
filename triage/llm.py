"""Совместимый интерфейс локального извлечения резюме; сетевого API нет."""
import re
import numpy as np
from triage.local_hf import MODEL_NAME, MODEL_REVISION, get_encoder


def summarize(text: str, categories: dict, encoder=None) -> dict:
    metadata = {"model": MODEL_NAME, "revision": MODEL_REVISION, "method": "extractive",
                "operator_review": True}
    try:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Нужен непустой текст")
        sentences = list(dict.fromkeys(m[0].strip() for m in re.finditer(
            r"[^.!?\n]+(?:[.!?]+|(?=\n|$))", text) if m[0].strip()))
        if not sentences:
            raise ValueError("Нет предложений для резюме")
        vectors = (encoder or get_encoder()).encode(sentences)
        if len(vectors) != len(sentences) or not np.isfinite(vectors).all():
            raise ValueError("Некорректные признаки предложений")
        center = vectors.mean(axis=0)
        scores = vectors @ center
        selected, used = [], 0
        for position in np.argsort(-scores, kind="stable"):
            sentence = sentences[int(position)]
            if used + len(sentence) + bool(selected) <= 800:
                selected.append(int(position))
                used += len(sentence) + (len(selected) > 1)
            if len(selected) == 3:
                break
        if not selected:
            evidence = [sentences[int(np.argmax(scores))][:800]]
        else:
            evidence = [sentences[index] for index in sorted(selected)]
        if any(quote not in text for quote in evidence):
            raise ValueError("Цитата отсутствует в исходном тексте")
        return {**metadata, "status": "draft",
                "result": {"summary": " ".join(evidence), "evidence": evidence, "fields": []}}
    except Exception as exc:
        return {**metadata, "status": "invalid_or_unavailable", "result": None,
                "error_type": type(exc).__name__}
