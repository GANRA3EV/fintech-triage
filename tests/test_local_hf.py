"""Проверки локального резюме и обучаемых признаков."""
import os
import numpy as np
import pandas as pd
import pytest
from triage.llm import summarize
from triage.models import TicketModel


class FakeEncoder:
    def encode(self, texts):
        return np.array([[text.count("карт") + 1, text.count("перевод") + 1, 1] for text in texts], dtype=float)


@pytest.mark.parametrize("text", ["", None, "   "])
def test_invalid_summary_input(text):
    assert summarize(text, {}, encoder=FakeEncoder())["result"] is None


def test_summary_budget_and_newlines():
    text = "Первая строка без точки\n" + "длинное предложение " * 100
    result = summarize(text, {}, encoder=FakeEncoder())
    assert result["status"] == "draft"
    assert len(result["result"]["summary"]) <= 800
    assert all(quote in text for quote in result["result"]["evidence"])
    assert result["result"]["evidence"][0] == "Первая строка без точки"


def test_rubert_supervised_training():
    model = TicketModel(["card", "transfer"], backend="rubert")
    model.encoder = FakeEncoder()
    frame = pd.DataFrame({"text": ["карта", "карта карта", "перевод", "перевод перевод"],
                          "primary": ["card", "card", "transfer", "transfer"],
                          "secondary": [None, None, None, "card"],
                          "urgency": ["high", "high", "normal", "normal"]})
    model.fit(frame)
    assert model.predict(["карта карта карта"])[0]["primary"] == "card"
    assert model.encoder_revision


@pytest.mark.skipif(os.getenv("TEST_LOCAL_HF") != "1", reason="Веса Hugging Face проверяются по явной команде")
def test_real_hf_cpu_and_summary():
    from triage.local_hf import get_encoder
    encoder = get_encoder()
    vectors = encoder.encode(["Карта заблокирована.", "Статус денежного перевода.", "карта " * 1200])
    assert vectors.shape == (3, 312)
    assert np.isfinite(vectors).all()
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    result = summarize("Карта заблокирована. Не могу совершить перевод. Прошу проверить обращение.", {}, encoder)
    assert result["status"] == "draft"
    frame = pd.DataFrame({"text": ["Моя карта заблокирована", "Не работает банковская карта",
                                   "Статус денежного перевода", "Деньги не пришли переводом"],
                          "primary": ["card", "card", "transfer", "transfer"],
                          "secondary": [None, None, None, "card"],
                          "urgency": ["high", "high", "normal", "normal"]})
    model = TicketModel(["card", "transfer"], backend="rubert").fit(frame)
    assert len(model.predict(["Не работает моя карта"])) == 1


def test_encoder_cache_normalizes_default_arguments(monkeypatch):
    import triage.local_hf as hf
    hf._cached_encoder.cache_clear()
    monkeypatch.setattr(hf, "LocalEncoder", lambda *args: object())
    try:
        assert hf.get_encoder() is hf.get_encoder(hf.MODEL_NAME, hf.MODEL_REVISION)
    finally:
        hf._cached_encoder.cache_clear()
