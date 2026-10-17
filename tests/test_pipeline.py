import json

import pandas as pd
import pytest

from triage.llm import summarize
from triage.models import TicketModel
from triage.preprocessing import clean_text, prepare, temporal_split


def test_end_to_end(tmp_path):
    from triage.experiment import train

    rows = []
    for month in (1, 2, 3):
        for category in ("card", "transfer"):
            for sample in range(4):
                identifier = f"{month}-{category}-{sample}"
                rows.append({"client_id": identifier, "conversation_id": identifier,
                             "timestamp": f"2025-{month:02d}-01", "text": f"{category} запрос {identifier}",
                             "primary": category, "secondary": None,
                             "urgency": "high" if sample % 2 else "normal"})
    source = tmp_path / "input.xlsx"
    pd.DataFrame(rows).to_excel(source, index=False)
    config = tmp_path / "categories.json"
    config.write_text(json.dumps({"card": "Карта", "transfer": "Перевод"}), encoding="utf-8")
    output = tmp_path / "results"
    metrics = train(source, output, config, "2025-02-01", "2025-03-01", "tfidf")
    assert metrics["split_sizes"] == {"train": 8, "validation": 8, "test": 8}
    assert (output / "model.joblib").exists()
    assert len(pd.read_excel(output / "predictions.xlsx")) == 8


def test_cleaning():
    result = clean_text('<p>Карта</p> a@example.org +7 (999) 123-45-67\nС уважением Иван')
    assert "example" not in result and "999" not in result and "Иван" not in result
    assert "Карта" in result


def test_split_purges_clients_and_crossing_threads():
    data = pd.DataFrame([
        {"client_id": "a", "start": "2025-01-01", "timestamp": "2025-01-01"},
        {"client_id": "a", "start": "2025-02-01", "timestamp": "2025-02-01"},
        {"client_id": "b", "start": "2025-02-01", "timestamp": "2025-02-01"},
        {"client_id": "b", "start": "2025-03-01", "timestamp": "2025-03-01"},
        {"client_id": "c", "start": "2025-03-01", "timestamp": "2025-03-01"},
        {"client_id": "d", "start": "2025-01-30", "timestamp": "2025-02-02"},
    ])
    for column in ("start", "timestamp"):
        data[column] = pd.to_datetime(data[column], utc=True)
    parts = temporal_split(data, "2025-02-01", "2025-03-01")
    assert [set(part.client_id) for part in parts] == [{"a"}, {"b"}, {"c"}]


def test_conflicting_annotation_rejected():
    frame = pd.DataFrame([{"client_id": "a", "conversation_id": "a", "timestamp": "2025-01-01",
                           "text": "карта", "primary": category, "secondary": None, "urgency": "normal"}
                          for category in ("card", "fraud")])
    with pytest.raises(ValueError):
        prepare(frame)


def test_deduplication_and_thread_merge():
    frame = pd.DataFrame([
        {"client_id": "a", "conversation_id": "a", "timestamp": "2025-01-01", "text": "<p>Карта</p>"},
        {"client_id": "a", "conversation_id": "a", "timestamp": "2025-01-02", "text": "заблокирована"},
        {"client_id": "b", "conversation_id": "b", "timestamp": "2025-01-03", "text": "Карта заблокирована"},
    ])
    result = prepare(frame, labeled=False)
    assert len(result) == 1
    assert result.iloc[0].text == "Карта заблокирована"


def test_classifier_and_abstention():
    data = pd.DataFrame({"text": ["блокировка карты", "заблокировать карту", "перевод денег", "перевести деньги"],
                         "primary": ["card", "card", "transfer", "transfer"],
                         "secondary": [None, None, "card", None], "urgency": ["high", "high", "normal", "normal"]})
    model = TicketModel(["card", "transfer"]).fit(data)
    result = model.predict(["блокировка карты"], threshold=1.01)[0]
    assert result["primary"] == "card"
    assert result["operator_review"] and result["operator_confirmation_required"]


def test_local_summary_quotes_source():
    import numpy as np
    class Encoder:
        def encode(self, texts):
            return np.ones((len(texts), 3))
    text = "Моя карта заблокирована. Перевод не проходит. Нужна проверка."
    result = summarize(text, {}, encoder=Encoder())
    assert result["status"] == "draft"
    assert all(quote in text for quote in result["result"]["evidence"])
    assert result["method"] == "extractive"


def test_local_summary_failure_falls_back():
    class Encoder:
        def encode(self, texts):
            raise RuntimeError("private details")
    result = summarize("Карта.", {}, encoder=Encoder())
    assert result["operator_review"] and result["result"] is None
    assert "private details" not in str(result)
