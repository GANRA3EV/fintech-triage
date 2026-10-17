import hashlib
import json
import importlib.metadata
import platform
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, f1_score

from triage.models import TicketModel
from triage.preprocessing import prepare, temporal_split


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def train(input_path, output, config_path, train_end, validation_end, backend):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    categories = json.loads(Path(config_path).read_text(encoding="utf-8"))
    raw = pd.read_excel(input_path, dtype={"client_id": str, "conversation_id": str})
    diagnostics = {"rows": len(raw), "types": raw.dtypes.astype(str).to_dict(),
                   "missing": raw.isna().sum().to_dict()}
    write_json(output / "diagnostics.json", diagnostics)
    data = prepare(raw)
    unknown = (set(data.primary) | set(data.secondary.dropna()) - {""}) - set(categories)
    if unknown:
        raise ValueError(f"Неизвестные категории: {sorted(unknown)}")
    if not set(data.urgency) <= {"low", "normal", "high", "critical"}:
        raise ValueError("Неизвестная срочность")
    parts = temporal_split(data, train_end, validation_end)
    train_data, validation, test = parts
    if set(train_data.primary) != set(categories):
        raise ValueError("Обучение должно содержать все категории")
    if train_data.urgency.nunique() < 2:
        raise ValueError("Для обучения срочности нужны минимум два класса")
    split_rows = []
    for name, part in zip(("train", "validation", "test"), parts):
        split_rows.extend({"digest": row.digest, "split": name} for row in part.itertuples())
    pd.DataFrame(split_rows).to_excel(output / "split_manifest.xlsx", index=False)
    statistics = data.assign(text_length=data.text.str.len()).groupby("primary").agg(
        count=("text", "size"), min_length=("text_length", "min"),
        median_length=("text_length", "median"), max_length=("text_length", "max"))
    statistics.to_excel(output / "descriptive_statistics.xlsx")
    model = TicketModel(categories, backend).fit(train_data)
    validation_predictions = model.predict(validation.text.tolist())
    candidates = []
    for threshold in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        accepted = [(truth, result) for truth, result in zip(validation.itertuples(), validation_predictions)
                    if min(result["confidence"], result["urgency_confidence"]) >= threshold]
        accuracy = sum(truth.primary == result["primary"] for truth, result in accepted) / len(accepted) if accepted else None
        urgency_accuracy = sum(truth.urgency == result["urgency"] for truth, result in accepted) / len(accepted) if accepted else None
        candidates.append({"threshold": threshold, "coverage": len(accepted) / len(validation),
                           "primary_accuracy": accuracy, "urgency_accuracy": urgency_accuracy})
    eligible = [row for row in candidates if row["primary_accuracy"] is not None
                and row["primary_accuracy"] >= 0.9 and row["urgency_accuracy"] >= 0.9]
    threshold = max(eligible, key=lambda row: row["coverage"])["threshold"] if eligible else 1.01
    predictions = model.predict(test.text.tolist(), threshold=threshold)
    predicted = [row["primary"] for row in predictions]
    predicted_topics = [[row["primary"]] + ([row["secondary"]] if row["secondary"] else []) for row in predictions]
    true_topics = [[row.primary] + ([row.secondary] if isinstance(row.secondary, str) and row.secondary else [])
                   for row in test.itertuples()]
    metrics = {"primary_macro_f1": f1_score(test.primary, predicted, labels=list(categories), average="macro", zero_division=0),
               "topics_macro_f1": f1_score(model.labels.transform(true_topics), model.labels.transform(predicted_topics), average="macro", zero_division=0),
               "urgency_macro_f1": f1_score(test.urgency, [row["urgency"] for row in predictions], average="macro", zero_division=0),
               "review_fraction": sum(row["operator_review"] for row in predictions) / len(predictions),
               "threshold": threshold, "threshold_candidates": candidates,
               "split_sizes": dict(zip(("train", "validation", "test"), map(len, parts))),
               "purged_or_duplicate_rows": len(raw) - len(data),
               "boundary_or_client_exclusions": len(data) - sum(map(len, parts)),
               "classification_report": classification_report(test.primary, predicted, labels=list(categories), zero_division=0, output_dict=True)}
    write_json(output / "metrics.json", metrics)
    pd.DataFrame(predictions).assign(digest=test.digest.tolist()).to_excel(output / "predictions.xlsx", index=False)
    model.encoder = None
    joblib.dump({"model": model, "threshold": threshold, "categories": categories}, output / "model.joblib")
    write_json(output / "run_manifest.json", {
        "input_sha256": hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "train_end": train_end, "validation_end": validation_end, "backend": backend,
        "encoder": model.encoder_name, "seed": 42,
        "encoder_revision": getattr(model, "encoder_revision", None),
        "local_hf_packages": {name: importlib.metadata.version(name) for name in
                              ("torch", "transformers")} if backend == "rubert" else None,
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("pandas", "scikit-learn", "openpyxl", "joblib")},
        "limitations": ["Вероятности не калиброваны", "Порог проверен только на validation",
                        "Корреляции числовых идентификаторов не имеют прикладного смысла",
                        "Регулярные выражения не удаляют все персональные данные"]})
    return metrics
