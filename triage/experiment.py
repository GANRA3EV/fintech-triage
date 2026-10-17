import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import time

import joblib
import pandas as pd

from triage.evaluation import choose_threshold, evaluate
from triage.models import TicketModel
from triage.preprocessing import prepare, temporal_split


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def load_dataset(input_path, config_path, train_end, validation_end):
    categories = json.loads(Path(config_path).read_text(encoding='utf-8'))
    raw = pd.read_excel(input_path, dtype={'client_id': str, 'conversation_id': str})
    data = prepare(raw)
    unknown = (set(data.primary) | (set(data.secondary.dropna()) - {''})) - set(categories)
    if unknown:
        raise ValueError(f'Неизвестные категории: {sorted(unknown)}')
    if not set(data.urgency) <= {'low', 'normal', 'high', 'critical'}:
        raise ValueError('Неизвестная срочность')
    parts = temporal_split(data, train_end, validation_end)
    if set(parts[0].primary) != set(categories):
        raise ValueError('Обучение должно содержать все категории')
    if parts[0].urgency.nunique() < 2:
        raise ValueError('Для обучения срочности нужны минимум два класса')
    diagnostics = {'rows': len(raw), 'types': raw.dtypes.astype(str).to_dict(),
                   'missing': raw.isna().sum().to_dict(), **data.attrs['preparation_counts'],
                   'boundary_or_client_exclusions': len(data) - sum(map(len, parts))}
    return categories, data, parts, diagnostics


def execute(parts, categories, output, backend, provenance):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    training, validation, test = parts
    start = time.perf_counter()
    model = TicketModel(categories, backend).fit(training)
    fit_seconds = time.perf_counter() - start
    threshold, candidates = choose_threshold(validation, model.predict(validation.text.tolist()))
    start = time.perf_counter()
    predictions = model.predict(test.text.tolist(), threshold=threshold)
    predict_seconds = time.perf_counter() - start
    metrics = evaluate(model, test, predictions)
    metrics.update(threshold=threshold, threshold_candidates=candidates,
                   split_sizes=dict(zip(('train', 'validation', 'test'), map(len, parts))),
                   fit_seconds=fit_seconds, predict_seconds=predict_seconds,
                   predict_seconds_per_conversation=predict_seconds / len(test))
    write_json(output / 'metrics.json', metrics)
    table = pd.DataFrame(predictions).assign(digest=test.digest.tolist(),
        true_primary=test.primary.tolist(), true_urgency=test.urgency.tolist())
    table.to_excel(output / 'predictions.xlsx', index=False)
    pd.DataFrame(metrics['classification_report']).T.to_excel(output / 'class_metrics.xlsx')
    model.encoder = None
    joblib.dump({'model': model, 'threshold': threshold, 'categories': categories}, output / 'model.joblib')
    write_json(output / 'run_manifest.json', {
        **provenance, 'backend': backend, 'encoder': model.encoder_name if backend in ('rubert', 'e5') else None,
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
        'encoder_revision': model.encoder_revision, 'seed': 42, 'python': platform.python_version(),
        'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions() if d.metadata['Name']},
        'limitations': ['Вероятности не калиброваны', 'Порог подобран только на validation',
            'Результаты относятся к новым клиентам после временной границы',
            'Время fit включает загрузку кодировщика; время predict относится к прогретой модели',
            'Регулярные выражения не удаляют все персональные данные']})
    return metrics


def train(input_path, output, config_path, train_end, validation_end, backend):
    categories, data, parts, diagnostics = load_dataset(input_path, config_path, train_end, validation_end)
    provenance = {'input_sha256': hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
                  'config_sha256': hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
                  'train_end': train_end, 'validation_end': validation_end}
    metrics = execute(parts, categories, output, backend, provenance)
    output = Path(output)
    write_json(output / 'diagnostics.json', diagnostics)
    pd.DataFrame([{'digest': row.digest, 'split': name} for name, part in zip(('train', 'validation', 'test'), parts)
                  for row in part.itertuples()]).to_excel(output / 'split_manifest.xlsx', index=False)
    data.assign(text_length=data.text.str.len()).groupby('primary').agg(
        count=('text', 'size'), min_length=('text_length', 'min'), median_length=('text_length', 'median'),
        max_length=('text_length', 'max')).to_excel(output / 'descriptive_statistics.xlsx')
    metrics.update(preparation_counts=data.attrs['preparation_counts'],
                   boundary_or_client_exclusions=diagnostics['boundary_or_client_exclusions'])
    write_json(output / 'metrics.json', metrics)
    return metrics
