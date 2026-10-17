import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from triage.experiment import load_dataset, write_json


def paired_intervals(truth, predictions, groups, labels, repetitions, seed):
    rng = np.random.default_rng(seed)
    units = list(dict.fromkeys(groups))
    indices = [np.flatnonzero(np.asarray(groups) == unit) for unit in units]
    pairs = list(itertools.combinations(predictions, 2))
    values = {pair: [] for pair in pairs}
    truth = np.asarray(truth)
    predictions = {name: np.asarray(p) for name, p in predictions.items()}
    for _ in range(repetitions):
        sample = np.concatenate([indices[i] for i in rng.integers(0, len(units), len(units))])
        scores = {name: f1_score(truth[sample], p[sample], labels=labels, average='macro', zero_division=0)
                  for name, p in predictions.items()}
        for a, b in pairs:
            values[(a, b)].append(scores[b] - scores[a])
    rows = []
    for a, b in pairs:
        low, high = np.quantile(values[(a, b)], [0.025, 0.975])
        difference = f1_score(truth, predictions[b], labels=labels, average='macro', zero_division=0) - f1_score(
            truth, predictions[a], labels=labels, average='macro', zero_division=0)
        rows.append({'reference': a, 'candidate': b, 'primary_macro_f1_difference': difference,
                     'ci_low': low, 'ci_high': high, 'clients': len(units), 'repetitions': repetitions})
    return pd.DataFrame(rows)


def run_study(input_path, config_path, categories_path, output):
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    backends = config['backends']
    if not backends or len(backends) != len(set(backends)) or set(backends) - {'majority', 'tfidf', 'rubert', 'e5'}:
        raise ValueError('Нужен непустой список уникальных способов классификации')
    repetitions = config.get('bootstrap_repetitions', 1000)
    if not isinstance(repetitions, int) or repetitions < 100:
        raise ValueError('Нужно хотя бы 100 повторений bootstrap')
    categories, prepared, parts, diagnostics = load_dataset(input_path, categories_path,
        config['train_end'], config['validation_end'])
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    for folder in ('article', 'data', 'tables', 'figures', 'code', 'references', 'logs', 'kb_context'):
        (root / folder).mkdir()
    provenance = {'input_sha256': hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
                  'categories_sha256': hashlib.sha256(Path(categories_path).read_bytes()).hexdigest(),
                  'train_end': config['train_end'], 'validation_end': config['validation_end']}
    write_json(root / 'code' / 'config.json', config)
    write_json(root / 'logs' / 'diagnostics.json', diagnostics)
    write_json(root / 'logs' / 'metadata.json', {'categories': categories, 'provenance': provenance})
    dataset = root / 'data' / 'split.xlsx'
    with pd.ExcelWriter(dataset) as writer:
        for name, part in zip(('train', 'validation', 'test'), parts):
            part = part.copy()
            for col in ('timestamp', 'start'):
                part[col] = part[col].astype(str)
            part.to_excel(writer, sheet_name=name, index=False)
        prepared.assign(text_length=prepared.text.str.len()).groupby('primary').text_length.describe().to_excel(
            writer, sheet_name='text_lengths')
    write_json(root / 'logs' / 'protocol.json', {
        'research_question': 'Влияние представления текста на классификацию тем, срочности и затраты ресурсов',
        'primary_metric': 'primary_macro_f1', 'secondary_metrics': ['urgency_macro_f1', 'critical_recall', 'review_fraction'],
        'config': config, 'provenance': provenance,
        'split_sha256': hashlib.sha256(dataset.read_bytes()).hexdigest(),
        'data_kind': config.get('data_kind', 'unspecified'),
        'bootstrap': 'Парные процентильные интервалы 95%; единица повторной выборки — клиент; обучение не повторяется',
        'limitations': ['Синтетические данные проверяют выполнение, предметное качество по ним не оценивается',
                        'Интервалы условны относительно обученных моделей и выбранного периода',
                        'Разные способы сравниваются на одной машине последовательно; системная нагрузка не контролируется']})
    summary, predictions, files = [], {}, {}
    for backend in backends:
        print(f'Эксперимент: {backend}', flush=True)
        destination = root / 'data' / backend
        command = [sys.executable, '-m', 'triage.study_worker', '--data', str(dataset),
            '--metadata', str(root / 'logs' / 'metadata.json'), '--backend', backend, '--output', str(destination)]
        try:
            result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace',
                env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}, timeout=config.get('worker_timeout_seconds', 900),
                cwd=Path(__file__).resolve().parents[1])
        except subprocess.TimeoutExpired:
            (root / 'logs' / f'{backend}.log').write_text('Превышено время ожидания эксперимента', encoding='utf-8')
            summary.append({'backend': backend, 'status': 'failed', 'error_type': 'TimeoutExpired'})
            continue
        (root / 'logs' / f'{backend}.log').write_text(result.stdout + result.stderr, encoding='utf-8')
        if result.returncode:
            summary.append({'backend': backend, 'status': 'failed', 'exit_code': result.returncode})
            continue
        metrics = json.loads((destination / 'metrics.json').read_text(encoding='utf-8'))
        resources = json.loads((destination / 'resources.json').read_text(encoding='utf-8'))
        summary.append({'backend': backend, 'status': 'ok', **{k: v for k, v in metrics.items()
            if isinstance(v, (float, int)) or v is None}, **resources})
        table = pd.read_excel(destination / 'predictions.xlsx')
        if table.digest.tolist() != parts[2].digest.tolist():
            raise ValueError('Нарушено соответствие тестовых строк')
        predictions[backend] = table.primary.tolist()
        files[backend] = table
    comparison = pd.DataFrame(summary)
    intervals = paired_intervals(parts[2].primary, predictions, parts[2].client_id.tolist(),
        list(categories), repetitions, config.get('seed', 42)) if len(predictions) >= 2 else pd.DataFrame()
    with pd.ExcelWriter(root / 'tables' / 'comparison.xlsx') as writer:
        comparison.to_excel(writer, sheet_name='metrics', index=False)
        intervals.to_excel(writer, sheet_name='paired_intervals', index=False)
        for name, table in files.items():
            pd.crosstab(table.true_primary, table.primary).to_excel(writer, sheet_name=f'confusion_{name}')
    coverage = pd.DataFrame([
        {'component': 'Данные', 'evidence': 'data/split.xlsx', 'limitation': config.get('data_kind', 'unspecified')},
        {'component': 'Методы', 'evidence': 'logs/protocol.json', 'limitation': 'Параметры классификаторов фиксированы'},
        {'component': 'Результаты', 'evidence': 'tables/comparison.xlsx', 'limitation': 'Один временной период'},
        {'component': 'Источники', 'evidence': 'logs/metadata.json', 'limitation': 'Библиографический обзор не выполнялся'}])
    coverage.to_excel(root / 'tables' / 'coverage.xlsx', index=False)
    make_figure(comparison, root / 'figures' / 'comparison.png')
    lines = ['# Сравнение классификаторов', '', f"Тип данных: {config.get('data_kind', 'unspecified')}.",
             f'Переписок в обучении, проверке и тесте: {len(parts[0])}, {len(parts[1])}, {len(parts[2])}.',
             f"После очистки осталось {len(prepared)} переписок; исключено по времени и клиентам {diagnostics['boundary_or_client_exclusions']}.", '',
             'В табл. 1 приведены метрики основной темы. Полные результаты находятся в tables/comparison.xlsx.', '',
             '| Вариант | Состояние | Macro-F1 темы |', '|---|---|---|',
             *[f"| {r['backend']} | {r['status']} | {r.get('primary_macro_f1', 'нет результата')} |" for r in summary], '',
             'На рис. 1 сопоставлены оценки Macro-F1 основной темы.', '',
             '![Рис. 1. Классификация основной темы](../figures/comparison.png)', '',
             'Ошибки запуска сохранены в logs.',
             'Синтетический пример не служит оценкой качества на реальных обращениях.',
             'Интервалы учитывают состав тестовых клиентов, но не повторное обучение.', '',
             *[f"- {r['backend']}: {r['status']}" for r in summary]]
    report = '\n'.join(lines) + '\n'
    (root / 'article' / 'report.md').write_text(report, encoding='utf-8')
    write_json(root / 'logs' / 'word_count_report.json', {'report_words': len(report.split())})
    write_json(root / 'logs' / 'artifacts.json', {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob('*') if p.is_file()})
    print('Созданные файлы:')
    for path in sorted(root.rglob('*')):
        if path.is_file():
            print(path)
    return not comparison.status.eq('failed').any()


def make_figure(table, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    table = table.loc[table.status.eq('ok')]
    if table.empty:
        return
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(table.backend, table.primary_macro_f1)
    ax.set_ylabel('Macro-F1 основной темы')
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
