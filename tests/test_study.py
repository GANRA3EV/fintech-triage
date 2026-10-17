import pandas as pd
import pytest

from triage.evaluation import choose_threshold
from triage.preprocessing import prepare
from triage.study import paired_intervals


def test_critical_predictions_excluded_from_coverage():
    data = pd.DataFrame({'primary': ['card', 'card'], 'urgency': ['critical', 'normal']})
    predictions = [{'primary': 'card', 'urgency': u, 'confidence': 0.99, 'urgency_confidence': 0.99}
                   for u in data.urgency]
    threshold, candidates = choose_threshold(data, predictions)
    assert threshold == 0.5
    assert all(r['coverage'] == 0.5 for r in candidates)


def test_all_critical_predictions_abstain():
    data = pd.DataFrame({'primary': ['card'], 'urgency': ['critical']})
    predictions = [{'primary': 'card', 'urgency': 'critical', 'confidence': 1., 'urgency_confidence': 1.}]
    threshold, _ = choose_threshold(data, predictions)
    assert threshold == 1.01


def test_preparation_counts_separate_operations():
    data = pd.DataFrame([
        {'client_id': 'a', 'conversation_id': 'a', 'timestamp': '2025-01-01', 'text': 'карта'},
        {'client_id': 'a', 'conversation_id': 'a', 'timestamp': '2025-01-02', 'text': 'заблокирована'},
        {'client_id': 'b', 'conversation_id': 'b', 'timestamp': '2025-01-03', 'text': 'карта заблокирована'},
        {'client_id': 'c', 'conversation_id': 'c', 'timestamp': '2025-01-04', 'text': '<p></p>'}])
    counts = prepare(data, labeled=False).attrs['preparation_counts']
    assert counts == {'input_messages': 4, 'empty_after_cleaning': 1, 'messages_absorbed_in_threads': 1,
                      'duplicate_conversations': 1, 'prepared_conversations': 1}


def test_paired_bootstrap_identical_predictions_have_zero_difference():
    result = paired_intervals(['a', 'b', 'a', 'b'], {'x': ['a', 'b', 'a', 'b'], 'y': ['a', 'b', 'a', 'b']},
                              ['c1', 'c1', 'c2', 'c3'], ['a', 'b'], 100, 42)
    assert result.iloc[0].primary_macro_f1_difference == 0
    assert result.iloc[0].ci_low == result.iloc[0].ci_high == 0
    assert result.iloc[0].clients == 3


def test_majority_uses_training_labels_only():
    from triage.models import TicketModel
    data = pd.DataFrame({'text': ['карта', 'перевод', 'счёт'], 'primary': ['a', 'b', 'a'],
                         'secondary': [None] * 3, 'urgency': ['normal', 'high', 'normal']})
    model = TicketModel(['a', 'b'], 'majority').fit(data)
    result = model.predict(['совсем новая тема'])[0]
    assert result['primary'] == 'a'
    assert result['urgency'] == 'normal'


def test_study_saves_common_splits_and_refuses_overwrite(tmp_path):
    import json
    from triage.study import run_study
    rows = []
    for month in (1, 2, 3):
        for category in ('card', 'transfer'):
            for sample in range(4):
                identifier = f'{month}-{category}-{sample}'
                rows.append({'client_id': identifier, 'conversation_id': identifier,
                    'timestamp': f'2025-{month:02d}-01', 'text': f'{category} запрос {identifier}',
                    'primary': category, 'secondary': None, 'urgency': 'high' if sample % 2 else 'normal'})
    source = tmp_path / 'source.xlsx'
    pd.DataFrame(rows).to_excel(source, index=False)
    categories = tmp_path / 'categories.json'
    categories.write_text(json.dumps({'card': 'Карта', 'transfer': 'Перевод'}), encoding='utf-8')
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'backends': ['majority', 'tfidf'], 'train_end': '2025-02-01',
        'validation_end': '2025-03-01', 'bootstrap_repetitions': 100, 'data_kind': 'synthetic'}), encoding='utf-8')
    output = tmp_path / 'study'
    assert run_study(source, config, categories, output)
    comparison = pd.read_excel(output / 'tables' / 'comparison.xlsx', sheet_name='metrics')
    assert comparison.status.tolist() == ['ok', 'ok']
    a = pd.read_excel(output / 'data' / 'majority' / 'predictions.xlsx')
    b = pd.read_excel(output / 'data' / 'tfidf' / 'predictions.xlsx')
    assert a.digest.tolist() == b.digest.tolist()
    assert len(pd.read_excel(output / 'tables' / 'comparison.xlsx', sheet_name='paired_intervals')) == 1
    with pytest.raises(FileExistsError):
        run_study(source, config, categories, output)
