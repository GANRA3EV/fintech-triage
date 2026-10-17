from sklearn.metrics import classification_report, f1_score, recall_score

from triage.routing import review_reasons

URGENCIES = ['low', 'normal', 'high', 'critical']


def choose_threshold(validation, predictions):
    candidates = []
    for threshold in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        accepted = [(truth, result) for truth, result in zip(validation.itertuples(), predictions)
                    if not review_reasons(result['confidence'], result['urgency_confidence'],
                                          result['urgency'], threshold)]
        candidates.append({'threshold': threshold, 'coverage': len(accepted) / len(validation),
            'primary_accuracy': sum(t.primary == p['primary'] for t, p in accepted) / len(accepted) if accepted else None,
            'urgency_accuracy': sum(t.urgency == p['urgency'] for t, p in accepted) / len(accepted) if accepted else None})
    eligible = [r for r in candidates if r['primary_accuracy'] is not None
                and r['primary_accuracy'] >= 0.9 and r['urgency_accuracy'] >= 0.9]
    return (max(eligible, key=lambda r: r['coverage'])['threshold'] if eligible else 1.01), candidates


def evaluate(model, test, predictions):
    predicted = [row['primary'] for row in predictions]
    urgency = [row['urgency'] for row in predictions]
    true_topics = [[r.primary] + ([r.secondary] if isinstance(r.secondary, str) and r.secondary else [])
                   for r in test.itertuples()]
    topics = [[r['primary']] + ([r['secondary']] if r['secondary'] else []) for r in predictions]
    critical = test.urgency.eq('critical').to_numpy()
    return {
        'primary_macro_f1': f1_score(test.primary, predicted, labels=model.categories, average='macro', zero_division=0),
        'topics_macro_f1': f1_score(model.labels.transform(true_topics), model.labels.transform(topics), average='macro', zero_division=0),
        'urgency_macro_f1': f1_score(test.urgency, urgency, labels=URGENCIES, average='macro', zero_division=0),
        'critical_support': int(critical.sum()),
        'critical_recall': recall_score(test.urgency, urgency, labels=['critical'], average='macro', zero_division=0) if critical.any() else None,
        'critical_review_recall': sum(p['operator_review'] for p, flag in zip(predictions, critical) if flag) / int(critical.sum()) if critical.any() else None,
        'review_fraction': sum(r['operator_review'] for r in predictions) / len(predictions),
        'classification_report': classification_report(test.primary, predicted, labels=model.categories, zero_division=0, output_dict=True)}
