import pytest

from triage.routing import review_reasons


@pytest.mark.parametrize("confidence, urgency_confidence, urgency, expected", [
    (0.9, 0.9, "normal", []),
    (0.6, 0.9, "normal", ["low_category_confidence"]),
    (0.9, 0.6, "normal", ["low_urgency_confidence"]),
    (0.9, 0.9, "critical", ["critical_urgency"]),
    (0.7, 0.7, "normal", []),
    (0.6, 0.6, "critical", ["low_category_confidence", "low_urgency_confidence", "critical_urgency"]),
])
def test_routing_reasons(confidence, urgency_confidence, urgency, expected):
    assert review_reasons(confidence, urgency_confidence, urgency, 0.7) == expected
