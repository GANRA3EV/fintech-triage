import pytest

from triage.models import TicketModel


@pytest.mark.parametrize("categories", [[], ["card"], ["card", "card"], ["card", ""], ["card", 1]])
def test_invalid_categories(categories):
    with pytest.raises(ValueError):
        TicketModel(categories)


def test_unknown_backend():
    with pytest.raises(ValueError):
        TicketModel(["card", "transfer"], backend="unknown")


@pytest.mark.parametrize("text", ["", " ", None, 42])
def test_invalid_prediction_text(text):
    with pytest.raises(ValueError):
        TicketModel(["card", "transfer"]).predict([text])


@pytest.mark.parametrize("threshold", [-1, 2, float("nan"), float("inf")])
def test_invalid_prediction_threshold(threshold):
    with pytest.raises(ValueError):
        TicketModel(["card", "transfer"]).predict([], threshold=threshold)


def test_empty_batch():
    assert TicketModel(["card", "transfer"]).predict([]) == []


def test_invalid_secondary_threshold():
    with pytest.raises(ValueError):
        TicketModel(["card", "transfer"]).predict([], secondary_threshold=1.1)
