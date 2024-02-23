from src.utils.metrics import precision_at_k


def test_precision():
    assert precision_at_k([1, 0, 1, 1, 0, 1, 0, 0, 0, 1], 5) == 0.6
