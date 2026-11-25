"""Guards per-class recall in the judge scoring seam imported by consumers."""

import pytest

from llm_utils.judge._metrics import scores

np = pytest.importorskip("numpy")
metrics = pytest.importorskip("sklearn.metrics")


@pytest.mark.parametrize("y,predicted,n_labels", [
    ([0, 0, 1, 1, 1], [0, 1, 1, 1, 0], 2),
    ([0, 1, 2, 0, 1, 2], [0, 0, 2, 1, 1, 2], 3),
    ([0, 0, 2, 2], [0, 1, 0, 2], 3),  # label 1 absent from truth
    ([1, 1, 1], [0, 1, 0], 2),
])
def test_recall_matches_sklearn_including_missing_labels(y, predicted, n_labels):
    y = np.asarray(y)
    probabilities = np.full((len(y), n_labels), 0.1 / (n_labels - 1))
    probabilities[np.arange(len(y)), predicted] = 0.9
    result = scores(y, probabilities, n_labels, ece_bins=10)
    expected = metrics.recall_score(y, predicted, labels=list(range(n_labels)),
                                    average=None, zero_division=0)
    assert result["recall_per_class"] == pytest.approx(expected.tolist())
    assert len(result["recall_per_class"]) == n_labels
    # Adding recall leaves the existing scalar metrics intact.
    assert result["accuracy"] == metrics.accuracy_score(y, predicted)
    assert result["log_loss"] == metrics.log_loss(y, probabilities, labels=list(range(n_labels)))
