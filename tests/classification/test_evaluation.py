"""Test per-class metrics and macro-average computation against sklearn."""

import math

import pytest
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer

from classification.evaluation.evaluate import AllClassificationMetrics, per_class_metric


def _sklearn_macro(y_true, y_pred):
    """Computes macro-averaged metrics using the sklearn functions."""
    if not y_true:
        return {"macro_precision": 0, "macro_recall": 0, "macro_f1": 0}

    # Transform to one-hot to match sklearn input
    mlb = MultiLabelBinarizer().fit(y_true + y_pred)
    y_true_bin = mlb.transform(y_true)
    y_pred_bin = mlb.transform(y_pred)

    precision = round(precision_score(y_true_bin, y_pred_bin, average="macro", zero_division=0), 4)
    recall = round(recall_score(y_true_bin, y_pred_bin, average="macro", zero_division=0), 4)
    f1 = round(f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0), 4)

    return {
        "macro_precision": precision if not math.isnan(precision) else 0.0,
        "macro_recall": recall if not math.isnan(recall) else 0.0,
        "macro_f1": f1 if not math.isnan(f1) else 0.0,
    }


@pytest.mark.parametrize(
    "y_true, y_pred",
    [
        pytest.param([], [], id="empty_inputs"),
        pytest.param([[0], [0], [0]], [[0], [1], [1]], id="no_1_in_true"),
        pytest.param([[0], [0], [1]], [[0], [0], [0]], id="no_1_in_pred"),
        pytest.param([[0], [0], [0]], [[0], [0], [0]], id="only_0_class"),
        pytest.param([[0], [0, 1], [1], [1]], [[0], [1], [1], [0]], id="typical_multi_label"),
        pytest.param([[0, 1], [0, 1], [0, 1]], [[0], [1], [1]], id="no_multi_in_true"),
    ],
)
def test_macro_average_matches_sklearn(y_true, y_pred):
    """Test the macro average implemented.

    It should not consider classes with zeros samples in the ground truth and the prediction. In the above examples,
    the classes 2 to 37 of the USCSClasses are not present at all, and will not be considered in the averaging.
    """
    metrics_dict = per_class_metric(y_pred, y_true)
    result = AllClassificationMetrics.compute_macro_average(list(metrics_dict.values()))
    expected = _sklearn_macro(y_true, y_pred)
    assert result == expected
