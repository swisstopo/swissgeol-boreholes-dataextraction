"""Test suite for benchmark specification classes."""

import pytest

from extraction.runner import _flatten_metrics


@pytest.mark.parametrize(
    "metrics,prefix,expected",
    [
        pytest.param(
            {"geology": {"layer_f1": 0.63, "tp": 10}, "metadata": {"name_f1": 0.9}},
            "",
            {"geology/layer_f1": 0.63, "geology/tp": 10.0, "metadata/name_f1": 0.9},
            id="nested-dicts-basic",
        ),
        pytest.param(
            {"a": {"b": {"c": 1}}},
            "",
            {"a/b/c": 1.0},
            id="deep-nesting",
        ),
        pytest.param(
            {"x": 1, "y": 2.5},
            "",
            {"x": 1.0, "y": 2.5},
            id="flat-dict",
        ),
        pytest.param(
            {"geology": {"layer_f1": 0.63}},
            "root",
            {"root/geology/layer_f1": 0.63},
            id="prefix-is-prepended",
        ),
    ],
)
def test_flatten_metrics_happy_paths(metrics, prefix, expected):
    """Test _flatten_metrics with various happy path scenarios.

    Args:
        metrics (dict): The metrics dictionary to flatten.
        prefix (str): The prefix to prepend to all keys.
        expected (dict): The expected flattened dictionary.
    """
    assert _flatten_metrics(metrics, prefix=prefix) == expected


@pytest.mark.parametrize(
    "metrics,expected",
    [
        pytest.param(
            {"a": None, "b": {"c": None, "d": 1}},
            {"b/d": 1.0},
            id="nones-are-dropped",
        ),
        pytest.param(
            {"a": "1.23", "b": {"c": "2", "d": " 3.0 "}},
            {"a": 1.23, "b/c": 2.0, "b/d": 3.0},
            id="numeric-strings-are-coerced",
        ),
        pytest.param(
            {"a": "NaN", "b": "inf", "c": "-inf"},
            {"a": float("nan"), "b": float("inf"), "c": float("-inf")},
            id="special-floats-are-currently-accepted",
        ),
        pytest.param(
            {"a": "abc", "b": {"c": "1.0m", "d": ""}, "e": []},
            {},
            id="non-numeric-strings-and-non-supported-types-are-ignored",
        ),
        pytest.param(
            {"a": True, "b": False},
            {"a": 1.0, "b": 0.0},
            id="bools-currently-treated-as-ints",
        ),
    ],
)
def test_flatten_metrics_filters_and_coercion(metrics, expected):
    """Test _flatten_metrics with filtering and coercion scenarios.

    Args:
        metrics (dict): The metrics dictionary to flatten.
        expected (dict): The expected flattened dictionary after filtering and coercion.
    """
    out = _flatten_metrics(metrics)
    # For NaN, normal equality doesn't work; handle that case explicitly.
    if any(isinstance(v, float) and (v != v) for v in expected.values()):  # NaN check
        assert out.keys() == expected.keys()
        for k, v in expected.items():
            if isinstance(v, float) and (v != v):  # NaN
                assert out[k] != out[k]
            else:
                assert out[k] == v
    else:
        assert out == expected
