"""Test suite for benchmark specification classes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from extraction.runner import _flatten_eval_summary_metrics


@pytest.mark.parametrize(
    "summary,expected",
    [
        pytest.param(
            {
                "ground_truth_path": "/tmp/gt.json",
                "n_documents": 3,
                "geology": {"layer_f1": 0.63, "tp": 10},
                "metadata": {"name_f1": 0.9},
                "artifacts": {"document_level_metrics_csv": "/tmp/a.csv"},
            },
            {"geology/layer_f1": 0.63, "geology/tp": 10.0, "metadata/name_f1": 0.9},
            id="flattens-geology-and-metadata-only",
        ),
        pytest.param(
            {
                "geology": {"a": {"b": {"c": 1}}},
                "metadata": {},
            },
            {"geology/a/b/c": 1.0},
            id="deep-nesting-under-geology",
        ),
        pytest.param(
            {
                "geology": {},
                "metadata": {"x": 1, "y": 2.5},
            },
            {"metadata/x": 1.0, "metadata/y": 2.5},
            id="flat-metadata-dict",
        ),
        pytest.param(
            {
                "geology": {"layer_f1": 0.63},
                # metadata key missing entirely should be fine
            },
            {"geology/layer_f1": 0.63},
            id="missing-metadata-key-is-ok",
        ),
        pytest.param(
            {
                "geology": None,
                "metadata": None,
            },
            {},
            id="non-mapping-subtrees-are-ignored",
        ),
    ],
)
def test_flatten_eval_summary_metrics_happy_paths(summary: Mapping[str, Any], expected: dict[str, float]) -> None:
    """Test that _flatten_eval_summary_metrics works on various happy-path inputs.

    Args:
        summary: The input evaluation summary dictionary.
        expected: The expected flattened metrics dictionary.
    """
    assert _flatten_eval_summary_metrics(summary) == expected


@pytest.mark.parametrize(
    "summary,expected",
    [
        pytest.param(
            {
                "geology": {"a": None, "b": {"c": None, "d": 1}},
                "metadata": {},
            },
            {"geology/b/d": 1.0},
            id="nones-are-dropped",
        ),
        pytest.param(
            {
                "geology": {},
                "metadata": {"a": "1.23", "b": {"c": "2", "d": " 3.0 "}},
            },
            {"metadata/a": 1.23, "metadata/b/c": 2.0, "metadata/b/d": 3.0},
            id="numeric-strings-are-coerced",
        ),
        pytest.param(
            {
                "geology": {"a": "NaN", "b": "inf", "c": "-inf"},
                "metadata": {},
            },
            {"geology/a": float("nan"), "geology/b": float("inf"), "geology/c": float("-inf")},
            id="special-floats-are-currently-accepted",
        ),
        pytest.param(
            {
                "geology": {},
                "metadata": {"a": "abc", "b": {"c": "1.0m", "d": ""}, "e": []},
            },
            {},
            id="non-numeric-strings-and-non-supported-types-are-ignored",
        ),
        pytest.param(
            {
                "geology": {"a": True, "b": False, "c": 1},
                "metadata": {},
            },
            {"geology/c": 1.0},
            id="bools-are-ignored",
        ),
        pytest.param(
            {
                "ground_truth_path": "/tmp/gt.json",
                "n_documents": 7,
                "artifacts": {"x": 1, "y": "2.0"},
                "geology": {},
                "metadata": {},
            },
            {},
            id="top-level-non-metric-fields-are-ignored",
        ),
    ],
)
def test_flatten_eval_summary_metrics_filters_and_coercion(
    summary: Mapping[str, Any], expected: dict[str, float]
) -> None:
    """Test that _flatten_eval_summary_metrics correctly filters and coerces values.

    Args:
        summary: The input evaluation summary dictionary.
        expected: The expected flattened metrics dictionary after filtering and coercion.
    """
    out = _flatten_eval_summary_metrics(summary)

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
