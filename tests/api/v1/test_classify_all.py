"""Tests for the unified `/classify` endpoint.

These tests exercise three real, published BERT models downloaded from HuggingFace once per test
session.

- `lithology` (consolidated, single-label) — also drives the consolidated/unconsolidated branch.
- `cementation` (consolidated, single-label) — a second, independently fine-tuned head.
- `grain_angularity` (unconsolidated, multi-label) — covers the multi-label response format.

Using three separate checkpoints lets us verify the core assumption behind the production two-step design:
the shared backbone is frozen and identical across every fine-tuned checkpoint, so one model's embedding
can be safely fed into a different model's task head-

The real-model tests are skipped (not failed) if the models can't be downloaded, so a run without
network access does not break CI.
"""

import logging

import pytest
from fastapi.testclient import TestClient

from app.api.v1.endpoints.classify_all import classify
from app.common.schemas import ClassificationConsolidationClasses, ClassifyRequest
from app.main import app
from classification.utils.datasets import ExistingClassificationSystems
from classification.utils.datasets.cementation import CementationSystem
from classification.utils.datasets.grain_angularity import GrainAngularitySystem
from classification.utils.datasets.lithology import LithologySystem

#   lithology=limestone, cementation=strongly_cemented, grain_angularity=not_specified
CONSOLIDATED_DESCRIPTION = "Kalkstein, hellgrau, geschichtet, feinkoernig, stark verkittet"
#   lithology=unconsolidated, cementation=not_specified, grain_angularity=[angular, sub_angular]
UNCONSOLIDATED_DESCRIPTION = "Siltiger Kies mit Sand und Steinen, braun, eckig bis kantengerundet"
#   lithology=sandstone (consolidated), cementation=not_specified — cementation IS a relevant,
#   loaded task here, but predicts only not_specified.
CONSOLIDATED_NOT_SPECIFIED_DESCRIPTION = "Sandstein, grau"

_REAL_TASK_NAMES = ("lithology", "cementation", "grain_angularity")


@pytest.fixture(scope="session")
def real_bert_models():
    """Load three real, published BERT models once for the whole test session.

    Skips rather than fails if the models can't be reached, so a run without
    network access doesn't break.
    """
    pytest.importorskip("torch")
    from classification.models.model import BertModel

    try:
        return {
            task_name: BertModel(
                f"swissgeol/{task_name}", ExistingClassificationSystems.get_classification_system_type(task_name)
            )
            for task_name in _REAL_TASK_NAMES
        }
    except Exception as exc:  # only hit without network/HuggingFace access
        pytest.skip(f"Could not download real BERT models from HuggingFace: {exc}")


@pytest.fixture
def bert_models_state():
    """Restore app.state.bert_models after the test, regardless of what the test sets it to."""
    original = getattr(app.state, "bert_models", None)
    yield
    app.state.bert_models = original


@pytest.mark.parametrize("task_name", _REAL_TASK_NAMES)
def test_two_step_inference_matches_single_pass_prediction(real_bert_models, task_name):
    """compute_shared_embedding + predict_from_embedding must reproduce predict_class exactly.

    Exercises the actual layer-11 split and tensor handling in BertModel against real weights,
    for both a single-label and a multi-label real classification system.
    """
    model = real_bert_models[task_name]
    for description in (CONSOLIDATED_DESCRIPTION, UNCONSOLIDATED_DESCRIPTION):
        single_pass = [c.name for c in model.predict_class(description)]
        hidden_states, extended_mask = model.compute_shared_embedding(description)
        two_step = [c.name for c in model.predict_from_embedding(hidden_states, extended_mask)]
        assert two_step == single_pass


def test_cross_model_shared_embedding_reuse_matches_own_prediction(real_bert_models):
    """Feeding one model's embedding into a different model's head must match that head's own prediction.

    This validates the core assumption behind the production two-step design in classify_all.py:
    since the backbone is frozen and identical across all fine-tuned checkpoints, the lithology
    model's embedding can be safely reused by any other task head's classifier, instead of each
    head recomputing its own backbone pass.
    """
    lithology_model = real_bert_models["lithology"]
    for description in (CONSOLIDATED_DESCRIPTION, UNCONSOLIDATED_DESCRIPTION):
        hidden_states, extended_mask = lithology_model.compute_shared_embedding(description)
        for task_name in ("cementation", "grain_angularity"):
            model = real_bert_models[task_name]
            own_prediction = [c.name for c in model.predict_class(description)]
            cross_prediction = [c.name for c in model.predict_from_embedding(hidden_states, extended_mask)]
            assert cross_prediction == own_prediction


def test_classify_consolidated_rock_description(real_bert_models):
    """A real consolidated-rock description runs the lithology and cementation heads (both loaded)."""
    response = classify(ClassifyRequest(description=CONSOLIDATED_DESCRIPTION), real_bert_models)

    assert response.consolidation == ClassificationConsolidationClasses.consolidated
    assert response.lithology == LithologySystem.LithologyClasses.limestone
    assert response.cementation == CementationSystem.CementationClasses.strongly_cemented
    # alteration_degree_consolidated/color/mineral_components/accessory_components are also
    # consolidated tasks but aren't loaded here, so they're skipped as not-loaded.
    assert response.alteration_degree_consolidated is None
    assert response.color is None
    assert response.mineral_components is None
    assert response.accessory_components is None


def test_classify_unconsolidated_sediment_description(real_bert_models, caplog):
    """A real unconsolidated description switches the branch; only grain_angularity is loaded for it."""
    with caplog.at_level(logging.WARNING):
        response = classify(ClassifyRequest(description=UNCONSOLIDATED_DESCRIPTION), real_bert_models)

    assert response.consolidation == ClassificationConsolidationClasses.unconsolidated
    assert response.grain_angularity == [
        GrainAngularitySystem.GrainAngularityClasses.angular,
        GrainAngularitySystem.GrainAngularityClasses.sub_angular,
    ]
    # lithology/cementation aren't part of the unconsolidated task set; en_main/uscs/debris/color/
    # grain_shape/organic_components aren't loaded here, so only grain_angularity comes back.
    assert response.lithology is None
    assert response.cementation is None
    assert response.en_main is None
    assert any("not loaded" in message for message in caplog.messages)


def test_post_classify_returns_503_when_bert_models_not_loaded(test_client: TestClient, bert_models_state):
    """The router must reject with 503 when BERT_ENABLED=false left bert_models unset."""
    app.state.bert_models = None

    request = ClassifyRequest(description=CONSOLIDATED_DESCRIPTION)
    response = test_client.post("/api/V1/classify", content=request.model_dump_json())

    assert response.status_code == 503
    assert "BERT_ENABLED" in response.json()["detail"]


def test_post_classify_returns_predictions_when_bert_models_loaded(
    test_client: TestClient, bert_models_state, real_bert_models
):
    """A full round-trip through the router dispatches into classify() and returns real predictions."""
    app.state.bert_models = real_bert_models

    request = ClassifyRequest(description=CONSOLIDATED_DESCRIPTION)
    response = test_client.post("/api/V1/classify", content=request.model_dump_json())

    assert response.status_code == 200
    body = response.json()
    assert body["lithology"] == "limestone"
    assert body["cementation"] == "strongly_cemented"
    assert "color" not in body  # color is now omitted, not null


def test_post_classify_omits_field_that_ran_but_predicted_not_specified(
    test_client: TestClient, bert_models_state, real_bert_models
):
    """A relevant, loaded task that predicts only not_specified must be omitted, not null.

    Unlike `color` above (omitted because it's not loaded at all), `cementation` here is loaded
    and relevant to the consolidated branch, runs, and predicts not_specified.
    """
    app.state.bert_models = real_bert_models

    request = ClassifyRequest(description=CONSOLIDATED_NOT_SPECIFIED_DESCRIPTION)
    response = test_client.post("/api/V1/classify", content=request.model_dump_json())

    assert response.status_code == 200
    body = response.json()
    assert body["lithology"] == "sandstone"
    assert "cementation" not in body
