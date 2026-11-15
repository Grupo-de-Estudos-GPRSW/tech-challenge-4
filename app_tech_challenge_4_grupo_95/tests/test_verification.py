from copy import deepcopy

import pytest

from clinical_audio.verification import integral_equivalence_verified, ser_checkpoint_verified


@pytest.fixture
def complete_report():
    # Relatório fictício para testar o critério, sem inferência, rede ou certificados reais.
    stages = {
        name: {"status": "completed", "exact_equal": True}
        for name in ("preprocessing", "whisper", "ast", "diarization", "full_pipeline")
    }
    stages["full_pipeline"].update(
        events_equal=True, openai_enabled=True, patient_segments_app=1, patient_segments_reference=1
    )
    stages["ser_loading_and_probe"] = {
        "status": "completed",
        "new_weights_warning": False,
        "unused_weights_warning": False,
        "independent_load_predictions_equal": True,
    }
    return stages


def test_only_complete_compared_inference_can_pass(complete_report):
    before = deepcopy(complete_report)
    assert integral_equivalence_verified(complete_report)
    assert complete_report == before


@pytest.mark.parametrize(
    "stage", ["preprocessing", "whisper", "ast", "diarization", "full_pipeline"]
)
@pytest.mark.parametrize("failure", ["missing", "blocked", "different"])
def test_final_output_alone_cannot_certify_missing_or_different_stages(
    complete_report, stage, failure
):
    if failure == "missing":
        complete_report.pop(stage)
    elif failure == "blocked":
        complete_report[stage]["status"] = "blocked"
    else:
        complete_report[stage]["exact_equal"] = False
    assert not integral_equivalence_verified(complete_report)


@pytest.mark.parametrize("field", ["new_weights_warning", "unused_weights_warning"])
@pytest.mark.parametrize("evidence", [True, None, "false"])
def test_ser_warnings_or_unknown_loading_evidence_block_certificate(
    complete_report, field, evidence
):
    complete_report["ser_loading_and_probe"][field] = evidence
    assert not integral_equivalence_verified(complete_report)


@pytest.mark.parametrize(
    "condition",
    [
        "missing_ser",
        "failed_ser",
        "different_ser",
        "missing_warning",
        "no_openai",
        "different_events",
    ],
)
def test_incomplete_real_audits_and_neutral_fallback_cannot_certify(complete_report, condition):
    if condition == "missing_ser":
        complete_report.pop("ser_loading_and_probe")
    elif condition == "failed_ser":
        complete_report["ser_loading_and_probe"]["status"] = "failed"
    elif condition == "different_ser":
        complete_report["ser_loading_and_probe"]["independent_load_predictions_equal"] = False
    elif condition == "missing_warning":
        complete_report["ser_loading_and_probe"].pop("unused_weights_warning")
    elif condition == "no_openai":
        complete_report["full_pipeline"]["openai_enabled"] = False
    else:
        complete_report["full_pipeline"]["events_equal"] = False
    assert not integral_equivalence_verified(complete_report)


def test_actual_audit_patterns_remain_not_verified(complete_report):
    assert not ser_checkpoint_verified({})
    complete_report["diarization"] = {"status": "failed", "error_type": "RuntimeError"}
    complete_report["full_pipeline"] = {"status": "blocked", "openai_called": False}
    complete_report["ser_loading_and_probe"].update(
        new_weights_warning=True, independent_load_predictions_equal=False
    )
    assert not integral_equivalence_verified(complete_report)


@pytest.mark.parametrize("counts", [(None, None), (0, 0), (1, 2), (True, True)])
def test_empty_or_unknown_patient_selection_cannot_certify(complete_report, counts):
    app_count, reference_count = counts
    complete_report["full_pipeline"].update(
        patient_segments_app=app_count, patient_segments_reference=reference_count
    )
    assert not integral_equivalence_verified(complete_report)
