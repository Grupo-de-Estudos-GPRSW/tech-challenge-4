"""Critérios para a auditoria real; não alteram a inferência do aplicativo."""


def ser_checkpoint_verified(loading: dict) -> bool:
    """Avisos ausentes do relatório são evidência insuficiente, não aprovação."""
    return (
        loading.get("new_weights_warning") is False
        and loading.get("unused_weights_warning") is False
    )


def integral_equivalence_verified(stages: dict) -> bool:
    """Exige inferência completa, SER verificado e chamadas textuais reais comparadas."""
    for name in ("preprocessing", "whisper", "ast", "diarization", "full_pipeline"):
        stage = stages.get(name, {})
        if stage.get("status") != "completed" or stage.get("exact_equal") is not True:
            return False
    ser = stages.get("ser_loading_and_probe", {})
    full = stages["full_pipeline"]
    patient_count = full.get("patient_segments_app")
    return (
        ser.get("status") == "completed"
        and ser_checkpoint_verified(ser)
        and ser.get("independent_load_predictions_equal") is True
        and stages["full_pipeline"].get("events_equal") is True
        and full.get("openai_enabled") is True
        and isinstance(patient_count, int)
        and not isinstance(patient_count, bool)
        and patient_count > 0
        and full.get("patient_segments_reference") == patient_count
    )
