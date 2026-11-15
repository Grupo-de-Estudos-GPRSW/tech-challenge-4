"""Auditoria real, por etapa, sem imprimir transcrições ou credenciais.

As funções de referência são extraídas sem executar comandos de instalação.
Saídas privadas ficam em outputs/real-parity (ignorado pelo Git).
"""

import argparse
import ast
import hashlib
import io
import json
import logging
import os
import shutil
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "real-parity"


def error_info(exc):
    response = getattr(exc, "response", None)
    return {"error_type": type(exc).__name__, "http_status": getattr(response, "status_code", None)}


def check_access(settings):
    from huggingface_hub import hf_hub_download

    try:
        hf_hub_download(
            "pyannote/speaker-diarization-community-1",
            "config.yaml",
            token=settings.hf_token.get_secret_value(),
        )
        return {"config_access": True}
    except Exception as exc:
        return {"config_access": False, **error_info(exc)}


def run(audio):
    import librosa
    import numpy as np
    import pandas as pd
    import soundfile as sf
    import torch

    from clinical_audio.audio import decode_upload
    from clinical_audio.backends import LocalModels
    from clinical_audio.config import Settings
    from clinical_audio.models import ConsultationTextEmotions, SegmentTextEmotion
    from clinical_audio.reports import notebook_event_rows, notebook_segment_rows
    from clinical_audio.rules import EMOTION_MAP, SOUND_MAP
    from clinical_audio.verification import integral_equivalence_verified

    torch.set_num_threads(2)
    settings = Settings.from_env().model_copy(
        update={"language": "en", "whisper_model": "base", "openai_model": "gpt-4o-mini"}
    )
    info = sf.info(audio)
    report = {
        "audio_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
        "audio": {
            "seconds": info.duration,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "subtype": info.subtype,
        },
        "versions": {
            name: version(name)
            for name in ["torch", "transformers", "librosa", "openai", "pyannote.audio"]
        },
        "ffmpeg_available": bool(shutil.which("ffmpeg")),
        "stages": {},
    }
    OUT.mkdir(parents=True, exist_ok=True)

    def stage(name, action):
        print(f"Iniciando: {name}", flush=True)
        start = time.monotonic()
        try:
            data = action()
            report["stages"][name] = {"status": "completed", **data}
        except Exception as exc:
            report["stages"][name] = {"status": "failed", **error_info(exc)}
        report["stages"][name]["elapsed_s"] = round(time.monotonic() - start, 2)
        (OUT / "verification.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps({name: report["stages"][name]}, ensure_ascii=False), flush=True)

    reference_data = json.loads(
        (ROOT / "scripts" / "fixtures" / "analysis_reference.json").read_text(encoding="utf-8")
    )
    names = {
        "detect_clinical_non_verbal_sounds",
        "transcribe_clinical_audio",
        "run_real_diarization",
        "get_patient_speaker_id_acoustically",
        "filter_patient_segments",
        "extract_prosodic_features",
        "analyze_voice_emotions",
        "analyze_text_emotion_with_openai",
        "calculate_discordance_locally",
        "run_text_and_discordance_analysis",
    }
    nodes = []
    for cell in reference_data["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and not source.startswith("!"):
            nodes.extend(
                n
                for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name in names
            )
    namespace = {
        "np": np,
        "pd": pd,
        "torch": torch,
        "librosa": librosa,
        "Path": Path,
        "List": list,
        "Dict": dict,
        "Any": Any,
        "SAMPLE_RATE": 16000,
        "OPENAI_MODEL": "gpt-4o-mini",
        "json": json,
        "CLINICAL_SOUND_MAP": SOUND_MAP,
        "EMOTION_TRANSLATION_MAP": EMOTION_MAP,
        "ConsultationTextEmotions": ConsultationTextEmotions,
        "SegmentTextEmotion": SegmentTextEmotion,
    }
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), "<analysis-reference-functions>", "exec"),
        namespace,
    )
    samples = decode_upload(audio.name, audio.read_bytes(), settings, notebook_resampling=True)
    reference_samples, _ = librosa.load(str(audio), sr=16000, mono=True)
    stage(
        "preprocessing",
        lambda: {
            "exact_equal": bool(np.array_equal(samples, reference_samples)),
            "max_abs_error": float(np.max(np.abs(samples - reference_samples))),
        },
    )
    backend = LocalModels(settings)
    state = {}

    def whisper_stage():
        state["transcription"] = backend.transcribe(samples)
        (OUT / "transcription-app.json").write_text(
            json.dumps(
                [s.model_dump() for s in state["transcription"]], ensure_ascii=False, indent=2
            ),
            encoding="utf-8",
        )
        namespace["whisper_model"] = backend.whisper
        reference = namespace["transcribe_clinical_audio"](audio)
        actual = [
            {"segment_id": s.id, "inicio_s": s.start, "fim_s": s.end, "transcricao": s.text}
            for s in state["transcription"]
        ]
        state["reference_transcription"] = reference
        return {"app_segments": len(actual), "exact_equal": actual == reference}

    stage("whisper", whisper_stage)
    # A falha do carregador de arquivos não impede auditar outras etapas reais.
    if "transcription" in state and "reference_transcription" not in state:
        report["stages"]["whisper"]["app_segments"] = len(state["transcription"])
        report["stages"]["whisper"]["reference_blocked_missing_ffmpeg"] = not report[
            "ffmpeg_available"
        ]

    def ast_stage():
        namespace["ast_classifier"] = backend.ast
        frame = namespace["detect_clinical_non_verbal_sounds"](audio)
        state["reference_events"] = frame
        events = []
        size, hop = int(1.2 * 16000), int(0.6 * 16000)
        from clinical_audio.models import Event

        for offset in range(0, max(1, len(samples) - size + 1), hop):
            clip = samples[offset : min(len(samples), offset + size)]
            rms = float(np.sqrt(np.mean(clip**2)))
            if rms < 0.015:
                continue
            predictions = backend.classify_events(clip)
            speech = sum(
                p["score"]
                for p in predictions
                if p["label"] in ("Speech", "Conversation", "Narration, monologue")
            )
            for pred in predictions:
                if pred["label"] in SOUND_MAP and pred["score"] >= 0.12:
                    event = Event(
                        start=round(offset / 16000, 2),
                        end=round(min(len(samples), offset + size) / 16000, 2),
                        label=SOUND_MAP[pred["label"]],
                        score=round(float(pred["score"]), 3),
                        rms=round(rms, 4),
                        label_en=pred["label"],
                        speech_score=round(speech, 3),
                    )
                    if (
                        events
                        and event.label == events[-1].label
                        and event.start <= events[-1].end + 0.35
                    ):
                        previous = events[-1]
                        events[-1] = previous.model_copy(
                            update={
                                "end": max(previous.end, event.end),
                                "score": max(previous.score, event.score),
                                "rms": max(previous.rms, event.rms),
                            }
                        )
                    else:
                        events.append(event)
                    break
        from clinical_audio.models import AnalysisResult

        actual = pd.DataFrame(
            notebook_event_rows(
                AnalysisResult(mode="real", duration_s=info.duration, segments=[], events=events)
            )
        )
        equal = (
            frame.empty and actual.empty or frame.to_dict("records") == actual.to_dict("records")
        )
        state["events"] = events
        return {"reference_events": len(frame), "app_events": len(actual), "exact_equal": equal}

    stage("ast", ast_stage)

    def diarization_stage():
        state["turns"] = backend.diarize(samples)
        namespace["diarization_pipeline"] = backend.diarizer
        reference = namespace["run_real_diarization"](audio)
        actual = [
            {"inicio_s": t.start, "fim_s": t.end, "speaker_raw": t.speaker} for t in state["turns"]
        ]
        state["reference_turns"] = reference
        return {"turns": len(actual), "exact_equal": actual == reference}

    stage("diarization", diarization_stage)

    def ser_stage():
        # Carregamentos independentes revelam cabeças inicializadas aleatoriamente.
        from transformers import pipeline
        from transformers.utils import logging as hf_logging

        buffer = io.StringIO()
        handler = logging.StreamHandler(buffer)
        logger = hf_logging.get_logger()
        logger.addHandler(handler)
        try:
            first = backend.ser
            second = pipeline("audio-classification", model=backend.ser_name, device=backend.device)
        finally:
            logger.removeHandler(handler)
        warnings = buffer.getvalue()
        has_new_weights = "newly initialized" in warnings or "were not initialized" in warnings
        clip = samples[: 5 * 16000]
        first_prediction = first({"raw": clip, "sampling_rate": 16000}, top_k=3)
        second_prediction = second({"raw": clip, "sampling_rate": 16000}, top_k=3)
        state["reference_ser"] = second
        namespace["ser_pipeline"] = second
        return {
            "probe_seconds": 5,
            "new_weights_warning": has_new_weights,
            "unused_weights_warning": "were not used" in warnings,
            "independent_load_predictions_equal": first_prediction == second_prediction,
            "app_top_label": first_prediction[0]["label"],
            "reference_top_label": second_prediction[0]["label"],
            "app_top_score": float(first_prediction[0]["score"]),
            "reference_top_score": float(second_prediction[0]["score"]),
        }

    stage("ser_loading_and_probe", ser_stage)

    def full_stage():
        from openai import OpenAI

        from clinical_audio.pipeline import AnalysisPipeline
        from clinical_audio.semantic import OpenAISemantic

        semantic = OpenAISemantic(settings) if settings.openai_key.get_secret_value() else None
        namespace["client"] = (
            OpenAI(api_key=settings.openai_key.get_secret_value()) if semantic else None
        )
        actual = AnalysisPipeline(settings, backend).analyze(samples, semantic=semantic)
        selected = namespace["filter_patient_segments"](
            state["reference_turns"], state["reference_transcription"]
        )
        voice = namespace["analyze_voice_emotions"](audio, selected, state["reference_events"])
        expected = namespace["run_text_and_discordance_analysis"](voice)
        actual_rows = notebook_segment_rows(actual)
        (OUT / "analysis-app.json").write_text(actual.model_dump_json(indent=2), encoding="utf-8")
        expected.to_json(
            OUT / "analysis-reference.json", orient="records", force_ascii=False, indent=2
        )
        rows_equal = pd.DataFrame(actual_rows).to_dict("records") == expected.to_dict("records")
        events_equal = notebook_event_rows(actual) == state["reference_events"].to_dict("records")
        return {
            "patient_segments_app": len(actual_rows),
            "patient_segments_reference": len(expected),
            "exact_equal": rows_equal and events_equal,
            "events_equal": events_equal,
            "rows_equal": rows_equal,
            "openai_enabled": semantic is not None,
        }

    prerequisites = [
        "reference_transcription",
        "reference_turns",
        "reference_ser",
        "reference_events",
    ]
    if all(key in state for key in prerequisites):
        stage("full_pipeline", full_stage)
    else:
        report["stages"]["full_pipeline"] = {
            "status": "blocked",
            "missing_prerequisites": [key for key in prerequisites if key not in state],
            "openai_called": False,
        }
    report["integral_equivalence_verified"] = integral_equivalence_verified(report["stages"])
    (OUT / "verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Relatório: outputs/real-parity/verification.json", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--access-only", action="store_true")
    parser.add_argument("--diarization-only", action="store_true")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    if not shutil.which("ffmpeg"):
        binaries = list((OUT / "tools" / "imageio_ffmpeg" / "binaries").glob("*.exe"))
        if binaries:
            bindir = OUT / "bin"
            bindir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(binaries[0], bindir / "ffmpeg.exe")
            os.environ["PATH"] = str(bindir) + os.pathsep + os.environ.get("PATH", "")
    if args.diarization_only:
        import traceback

        import torch

        from clinical_audio.audio import decode_upload
        from clinical_audio.backends import LocalModels
        from clinical_audio.config import Settings

        torch.set_num_threads(2)
        settings = Settings.from_env()
        OUT.mkdir(parents=True, exist_ok=True)
        try:
            samples = decode_upload(
                args.audio.name, args.audio.read_bytes(), settings, notebook_resampling=True
            )
            backend = LocalModels(settings)
            turns = backend.diarize(samples)
            result = {"status": "completed", "turns": len(turns)}
        except Exception as exc:
            detail = traceback.format_exc()
            for secret in [
                settings.hf_token.get_secret_value(),
                settings.openai_key.get_secret_value(),
            ]:
                if secret:
                    detail = detail.replace(secret, "[REDACTED]")
            (OUT / "diarization-error.txt").write_text(detail, encoding="utf-8")
            result = {"status": "failed", **error_info(exc), "detail": detail.splitlines()[-1]}
        (OUT / "diarization-diagnostic.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False), flush=True)
    elif args.access_only:
        from clinical_audio.config import Settings

        result = check_access(Settings.from_env())
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "access.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result), flush=True)
    else:
        run(args.audio)


if __name__ == "__main__":
    main()
