"""Verifica o pré-requisito SER com áudio real antes da auditoria integral custosa."""

import argparse
import ast
import hashlib
import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Any


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    args = parser.parse_args()
    # Todos os pesos deste checkpoint já foram obtidos na auditoria anterior.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    print("Carregando bibliotecas locais para auditar o SER...", flush=True)
    import librosa
    import numpy as np
    import pandas as pd
    import soundfile as sf
    import torch
    from transformers import pipeline
    from transformers.utils import logging as hf_logging

    from clinical_audio.audio import decode_upload
    from clinical_audio.backends import LocalModels
    from clinical_audio.config import SER_MODEL_NAME, Settings
    from clinical_audio.verification import ser_checkpoint_verified

    torch.set_num_threads(2)
    start = time.monotonic()
    root = Path(__file__).resolve().parents[1]
    out = root / "outputs" / "real-parity" / args.audio.stem
    out.mkdir(parents=True, exist_ok=True)
    settings = Settings()
    info = sf.info(args.audio)
    samples = decode_upload(
        args.audio.name, args.audio.read_bytes(), settings, notebook_resampling=True
    )
    expected_samples, _ = librosa.load(str(args.audio), sr=16000, mono=True)
    result = {
        "audio": {
            "name": args.audio.name,
            "sha256": hashlib.sha256(args.audio.read_bytes()).hexdigest(),
            "seconds": info.duration,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "subtype": info.subtype,
        },
        "checkpoint": SER_MODEL_NAME,
        "preprocessing_equal": bool(np.array_equal(samples, expected_samples)),
        "ser_probes": [],
        "openai_called": False,
        "integral_equivalence_verified": False,
    }

    def save():
        (out / "checkpoint-verification.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    save()
    print(
        json.dumps(
            {"audio": result["audio"], "preprocessing_equal": result["preprocessing_equal"]}
        ),
        flush=True,
    )
    reference_data = json.loads(
        (root / "scripts" / "fixtures" / "analysis_reference.json").read_text(encoding="utf-8")
    )
    namespace = {
        "np": np,
        "pd": pd,
        "librosa": librosa,
        "Path": Path,
        "List": list,
        "Dict": dict,
        "Any": Any,
        "SAMPLE_RATE": 16000,
    }
    nodes = []
    for cell in reference_data["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] != "code" or source.startswith("!"):
            continue
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef) and node.name in {
                "extract_prosodic_features",
                "analyze_voice_emotions",
            }:
                nodes.append(node)
            if (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == "EMOTION_TRANSLATION_MAP"
            ):
                namespace["EMOTION_TRANSLATION_MAP"] = ast.literal_eval(node.value)
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), "<analysis-reference>", "exec"), namespace
    )
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    logger = hf_logging.get_logger()
    logger.addHandler(handler)
    backend = LocalModels(settings)
    try:
        print("Carregando o checkpoint duas vezes, como execuções independentes...", flush=True)
        _ = backend.ser  # carregamento do aplicativo
        reference_ser = pipeline(
            "audio-classification", model=SER_MODEL_NAME, device=backend.device
        )
    except Exception as exc:
        result["load_error_type"] = type(exc).__name__
        save()
        raise
    finally:
        logger.removeHandler(handler)
    log = buffer.getvalue()
    result["new_weights_warning"] = "newly initialized" in log or "were not initialized" in log
    result["unused_weights_warning"] = "were not used" in log
    namespace["ser_pipeline"] = reference_ser
    save()
    points = [0.0, round(info.duration / 2, 2), round(max(0, info.duration - 5), 2)]
    fields = {
        "voice_emotion": "emocao_voz_primaria",
        "voice_score": "confianca_voz_primaria",
        "secondary_emotion": "emocao_voz_secundaria",
        "secondary_score": "confianca_voz_secundaria",
        "pitch_mean": "pitch_medio_hz",
        "pitch_std": "pitch_std_hz",
        "rms": "energia_rms",
    }
    for index, begin in enumerate(points, 1):
        end = min(info.duration, begin + 5)
        print(f"Comparando trecho real {index}/3: {begin:.2f}–{end:.2f}s", flush=True)
        actual = backend.voice_features(samples[int(begin * 16000) : int(end * 16000)])
        expected = namespace["analyze_voice_emotions"](
            args.audio,
            [{"segment_id": index, "inicio_s": begin, "fim_s": end, "transcricao": ""}],
            pd.DataFrame(),
        )[0]
        mismatched = [key for key, native in fields.items() if actual[key] != expected[native]]
        probe = {
            "start": begin,
            "end": end,
            "app": actual,
            "notebook": {key: expected[native] for key, native in fields.items()},
            "different_fields": mismatched,
            "exact_equal": not mismatched,
        }
        result["ser_probes"].append(probe)
        save()
        print(json.dumps(probe, ensure_ascii=False), flush=True)
    result["checkpoint_reproducible_in_this_run"] = all(
        p["exact_equal"] for p in result["ser_probes"]
    )
    result["elapsed_s"] = round(time.monotonic() - start, 2)
    result["remaining_stages"] = (
        "not_executed_after_failed_ser_prerequisite"
        if not ser_checkpoint_verified(result) or not result["checkpoint_reproducible_in_this_run"]
        else "requires_full_audit"
    )
    save()
    print("Relatório: " + str(out / "checkpoint-verification.json"), flush=True)


if __name__ == "__main__":
    main()
