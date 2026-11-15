"""Exportações em memória, sem salvar consultas no servidor."""

import csv
import io

from clinical_audio.models import AnalysisResult


def segment_rows(result: AnalysisResult) -> list[dict]:
    semantic = {s.id: s for s in result.text_analysis.segments} if result.text_analysis else {}
    rows = []
    for segment in result.segments:
        text = semantic.get(segment.id)
        rows.append(
            {
                **segment.model_dump(),
                "nearby_events": ", ".join(segment.nearby_events),
                "text_emotion": text.emotion if text else "Não avaliada",
                "discordance": text.discordance if text else "Não avaliada",
                "has_discordance": text.discordance in {"Leve", "Moderada", "Alta"}
                if text
                else False,
                "explanation": text.explanation
                if text
                else "Análise textual indisponível ou desabilitada.",
            }
        )
    return rows


def csv_bytes(result: AnalysisResult) -> bytes:
    rows = segment_rows(result)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=list(rows[0]) if rows else ["id", "text", "discordance"]
    )
    writer.writeheader()
    for row in rows:
        # Evita execução de fórmulas quando a transcrição é aberta no Excel.
        writer.writerow(
            {
                k: "'" + v
                if isinstance(v, str) and v.lstrip().startswith(("=", "+", "-", "@"))
                else v
                for k, v in row.items()
            }
        )
    return buffer.getvalue().encode("utf-8-sig")


def notebook_segment_rows(result: AnalysisResult) -> list[dict]:
    return [
        {
            "segment_id": r["id"],
            "inicio_s": r["start"],
            "fim_s": r["end"],
            "transcricao": r["text"],
            "emocao_voz_primaria": r["voice_emotion"],
            "confianca_voz_primaria": r["voice_score"],
            "emocao_voz_secundaria": r["secondary_emotion"],
            "confianca_voz_secundaria": r["secondary_score"],
            "pitch_medio_hz": r["pitch_mean"],
            "pitch_std_hz": r["pitch_std"],
            "energia_rms": r["rms"],
            "eventos_nao_verbais_proximos": next(
                s.nearby_events for s in result.segments if s.id == r["id"]
            ),
            "emocao_discurso_texto": r["text_emotion"],
            "interlocutor": "Paciente",
            "ha_discordancia": r["has_discordance"],
            "nivel_discordancia": r["discordance"],
            "explicacao_clinica": r["explanation"],
        }
        for r in segment_rows(result)
    ]


def notebook_csv_bytes(result: AnalysisResult) -> bytes:
    import pandas as pd

    return pd.DataFrame(notebook_segment_rows(result)).to_csv(index=False).encode("utf-8-sig")


def markdown_report(result: AnalysisResult) -> str:
    import pandas as pd

    from clinical_audio.notebook_v2 import generate_clinical_discordance_report

    rows = notebook_segment_rows(result)
    full = pd.DataFrame(rows)
    if not rows:
        full = pd.DataFrame(columns=["ha_discordancia"])
    events = pd.DataFrame(
        [
            {"inicio_s": e.start, "fim_s": e.end, "evento_clinico": e.label, "confianca": e.score}
            for e in result.events
        ]
    )
    return generate_clinical_discordance_report(full, events)


def notebook_event_rows(result: AnalysisResult) -> list[dict]:
    return [
        {
            "inicio_s": e.start,
            "fim_s": e.end,
            "evento_en": e.label_en,
            "evento_clinico": e.label,
            "confianca": e.score,
            "energia_rms": e.rms,
            "prob_fala_simultanea": e.speech_score,
        }
        for e in result.events
    ]
