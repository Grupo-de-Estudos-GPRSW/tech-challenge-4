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


def markdown_report(result: AnalysisResult) -> str:
    lines = [
        "# Relatório acadêmico de áudio e texto",
        "",
        f"Modo: {result.mode}",
        f"Duração: {result.duration_s:.1f} segundos",
        "",
        "## Limitações",
    ]
    lines.extend(f"- {warning}" for warning in result.warnings)
    lines.extend(
        [
            "",
            "## Resumo",
            result.text_analysis.summary
            if result.text_analysis
            else "Análise textual não disponível.",
            "",
            "## Eventos acústicos",
        ]
    )
    lines.extend(
        f"- {e.start:.1f}–{e.end:.1f}s: {e.label} (score {e.score:.3f})" for e in result.events
    )
    lines.extend(["", "## Segmentos"])
    for row in segment_rows(result):
        lines.extend(
            [
                f"### Segmento {row['id']} ({row['start']:.1f}–{row['end']:.1f}s)",
                f"Falante: {row['speaker']}",
                f"Transcrição: {row['text']}",
                f"Voz: {row['voice_emotion']} (score {row['voice_score']:.3f})",
                f"Texto: {row['text_emotion']}; discordância: {row['discordance']}",
                row["explanation"],
                "",
            ]
        )
    lines.append("## Menções para revisão humana")
    lines.extend(f"- {a.start:.1f}s: {a.term}. {a.message}" for a in result.alerts)
    if result.text_analysis:
        lines.extend(["", "## Observações"])
        lines.extend(f"- {item}" for item in result.text_analysis.observations)
    lines.extend(
        ["", "## Modelos e configuração", *[f"- {k}: {v}" for k, v in result.metadata.items()]]
    )
    return "\n".join(lines)
