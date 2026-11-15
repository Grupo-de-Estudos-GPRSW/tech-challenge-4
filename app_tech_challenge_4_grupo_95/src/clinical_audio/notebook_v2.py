"""Funções de referência; relatório retorna texto para download, sem gravar no servidor."""

from pathlib import Path
from typing import Any

import pandas as pd


def get_patient_speaker_id_acoustically(
    diarized_segments: list[dict[str, Any]], transcription_segments: list[dict[str, Any]]
) -> str:
    """
    Localiza o falante que se apresenta como médico/robô e define o outro como Paciente,
    usando interseção temporal e análise de palavras-chave.
    """
    robot_speaker = None

    # Palavras-chave indicando o médico/robô
    keywords = ["doctor", "robot", "doutor", "bom dia", "sentido", "consulta"]

    for seg in transcription_segments:
        texto = seg["transcricao"].lower()
        if any(kw in texto for kw in keywords):
            seg_start = seg["inicio_s"]
            seg_end = seg["fim_s"]

            # Encontra qual speaker tem maior sobreposição temporal com este segmento
            best_speaker = None
            max_overlap = 0.0
            for turn in diarized_segments:
                # Calcula a sobreposição (interseção) de intervalos
                overlap_start = max(seg_start, turn["inicio_s"])
                overlap_end = min(seg_end, turn["fim_s"])
                overlap = max(0.0, overlap_end - overlap_start)

                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = turn["speaker_raw"]

            if best_speaker:
                robot_speaker = best_speaker
                break

    all_speakers = list(set(turn["speaker_raw"] for turn in diarized_segments))

    if robot_speaker and len(all_speakers) >= 2:
        patient_speaker = [sp for sp in all_speakers if sp != robot_speaker][0]
    else:
        # Caso não consiga diferenciar pelo texto, assumimos o speaker que mais fala ou o SPEAKER_01 como padrão
        speaker_counts = {}
        for turn in diarized_segments:
            dur = turn["fim_s"] - turn["inicio_s"]
            speaker_counts[turn["speaker_raw"]] = speaker_counts.get(turn["speaker_raw"], 0.0) + dur

        # Paciente costuma falar menos ou ser o segundo speaker ativo
        sorted_speakers = sorted(speaker_counts.items(), key=lambda x: x[1])
        if sorted_speakers:
            patient_speaker = sorted_speakers[0][0]
        else:
            patient_speaker = "SPEAKER_01"

    return patient_speaker


def filter_patient_segments(
    diarized_turns: list[dict[str, Any]], transcribed_segments: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    patient_speaker_id = get_patient_speaker_id_acoustically(diarized_turns, transcribed_segments)
    patient_segments = []
    for seg in transcribed_segments:
        seg_start = seg["inicio_s"]
        seg_end = seg["fim_s"]

        best_speaker = None
        max_overlap = 0.0

        for turn in diarized_turns:
            overlap_start = max(seg_start, turn["inicio_s"])
            overlap_end = min(seg_end, turn["fim_s"])
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = turn["speaker_raw"]

        if best_speaker == patient_speaker_id:
            patient_segments.append(seg)
    return patient_segments


def calculate_discordance_locally(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compara text_emotion vs voice_emotion e non_verbal_events programaticamente."""
    results = []
    positive_emotions = [
        "alegria",
        "calma",
        "neutro",
        "tranquilidade",
        "otimismo",
        "bem-estar",
        "positivo",
    ]
    negative_emotions = [
        "tristeza",
        "raiva",
        "medo",
        "ansiedade",
        "desconforto",
        "aversão",
        "abatimento",
        "irritação",
        "negativo",
    ]

    for seg in segments:
        texto_emo = seg.get("emocao_discurso_texto", "").lower()
        voz_emo = seg.get("emocao_voz_primaria", "").lower()

        is_text_positive = any(p in texto_emo for p in positive_emotions)
        is_voice_negative = any(n in voz_emo for n in negative_emotions)

        is_text_negative = any(n in texto_emo for n in negative_emotions)
        is_voice_positive = any(p in voz_emo for p in positive_emotions)

        discordancia = False
        nivel = "Nenhuma"
        explicacao = "Convergência entre texto e voz."

        # Regra 1: Text Positivo x Voz Negativa (ou vice-versa)
        if (is_text_positive and is_voice_negative) or (is_text_negative and is_voice_positive):
            discordancia = True
            nivel = "Alta" if seg.get("confianca_voz_primaria", 0) > 0.6 else "Moderada"
            explicacao = f"Divergência direta: Texto indicou '{seg['emocao_discurso_texto']}' mas Voz indicou '{seg['emocao_voz_primaria']}'."
        # Regra 2: Sinais Físicos com Texto Positivo
        elif seg.get("eventos_nao_verbais_proximos") and is_text_positive:
            discordancia = True
            nivel = "Leve"
            explicacao = f"Sinais físicos ({', '.join(seg['eventos_nao_verbais_proximos'])}) contradizem o relato textual positivo/neutro."

        seg_copy = seg.copy()
        seg_copy["ha_discordancia"] = discordancia
        seg_copy["nivel_discordancia"] = nivel
        seg_copy["explicacao_clinica"] = explicacao
        results.append(seg_copy)
    return results


def generate_clinical_discordance_report(
    df_full: pd.DataFrame,
    df_events: pd.DataFrame,
    output_md_path: Path = Path("relatorio_discordancias_clinicas.md"),
) -> str:
    """Formata e salva o relatório programático de discordâncias emocionais e eventos não-verbais."""
    total_segs = len(df_full)
    discordant_df = df_full[df_full["ha_discordancia"].eq(True)]
    num_discordant = len(discordant_df)

    lines = [
        "# 📋 Relatório Clínico Analítico: Eventos Não-Verbais e Discordâncias Emocionais",
        "",
        "## 1. Resumo Quantitativo da Consulta",
        f"- **Total de segmentos analisados:** {total_segs}",
        f"- **Total de discordâncias detectadas:** {num_discordant} ({(num_discordant / max(1, total_segs)) * 100:.1f}%)",
        f"- **Total de eventos não-verbais:** {len(df_events)}",
        "",
        "## 2. Sons Não-Verbais Detectados",
    ]

    if df_events.empty:
        lines.append("_Nenhum evento não-verbal relevante detectado._\n")
    else:
        lines.append("| Início (s) | Fim (s) | Evento Clínico Detectado | Confiança Acústica |")
        lines.append("| :---: | :---: | :--- | :---: |")
        for _, ev in df_events.iterrows():
            lines.append(
                f"| {ev['inicio_s']:.2f}s | {ev['fim_s']:.2f}s | **{ev['evento_clinico']}** | "
                f"{ev['confianca'] * 100:.1f}% |"
            )
        lines.append("")

    lines.extend(
        [
            "## 3. Comparativo Geral: Texto vs Voz",
            "| Seg # | Tempo | Transcrição | Emoção Texto | Emoção Voz | Sons Próximos | Discordância |",
            "| :---: | :---: | :--- | :--- | :--- | :--- | :---: |",
        ]
    )

    for _, row in df_full.iterrows():
        ev_str = (
            ", ".join(row["eventos_nao_verbais_proximos"])
            if row["eventos_nao_verbais_proximos"]
            else "—"
        )
        disc_badge = (
            f"⚠️ **{row['nivel_discordancia']}**" if row["ha_discordancia"] else "✅ Nenhuma"
        )
        lines.append(
            f"| {row['segment_id']} | {row['inicio_s']:.1f}s–{row['fim_s']:.1f}s | "
            f'"{row["transcricao"]}" | {row["emocao_discurso_texto"]} | '
            f"{row['emocao_voz_primaria']} | "
            f"{ev_str} | {disc_badge} |"
        )

    lines.extend(
        [
            "",
            "## 4. Detalhamento Clínico Programático",
        ]
    )

    if discordant_df.empty:
        lines.append("_Não foram identificadas discordâncias significativas pelo sistema._\n")
    else:
        for _, row in discordant_df.iterrows():
            lines.extend(
                [
                    f"### 🔍 Segmento #{row['segment_id']} ({row['inicio_s']:.1f}s a {row['fim_s']:.1f}s)",
                    f'- **Fala:** *"{row["transcricao"]}"*',
                    f"- **Texto indicou:** `{row['emocao_discurso_texto']}` | **Voz indicou:** `{row['emocao_voz_primaria']}`",
                    f"- **Nível:** {row['nivel_discordancia']} — **Motivo:** {row['explicacao_clinica']}",
                    "",
                ]
            )

    report_md = "\n".join(lines)

    return report_md
