"""Entrada da interface: python -m streamlit run app.py."""

import hashlib
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import soundfile as sf
import streamlit as st
from dotenv import load_dotenv

from clinical_audio.audio import decode_upload
from clinical_audio.backends import LocalModels
from clinical_audio.config import Settings
from clinical_audio.demo import demo_audio, demo_result
from clinical_audio.dependencies import (
    MissingDependenciesError,
    ensure_dependencies,
    install_command,
)
from clinical_audio.errors import processing_error_message
from clinical_audio.pipeline import AnalysisPipeline
from clinical_audio.reports import csv_bytes, markdown_report, segment_rows
from clinical_audio.semantic import OpenAISemantic


@st.cache_resource(show_spinner=False)
def local_pipeline(whisper_model: str, language: str, hf_token: str):
    settings = Settings(whisper_model=whisper_model, language=language, hf_token=hf_token)
    return AnalysisPipeline(settings, LocalModels(settings))


def render_results(result):
    st.subheader("Resultados da análise")
    if result.mode == "demo":
        st.warning("DEMONSTRAÇÃO com dados fixos. Nenhum arquivo enviado foi analisado.")
    for warning in result.warnings:
        st.info(warning)
    columns = st.columns(4)
    columns[0].metric("Duração", f"{result.duration_s:.1f}s")
    columns[1].metric("Segmentos", len(result.segments))
    columns[2].metric("Eventos acústicos", len(result.events))
    columns[3].metric("Menções para revisão", len(result.alerts))
    st.audio(st.session_state["audio"], format="audio/wav")

    rows = segment_rows(result)
    speakers = sorted(
        {s.speaker for s in result.segments if s.speaker not in {"Não identificado", "Ambíguo"}}
    )
    selected = st.selectbox(
        "Filtrar falante (selecione o paciente após conferir a transcrição)", ["Todos", *speakers]
    )
    visible = [r for r in rows if selected == "Todos" or r["speaker"] == selected]
    if selected != "Todos":
        st.caption(
            "O filtro altera somente a visualização; os downloads contêm a análise completa."
        )
    tabs = st.tabs(["Visão geral", "Transcrição e emoções", "Eventos e menções", "Relatórios"])
    with tabs[0]:
        if result.text_analysis:
            st.write(result.text_analysis.summary)
            for observation in result.text_analysis.observations:
                st.write(f"• {observation}")
        else:
            st.write(
                "Análise semântica não disponível. Consulte a transcrição e os resultados acústicos."
            )
        waveform = st.session_state.get("waveform")
        if waveform is not None:
            st.line_chart(
                pd.DataFrame({"tempo_s": waveform[0], "amplitude": waveform[1]}),
                x="tempo_s",
                y="amplitude",
            )
        figure = go.Figure()
        for row in visible:
            figure.add_trace(
                go.Bar(
                    x=[row["end"] - row["start"]],
                    base=[row["start"]],
                    y=[row["speaker"]],
                    orientation="h",
                    name=f"S{row['id']}: {row['voice_emotion']}",
                    hovertext=[f"Segmento {row['id']} | Discordância: {row['discordance']}"],
                )
            )
        for event in result.events:
            figure.add_trace(
                go.Bar(
                    x=[event.end - event.start],
                    base=[event.start],
                    y=["Eventos"],
                    orientation="h",
                    name=event.label,
                )
            )
        figure.update_layout(
            barmode="overlay", title="Linha do tempo de fala e eventos", xaxis_title="Segundos"
        )
        st.plotly_chart(figure, use_container_width=True)
        assessed = [r for r in visible if r["discordance"] != "Não avaliada"]
        if assessed:
            counts = pd.Series([r["discordance"] for r in assessed]).value_counts()
            st.bar_chart(counts.rename("Segmentos"))
    with tabs[1]:
        if visible:
            st.dataframe(pd.DataFrame(visible), hide_index=True, use_container_width=True)
        else:
            st.write("Nenhum segmento de fala disponível.")
    with tabs[2]:
        st.caption(
            "Eventos são hipóteses acústicas. Menções textuais incluem frases negadas e exigem leitura do contexto."
        )
        st.dataframe(pd.DataFrame([e.model_dump() for e in result.events]), hide_index=True)
        st.dataframe(pd.DataFrame([a.model_dump() for a in result.alerts]), hide_index=True)
    with tabs[3]:
        st.download_button(
            "Baixar relatório Markdown", markdown_report(result), "relatorio.md", "text/markdown"
        )
        st.download_button("Baixar segmentos CSV", csv_bytes(result), "segmentos.csv", "text/csv")
        st.download_button(
            "Baixar análise JSON",
            result.model_dump_json(indent=2),
            "analise.json",
            "application/json",
        )
        with st.expander("Modelos e configuração da execução"):
            st.json(result.metadata)


def main():
    load_dotenv()
    st.set_page_config(page_title="G95 | Áudio e texto", page_icon="🎙️", layout="wide")
    st.title("Análise de áudio e texto de consultas")
    st.write("Tech Challenge 4 · Grupo 95 · Transcrição, eventos acústicos, voz e discurso")
    st.caption(
        "Protótipo acadêmico por arquivo. Resultados para revisão humana, sem diagnóstico ou monitoramento em tempo real."
    )
    settings = Settings.from_env()
    with st.sidebar:
        st.header("Configuração")
        language = st.selectbox(
            "Idioma da fala",
            ["pt", "en", "auto"],
            index=["pt", "en", "auto"].index(settings.language)
            if settings.language in ["pt", "en", "auto"]
            else 0,
        )
        sizes = ["tiny", "base", "small", "medium", "large"]
        size = st.selectbox(
            "Modelo Whisper",
            sizes,
            index=sizes.index(settings.whisper_model) if settings.whisper_model in sizes else 1,
        )
        diarize = st.checkbox("Separar falantes com Pyannote", value=False)
        use_openai = st.checkbox("Analisar texto com OpenAI", value=False)
        st.caption(
            "Diarização requer HF_TOKEN. OpenAI requer OPENAI_API_KEY no .env e pode gerar cobrança."
        )
        if st.button("Limpar resultados"):
            for key in ["result", "audio", "waveform", "fingerprint"]:
                st.session_state.pop(key, None)
    st.subheader("Enviar áudio")
    upload = st.file_uploader("Arquivo WAV ou FLAC (até 25 MB e 10 minutos)", type=["wav", "flac"])
    consent = st.checkbox(
        "Confirmo autorização para processar este áudio e, se habilitado, enviar transcrição e descritores à OpenAI."
    )
    if use_openai:
        st.info(
            "Somente transcrição e descritores serão enviados à OpenAI. O áudio é processado localmente."
        )
    data = upload.getvalue() if upload is not None else b""
    fingerprint = hashlib.sha256(
        data + f"{language}|{size}|{diarize}|{use_openai}".encode()
    ).hexdigest()
    if st.session_state.get("fingerprint") not in (None, fingerprint):
        for key in ["result", "audio", "waveform", "fingerprint"]:
            st.session_state.pop(key, None)
    if st.button("Analisar arquivo", type="primary", disabled=upload is None or not consent):
        for key in ["result", "audio", "waveform"]:
            st.session_state.pop(key, None)
        if use_openai and not settings.openai_key.get_secret_value():
            st.error("Configure OPENAI_API_KEY no .env ou desabilite a análise textual.")
        elif diarize and not settings.hf_token.get_secret_value():
            st.error("Configure HF_TOKEN no .env ou desabilite a separação de falantes.")
        else:
            current_stage = "Verificando dependências"

            def report_progress(stage):
                nonlocal current_stage
                current_stage = stage
                st.write(stage)

            try:
                ensure_dependencies(diarize)
                with st.status(
                    "Preparando análise; a primeira execução pode baixar modelos.", expanded=True
                ) as status:
                    current_stage = "Lendo áudio WAV/FLAC"
                    samples = decode_upload(upload.name, data, settings)
                    current_stage = "Carregando modelos locais"
                    service = local_pipeline(size, language, settings.hf_token.get_secret_value())
                    semantic = OpenAISemantic(settings) if use_openai else None
                    result = service.analyze(
                        samples, diarize=diarize, semantic=semantic, progress=report_progress
                    )
                    current_stage = "Preparando resultados"
                    buffer = BytesIO()
                    sf.write(buffer, samples, settings.sample_rate, format="WAV")
                    stride = max(1, len(samples) // 5000)
                    st.session_state.update(
                        {
                            "result": result,
                            "audio": buffer.getvalue(),
                            "fingerprint": fingerprint,
                            "waveform": (
                                np.arange(0, len(samples), stride) / settings.sample_rate,
                                samples[::stride],
                            ),
                        }
                    )
                    status.update(label="Análise concluída", state="complete", expanded=False)
            except MissingDependenciesError as exc:
                st.error(str(exc))
                st.write("Execute no terminal, na pasta do projeto, e reinicie a aplicação:")
                st.code(install_command(exc.diarize), language="powershell")
            except ImportError as exc:
                module = exc.name or "não identificado"
                st.error(
                    f"Falha ao importar o módulo '{module}'. Há um pacote ausente ou incompatibilidade entre dependências."
                )
                st.write("Atualize os extras no mesmo ambiente usado pela aplicação e reinicie:")
                st.code(install_command(diarize), language="powershell")
                st.caption(
                    "Se persistir, confira as versões de torch e torchaudio e use Python 3.11 em um ambiente limpo, conforme o README."
                )
            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(processing_error_message(exc, current_stage))
    if st.button("Explorar demonstração sem modelos ou credenciais"):
        st.session_state.update(
            {"result": demo_result(), "audio": demo_audio(), "fingerprint": fingerprint}
        )
        st.session_state.pop("waveform", None)
    if "result" in st.session_state:
        render_results(st.session_state["result"])


if __name__ == "__main__":
    main()
