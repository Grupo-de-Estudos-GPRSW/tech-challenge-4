"""Extração estruturada de emoções a partir do texto transcrito."""

import json

from clinical_audio.config import Settings
from clinical_audio.models import ConsultationTextEmotions, Segment, TextEmotion, TextEmotions

SYSTEM_PROMPT = (
    "Você é um assistente de extração de características. Identifique a emoção predominante "
    "estritamente pelo conteúdo verbal de cada segmento. Use termos simples como: "
    "'Alegria', 'Calma', 'Neutro', 'Tristeza', 'Raiva', 'Medo', 'Ansiedade', 'Desconforto'."
)


class OpenAISemantic:
    def __init__(self, settings: Settings, client=None):
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=settings.openai_key.get_secret_value())
        self.client = client
        self.model = settings.openai_model

    def analyze(self, segments: list[Segment]) -> TextEmotions:
        payload = [{"segment_id": s.id, "transcricao": s.text} for s in segments]
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Analise o texto:\n{json.dumps(payload, ensure_ascii=False, indent=2)}",
                },
            ],
            response_format=ConsultationTextEmotions,
            temperature=0.0,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError("A API não retornou resposta estruturada.")
        return TextEmotions(
            segments=[
                TextEmotion(id=s.segment_id, emotion=s.emocao_discurso_texto)
                for s in parsed.segmentos_analisados
            ]
        )
