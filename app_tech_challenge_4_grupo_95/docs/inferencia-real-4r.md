# Auditoria do novo áudio START_VIDEO_RECORDING_4r — 14/11/2026

Arquivo: `D:\downloads\START_VIDEO_RECORDING_4r.wav`.
SHA-256: `be5280dc5e7f55af23125ce218e33af30d4dc2ac8f9fdc557131b653a8f568ba`.
WAV PCM16, mono, 16 kHz, duração 418,816 segundos (aproximadamente 6 min 59 s).

## Conclusão e escopo executado

**A equivalência integral não foi confirmada: a etapa SER divergiu novamente.**
Esta auditoria começou pelo pré-requisito que falhou no arquivo anterior. Foram usados
pesos locais reais e três trechos do novo arquivo, sem modelo simulado, sem áudio sintético,
sem atribuir esses trechos a um paciente e sem inventar transcrição.
O campo textual vazio nos probes serve apenas ao contrato da função de voz, que não utiliza texto.

O carregamento original ignora `classifier.dense.bias`, `classifier.dense.weight`,
`classifier.output.bias` e `classifier.output.weight`. O Transformers inicializa novos
`classifier.bias`, `classifier.weight`, `projector.bias` e `projector.weight` ao instanciar
`Wav2Vec2ForSequenceClassification`. O aviso apareceu nos dois carregamentos independentes.
Isso indica que o classificador treinado não foi restaurado pelo carregador automático
utilizado nos dois carregamentos auditados. A troca por um áudio mais claro não
corrige essa incompatibilidade de carregamento.

| Trecho real | Caminho do aplicativo | Segundo carregamento independente |
|---|---|---|
| 0.00–5.000 s | Neutro (0.138) | Tristeza / Abatimento (0.151) |
| 209.41–214.410 s | Neutro (0.133) | Calma (0.132) |
| 413.82–418.816 s | Neutro (0.149) | Alegria / Entusiasmo (0.136) |

Em todos os trechos, também diferiram a emoção secundária e seu score. Pitch médio,
desvio de pitch e RMS foram iguais nos dois caminhos. Os arrays de áudio após o
pré-processamento foram exatamente iguais. As emoções acima são saídas observadas de
classificadores com pesos novos; não devem ser interpretadas como emoções reais do falante.

A avaliação foi interrompida após essa divergência. **Whisper, AST, diarização, OpenAI,
seleção do paciente e relatório final não foram executados para este novo áudio.**
Portanto, não há resultado integral deste arquivo, nem afirmação de igualdade das etapas
não executadas. A divergência comprovada no SER é suficiente para impedir certificar
igualdade de toda a inferência. A auditoria completa do arquivo anterior permanece registrada
separadamente em [Inferência real](inferencia-real.md).

## Reprodução e evidências

Com os pesos já disponíveis no cache local:

```powershell
.\.venv\Scripts\python.exe -u scripts/verify_audio_checkpoint.py 'D:\downloads\START_VIDEO_RECORDING_4r.wav'
```

O script usa `HF_HUB_OFFLINE=1` e `TRANSFORMERS_OFFLINE=1`: não faz downloads nem chamadas
à OpenAI. Carrega o mesmo checkpoint duas vezes, como execuções independentes; não fixa
uma semente adicional para mascarar a divergência. O caminho do app executa
`LocalModels.voice_features`; o outro executa `analyze_voice_emotions` e
`extract_prosodic_features` no segundo caminho auditado. Os trechos têm aproximadamente cinco
segundos no início, meio e fim do arquivo.

Resultados estruturados: `outputs/real-parity/START_VIDEO_RECORDING_4r/checkpoint-verification.json`.
A pasta é ignorada pelo Git. O áudio original não foi modificado. Nenhum checkpoint,
regra do aplicativo foi alterado nesta nova auditoria.
Ruff lint e verificação de formatação passaram. Não foi necessária uma nova dependência.

Antes de certificar equivalência real, o carregamento do classificador treinado precisa ser
corrigido e validado em conjunto nos dois pontos de execução, seguido pela auditoria completa.
Os testes de regras existentes não substituem essa validação de pesos.
