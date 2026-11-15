# Auditoria da inferência real — 14/11/2026

Arquivo informado pela equipe: `D:\downloads\START_VIDEO_RECORDING_ky56r.wav`.
SHA-256: `b2a60101e1302e10156d1bfd87bd6696baf021579afa91e91a8feaea4bef7597`.
WAV PCM16, mono, 16 kHz, 739.712 amostras, duração 46,232 segundos.

## Resultado observado

**A equivalência integral da inferência real não foi comprovada.** As etapas de áudio,
Whisper e AST coincidiram, mas a primeira tentativa de diarização falhou e o SER original apresentou pesos novos
e previsões diferentes entre carregamentos independentes. O alinhamento de regras e os
testes de software não eliminam esses problemas de execução/modelo.

| Etapa | Evidência real |
|---|---|
| Áudio | Arrays do app e `librosa.load` exatamente iguais; erro máximo zero |
| Whisper base/en | 12 segmentos; IDs, tempos e transcrições exatamente iguais entre array e arquivo |
| AST | 4 eventos em cada caminho, com igualdade dos campos comparados; 76 janelas por caminho |
| Diarização | Primeira tentativa: RuntimeError; repetição isolada: 11 turnos concluídos. A igualdade dos turnos entre arquivo e tensor não foi comprovada |
| SER ehcalabres | Aviso de pesos novos; dois carregamentos independentes deram previsões diferentes no mesmo trecho de 5 segundos |
| OpenAI e resultado final | Não executados na auditoria integral, interrompida pela primeira falha de diarização |

No probe SER, o primeiro carregamento classificou o trecho como `happy` (score
0,13256673514842987), e o segundo como `surprised` (0,13596931099891663).
Os classificadores foram carregados separadamente pelo mesmo comando `pipeline` usado
na aplicação. Esse resultado evidencia falta de reprodutibilidade no carregamento observado;
não deve ser interpretado como avaliação clínica das emoções do falante. Não houve troca
por outro checkpoint nem uso de voz/transcrição simuladas nesta auditoria.

O acesso ao `config.yaml` do Community-1 foi confirmado com o token configurado. A falha
posterior não deve ser descrita automaticamente como o antigo erro de acesso 403.
A repetição com `--diarization-only` concluiu com 11 turnos e não reproduziu a exceção.
A causa exata do RuntimeError inicial não foi determinada. O resultado da repetição está em
`outputs/real-parity/diarization-diagnostic.json`; ele não equivale a concluir a comparação integral.

## Ambiente e execução

Windows, Python da `.venv`, PyTorch 2.15.0, Transformers 4.57.6, librosa 0.11.0,
OpenAI SDK 2.54.0 e pyannote.audio 4.0.7. CPU com duas threads PyTorch.
A auditoria compara caminhos de carregamento e processamento com o mesmo áudio e configurações.
O carregador por arquivo do Whisper exigiu uma cópia temporária de FFmpeg, disponibilizada via
`imageio-ffmpeg` em `outputs/real-parity/tools`. Isso não altera `requirements.txt` nem a necessidade
de FFmpeg do aplicativo, que continua recebendo arrays para Whisper.

Tempos desta execução: Whisper 49,92 s, comparação AST 1.250,59 s, tentativa de diarização
215,79 s e carregamento/probe SER 29,01 s. Esses tempos incluem trabalho dos dois caminhos
quando aplicável; não são benchmark de uma única execução da interface.

Comando de auditoria:

```powershell
.\.venv\Scripts\python.exe -u scripts/verify_real_parity.py 'D:\downloads\START_VIDEO_RECORDING_ky56r.wav'
```

O relatório estruturado está em `outputs/real-parity/verification.json`. Essa pasta é ignorada
pelo Git; transcrições privadas, quando geradas, permanecem nela e não são copiadas para esta
documentação. O script imprime métricas de comparação, não credenciais ou transcrições.
A igualdade dos modelos reais e a comparação final com OpenAI permanecem pendentes até resolver
os bloqueios. Não se deve anunciar equivalência integral com base apenas nos 159 testes de software.


## Conclusão da nova análise

Os testes controlados verificam a sequência, os modelos configurados e as regras da aplicação.
A execução real demonstrou igualdade das etapas de áudio/Whisper/AST, mas não uma cadeia
integral equivalente. O SER mostrou divergência real entre carregamentos, com indicação de
pesos novos na classificação. Antes de certificar reprodutibilidade, é necessário tratar
o carregamento do classificador treinado e validar sua reprodutibilidade. A origem da falha inicial de diarização e a igualdade
arquivo/tensor também precisam ser conferidas em uma execução completa.

Nesta nova análise, a suíte voltou a passar com **159 testes e 99,27% de cobertura**.
Ruff lint e formatação também passaram. A configuração do Ruff agora exclui `outputs` e `data`,
para não analisar bibliotecas de terceiros instaladas temporariamente como ferramentas da auditoria.
O README contém a descrição do dataset fornecida pela equipe.
