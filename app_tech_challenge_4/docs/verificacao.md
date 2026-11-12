# Verificação da entrega

Verificação local realizada em 11/11/2026, Windows, Python 3.13.12, em ambiente virtual isolado deste projeto.

| Verificação | Resultado |
|---|---|
| pytest após diagnóstico por etapa, recuperação da diarização e validação do classificador SER | 75 testes aprovados, sem skips |
| Cobertura do pacote `clinical_audio` | 99,30% de statements; mínimo exigido de 80% |
| Ruff lint | Aprovado |
| Ruff format | Aprovado |
| pip check | Nenhuma dependência incompatível detectada na instalação leve |
| Build Python da entrega inicial | sdist e wheel construídos com sucesso antes da migração do leitor de áudio |
| Inicialização Streamlit da entrega inicial | Servidor iniciado em endereço local |
| HTTP `/_stcore/health` da entrega inicial | 200, corpo `ok` |
| Interface via AppTest após migração | Demonstração, filtro, limpeza, upload WAV real com backend de IA substituto e invalidação de resultados aprovados |
| Leitura sem processos externos | WAV e FLAC reais, estéreo, 8/16/44,1/48 kHz, normalização para mono em 16 kHz aprovados |
| Validação de áudio | Extensões recusadas, conteúdo disfarçado, corrupção, NaN, duração e limites de canais/taxa aprovados |
| Reamostragem | Atenuação de frequência acima de Nyquist verificada |
| Diagnóstico de dependências | Lista de pacotes ausentes, extra opcional de diarização e comando com o Python da aplicação verificados |
| Erros de importação na interface | Ausência de pacote e incompatibilidade simuladas, com mensagem e comando de recuperação verificados |
| Erros por etapa | Rede, acesso restrito, certificados, memória e permissões diferenciados sem expor o conteúdo das exceções |
| Falha opcional de diarização | Preserva transcrição, eventos e análise de voz, sem inventar falantes, com aviso e metadados de indisponibilidade |
| Carregamento do modelo SER | Rejeita checkpoints que deixem pesos da classificação ausentes ou incompatíveis |

O leitor atual utiliza SoundFile e SciPy. Os testes de decodificação bloqueiam `subprocess.run` para verificar que WAV/FLAC não dependem de processos externos. FFmpeg foi removido da configuração, da CI e da instalação Docker.

A cobertura mede o pacote de serviços; `app.py` é exercitado separadamente pelos testes de interface. Na suíte automatizada, os adaptadores de modelos e OpenAI são substituídos por doubles, sem chamadas faturadas ou downloads de pesos. O diagnóstico manual do áudio informado pela equipe é registrado separadamente abaixo. Não foram medidas qualidade preditiva ou validade clínica.

Após identificar os pacotes de IA ausentes, o extra `ml` foi instalado na `.venv` local. `pip check` concluiu sem incompatibilidades declaradas. Os imports de Whisper, torch, torchaudio, librosa e `transformers.pipeline` foram executados com sucesso, sem carregar pesos. Versões verificadas: torch 2.15.0+cpu, torchaudio 2.11.1+cpu e librosa 0.11.0. A diarização permanece opcional e exige instalação do extra correspondente e configuração de HF_TOKEN.

O extra `diarization` também foi instalado posteriormente na `.venv` local, com `pyannote.audio` 4.0.7. O import de `pyannote.audio.Pipeline`, a verificação de presença de todos os módulos de IA/diarização e `pip check` concluíram com sucesso. O modelo Community-1 não foi baixado ou executado nessa verificação; seu uso ainda exige HF_TOKEN e aceite dos termos pela conta da equipe.

No diagnóstico do arquivo `START_VIDEO_RECORDING_ky56r.wav` fornecido pela equipe, a leitura confirmou WAV PCM de 16 bits, mono, 16 kHz e duração de 46,232 segundos. A transcrição local com Whisper concluiu. Ao testar a diarização com a configuração local, o Hugging Face retornou `GatedRepoError` para Community-1. O código foi ajustado para continuar sem identificação dos falantes nessa situação. Nenhum áudio ou transcrição foi enviado a serviços de análise externos; houve somente acesso ao Hugging Face para autenticação e obtenção dos modelos.

A detecção AST concluiu no mesmo arquivo. No carregamento do SER original, o Transformers informou que ignorou os pesos `classifier.dense`/`classifier.output` e criou novos pesos de classificação/projeção. Essa execução foi interrompida sem aceitar suas previsões como resultado. O modelo foi substituído por `superb/wav2vec2-base-superb-er`, compatível com a arquitetura automática e com quatro categorias. A implementação passou a validar o relatório de carregamento dos pesos.

O checkpoint substituto carregou com a verificação de pesos aprovada. A classificação de voz e a extração de pitch/RMS concluíram em um trecho de cinco segundos do WAV fornecido. Esse teste manual foi executado em CPU, com duas threads PyTorch, sem OpenAI. A transcrição e o AST foram exercitados no áudio completo; o SER substituto foi exercitado no trecho de cinco segundos, não em uma nova execução completa de ponta a ponta. A diarização real permanece sem validação de inferência enquanto o acesso ao Community-1 não for liberado.

Docker não estava disponível no ambiente local. A imagem e o workflow foram preparados, mas a construção/smoke test do container e a execução remota do GitHub Actions precisam ocorrer no ambiente da equipe. Nenhuma publicação, release ou implantação externa foi realizada.

Para repetir os testes, siga o README. Todos os testes de leitura são executados sem FFmpeg, sem necessidade de instalação ou configuração de PATH para esse programa.
