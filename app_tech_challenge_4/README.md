# Tech Challenge 4 G95 — análise de áudio e texto

Aplicação acadêmica para analisar gravações de consultas, com upload e reprodução de áudio, transcrição temporal, identificação de eventos não verbais, classificação de emoção na voz, análise semântica opcional e exportação de resultados. A interface abre no navegador e é executada pelo VS Code ou terminal.

**Formatos aceitos: WAV e FLAC. Não é necessário instalar FFmpeg.** O áudio é lido com SoundFile, convertido para mono e reamostrado para 16 kHz com SciPy antes de ser enviado aos modelos locais.

O projeto foi modularizado a partir de `Tech_Challenge_4_(Antigravity) - Turma.ipynb`. O notebook original foi preservado como referência histórica; ele não é dependência da aplicação e pode conter saídas de consultas. Revise e limpe essas saídas antes de publicar o notebook em um repositório público.

## Escopo da entrega

O desafio propõe modalidades de vídeo, áudio e texto, além de exemplos de anomalias em sinais vitais e integração com nuvem. Conforme a orientação do professor relatada pela equipe, esta entrega adota **áudio + texto**. Whisper local e OpenAI mantêm a escolha tecnológica da equipe, em lugar dos serviços Azure citados no enunciado. A justificativa e as lacunas estão em [docs/escopo.md](docs/escopo.md).

O sistema processa arquivos em lote. Menções a termos críticos e eventos acústicos são sinalizações para revisão humana. Não implementa monitoramento contínuo, diagnóstico, disartria, classificação validada de fadiga, séries temporais de sinais vitais ou alertas enviados à equipe médica. Emoções estimadas não revelam o estado emocional real de uma pessoa.

## Funcionalidades

- Upload somente de WAV e FLAC, até 25 MB e 10 minutos; leitura em memória com SoundFile, conversão para mono e reamostragem para 16 kHz com SciPy, sem FFmpeg.
- Transcrição local com Whisper, em português, inglês ou idioma automático, com início e fim dos segmentos.
- Detecção de tosse, espirro, pigarro, suspiro e outros rótulos do AudioSet usando AST e janelas deslizantes, incluindo a cauda do áudio.
- Emoções acústicas com `superb/wav2vec2-base-superb-er` (raiva, alegria, tristeza e neutro), pitch médio, desvio de pitch e energia RMS. Clips curtos/silenciosos ficam indeterminados. Por custo computacional, SER/prosódia consideram até 20 segundos por segmento.
- Diarização opcional com Pyannote Community-1; associação pelo total de sobreposição temporal. Empates ficam ambíguos. IDs não são automaticamente chamados de paciente ou médico.
- Análise textual opcional via OpenAI, com resumo, emoção no texto, comparação com voz e explicações estruturadas, em lotes de até 15 segmentos.
- Menções de termos como “falta de ar” e “dor no peito”, inclusive quando negados, para conferência do contexto.
- Forma de onda, linha do tempo, tabelas, filtro por falante e download Markdown, CSV e JSON. O filtro não altera os downloads completos.
- Demonstração offline com dados fixos claramente identificados, independente do áudio enviado.

## Instalação no Windows e VS Code

Use Python **3.11** como referência para a pilha de IA. O projeto declara compatibilidade com Python 3.11–3.13 e configura essa matriz na CI; a verificação local foi realizada com Python **3.13.12**. Os extras de ML precisam ser verificados no sistema operacional e hardware escolhidos. A disponibilidade de wheels e o suporte CUDA variam.

1. Instale o Python e o VS Code. No VS Code, instale as extensões recomendadas Python, Python Debugger e Ruff.
2. Abra esta pasta em **File → Open Folder**. No terminal PowerShell, crie o ambiente:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Os comandos chamam o executável diretamente e dispensam alterar a política de execução do PowerShell. Em **Python: Select Interpreter**, selecione `.venv\Scripts\python.exe`.

O [requirements.txt](requirements.txt) instala todas as dependências diretas da aplicação, dos modelos locais, da diarização e das ferramentas de desenvolvimento, além de registrar o pacote local em modo editável. As dependências transitivas são instaladas automaticamente pelo pip. Execute os comandos na raiz do projeto. O download pode ocupar vários GB; os pesos dos modelos são baixados quando usados pela primeira vez.

O arquivo também explicita `huggingface-hub` e `safetensors` para obtenção/carregamento dos pesos, e `build`, `setuptools` e `wheel` para empacotamento. PyTorch e Torchaudio têm versão mínima 2.8, conforme exigido pelo Pyannote 4. Bibliotecas transitivas como Numba, TorchCodec, Lightning e os subpacotes do Pyannote são resolvidas pelo pip. `hf_xet` é uma otimização opcional de download, não um requisito deste fluxo.

As dependências são destinadas à aplicação modular e aos seus testes. O notebook histórico usa recursos específicos do Colab, IPython e geração demonstrativa com gTTS; ele não é um ponto de execução suportado por este `requirements.txt`.

3. Execute a aplicação:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

Abra [a aplicação local](http://localhost:8501). Clique em **Explorar demonstração sem modelos ou credenciais**. Ela funciona sem FFmpeg, chaves ou downloads de IA. O áudio é um tom sintético e **não contém as falas ou os eventos apresentados**: os resultados são fixtures inventadas para demonstrar a interface.

4. Para depurar pelo VS Code, abra **Run and Debug**, selecione **Aplicação Streamlit** e pressione **F5**. Para encerrar no terminal, pressione **Ctrl+C**.

### Preparar a análise real de áudio

Se você seguiu a instalação por `requirements.txt`, as dependências necessárias já estão incluídas, tanto para analisar o áudio quanto para separar falantes. Não é necessário instalar FFmpeg ou executar instalações separadas de extras.

Se ainda estiver usando uma instalação anterior ou incompleta, atualize no mesmo ambiente usado para iniciar o Streamlit:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

O arquivo inclui `openai-whisper` (importado como `whisper`), `torch`, `torchaudio`, `transformers`, `librosa` e `pyannote.audio`. A interface verifica os módulos necessários antes de iniciar a análise e informa os pacotes ausentes com um comando de instalação usando o Python em execução. Depois de instalar, pare o servidor com **Ctrl+C** e inicie-o novamente.

Para executar somente a demonstração e os testes sem instalar a pilha de IA, a alternativa leve continua disponível: `.\.venv\Scripts\python.exe -m pip install -e ".[dev]"`. Essa alternativa não permite analisar uploads reais até que as dependências de IA sejam instaladas.

A primeira análise baixa os pesos de Whisper, AST e Wav2Vec2. Reserve vários GB de disco/memória, acesso à internet e tempo para o download. CPU é suportada e pode ser lenta; GPU CUDA compatível é detectada automaticamente. Para GPU, instale versões compatíveis de torch e torchaudio seguindo o [seletor oficial do PyTorch](https://pytorch.org/get-started/locally/). Mantenha os dois pacotes na mesma versão compatível.

### Atualizar uma instalação anterior

Se já executava a versão que exigia FFmpeg, pare o servidor com **Ctrl+C**, abra o terminal na pasta do projeto e atualize as dependências:

```powershell
# Instala as dependências completas, incluindo IA, diarização e testes.
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Reinicie a aplicação.
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

A opção antiga `FFMPEG_BINARY` não é mais utilizada e pode ser removida do `.env`. Whisper recebe o áudio já decodificado como um array, evitando seu carregador de arquivos baseado em FFmpeg. Uma instalação de FFmpeg já existente na máquina não precisa ser alterada.

### Instalação no Linux/macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# Se SoundFile não encontrar libsndfile no Linux: sudo apt-get install libsndfile1
python -m streamlit run app.py --server.address 127.0.0.1
```

### Configurar credenciais opcionais

Se ainda não tiver um `.env`, crie-o a partir do exemplo. Se ele já existir, edite o arquivo atual para preservar suas configurações:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Edite `.env` localmente:

```dotenv
OPENAI_API_KEY=sua-chave
OPENAI_MODEL=gpt-4o-mini
HF_TOKEN=seu-token
WHISPER_MODEL=base
AUDIO_LANGUAGE=pt
```

Não coloque credenciais em código, screenshots, notebook ou commits. `.env` está ignorado pelo Git. Variáveis do ambiente têm prioridade sobre `.env`. Reinicie a aplicação ao mudar credenciais.

**OpenAI:** habilite a opção na interface apenas se quiser análise semântica e tiver chave, quota e acesso ao modelo configurado. A API pode gerar custos. Sem habilitar, Whisper, eventos, prosódia e SER continuam locais; discordâncias ficam **Não avaliadas**. Falhas da API preservam os resultados locais e geram um aviso.

**Pyannote:** a biblioteca já é instalada pelo `requirements.txt`. Para utilizá-la, configure `HF_TOKEN` e aceite os termos do [modelo Community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) com a conta do token. Instalar a biblioteca não concede acesso automático ao modelo.

Ative **Separar falantes com Pyannote**. Sem isso, o sistema analisa todos os segmentos com falante não identificado. Depois de diarizar, confira a transcrição e selecione o ID do paciente no filtro. Um segmento que cruza falantes ainda pode ter atribuição imprecisa; há limitações com sobreposição e erros de segmentação.

Se o Hugging Face responder com **GatedRepoError**, **401** ou **403**, a biblioteca está instalada, mas o modelo não está acessível com o token configurado. Entre na conta do Hugging Face que criou o token, abra a página do Community-1, aceite/solicite o acesso conforme indicado e confira se o token tem permissão de leitura desse repositório. Atualize `HF_TOKEN` no `.env` se necessário e reinicie o Streamlit. O código não aceita termos de uso automaticamente.

Falhas nessa etapa opcional geram um aviso e a análise continua com **falantes não identificados**, sem simular diarização nem filtrar o paciente. O relatório registra `diarization: indisponível`. A interface informa a etapa e o tipo de falha, distinguindo problemas de acesso, rede, certificados e memória, sem exibir os dados contidos na exceção.

## Uso com um arquivo real

1. Configure idioma e modelo Whisper na barra lateral. Comece com `tiny` ou `base` em CPU.
2. Se necessário, habilite diarização e/ou análise textual e prepare suas credenciais.
3. Envie uma gravação autorizada ou fictícia em **WAV ou FLAC**, com até **25 MB**, **10 minutos**, **8 canais** e **192 kHz**. Marque a confirmação de autorização; a interface informa quando há envio de transcrição à OpenAI.
4. Clique em **Analisar arquivo** e acompanhe as etapas. Arquivos corrompidos, formatos diferentes de WAV/FLAC e excesso de duração produzem mensagens de erro. Renomear uma extensão não converte o áudio.
5. Confira avisos, transcrição, eventos, emoções, linha do tempo e menções. O score é a saída do classificador, não probabilidade clínica calibrada.
6. Baixe relatórios na aba **Relatórios**. Use **Limpar resultados** para remover os resultados da sessão; o controle de upload pode ser limpo no próprio widget.

Uploads são lidos diretamente da memória, sem arquivos temporários ou processos externos. O formato real, duração e limites de até 8 canais e 192 kHz são verificados antes da leitura completa. Os resultados ficam na memória da sessão e não são persistidos em arquivos pelo servidor. Pesos dos modelos são cacheados em disco pelas bibliotecas. A saída gerada pode conter informações sensíveis: quem baixa o relatório é responsável pelo armazenamento e descarte.

## Organização

```text
app.py                         Interface Streamlit
src/clinical_audio/
  config.py                    Configuração e segredos
  models.py                    Contratos Pydantic
  audio.py                     Upload WAV/FLAC, SoundFile, SciPy e janelas temporais
  backends.py                  Whisper, AST, SER, prosódia e Pyannote
  rules.py                     Sobreposição, fusão, termos e validação de IDs
  semantic.py                  Adaptador OpenAI e saídas estruturadas
  pipeline.py                  Orquestração com dependências injetáveis
  reports.py                   Exportação em memória
  demo.py                      Cenário fictício offline
  dependencies.py              Diagnóstico dos extras e comando de instalação
  errors.py                    Mensagens de erro por etapa, sem expor dados sensíveis
tests/                         Testes de regras, serviços e interface
docs/                          Escopo, arquitetura e avaliação
.github/workflows/ci-cd.yml     CI, container e entrega no GHCR
.vscode/                       Execução e testes no VS Code
Dockerfile                     Empacotamento da aplicação
requirements.txt               Instalação completa das dependências
pyproject.toml                 Metadados, dependências e configuração das ferramentas
```

Consulte [a arquitetura e decisões](docs/arquitetura.md) e [o plano de avaliação](docs/avaliacao.md). Os adaptadores usam imports tardios: testar regras e executar a demonstração não carrega modelos nem faz chamadas externas.

## Testes e qualidade

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m build
```

O pytest aplica cobertura mínima de **80% do pacote** e gera `coverage.xml`. Há testes de limites de upload, formatos reais WAV/FLAC, áudio estéreo, reamostragem com filtro antialiasing, janelas finais, sobreposição de falantes, fusão de eventos, termos negados, indisponibilidade de API, respostas estruturadas e IDs, exportações e interface via Streamlit AppTest. Modelos e API são substituídos por doubles; isso verifica contratos e lógica, não a qualidade preditiva dos pesos. Os testes de leitura usam arquivos codificados em memória e bloqueiam a chamada a processos externos.

Na verificação local, incluindo diagnóstico por etapa, continuidade quando a diarização falha e validação do carregamento do classificador SER, **75 testes passaram, sem testes pulados**, com **99,30% de cobertura do pacote `clinical_audio`**. O Ruff também passou. A interface foi exercitada com upload WAV real, modelos de IA substituídos em teste e falhas de importação simuladas. Consulte [docs/verificacao.md](docs/verificacao.md) para as condições e limites da execução. Docker e CI remota ainda não foram validados nesse ambiente.

Para formatar: `python -m ruff check . --fix` e `python -m ruff format .`, usando o Python do ambiente. No VS Code, o painel **Testing** descobre os testes em `tests/`.

## CI/CD com GitHub Actions

A pasta recebida não contém repositório `.git` nem destino remoto configurado. Os arquivos do workflow estão prontos para serem versionados e executados quando a equipe publicar o projeto no GitHub.

**CI:** em pull requests, pushes para `main`/`master` e execução manual, o workflow instala a aplicação leve, executa Ruff, testes e cobertura em Python 3.11, 3.12 e 3.13, constrói o pacote e armazena o XML de cobertura. Sem credenciais de OpenAI/Hugging Face e sem download de pesos. Após os testes, constrói a imagem Docker e verifica o endpoint HTTP de saúde.

O workflow não instala FFmpeg; os testes de WAV/FLAC usam SoundFile e SciPy.

**CD (entrega contínua):** ao enviar uma tag como `v1.0.0`, após passar pelos mesmos controles, publica a imagem leve em `ghcr.io/<organizacao-ou-usuario>/<repositorio>:v1.0.0` usando `GITHUB_TOKEN`. Não faz implantação em servidor. Um destino de hospedagem não foi informado; a implantação é manual com Docker. O job pede `packages: write`; políticas da organização podem precisar liberar criação de pacotes. Configure proteção de branch para exigir os checks de qualidade e container antes do merge.

Depois de configurar seu repositório e remoto, uma release é disparada por:

```bash
git tag v1.0.0
git push origin v1.0.0
```

Não rode os comandos de release antes de revisar o que será publicado, especialmente o notebook e seus outputs. Credenciais reais nunca são necessárias para a CI.

### Container e execução

```bash
# Aplicação leve e demonstração
docker build -t g95:demo .
docker run --rm -p 127.0.0.1:8501:8501 g95:demo

# Análise real: imagem muito maior, com os extras de ML e diarização
docker build --build-arg INSTALL_ML=true -t g95:ml .
docker run --rm -p 127.0.0.1:8501:8501 --env-file .env g95:ml
```

O container roda como usuário sem privilégios, inclui libsndfile e possui healthcheck; FFmpeg não é instalado. A imagem publicada automaticamente é **leve**: permite demonstração; para analisar uploads com IA, construa a variante `INSTALL_ML=true`. Pesos ainda são baixados na primeira execução. GPU em Docker exige configuração adicional do host e runtime NVIDIA; os comandos acima usam CPU.

Após uma release, substitua os nomes abaixo pelos do seu projeto:

```bash
docker pull ghcr.io/SEU_USUARIO/SEU_REPOSITORIO:v1.0.0
docker run --rm -p 127.0.0.1:8501:8501 ghcr.io/SEU_USUARIO/SEU_REPOSITORIO:v1.0.0
```

Pacotes privados exigem autenticação no registro. A aplicação não possui autenticação de usuários, TLS, auditoria ou isolamento de uma plataforma clínica; execute localmente para a apresentação acadêmica.

## Limitações e avaliação acadêmica

O modelo SER atual foi treinado em inglês e não foi validado para português clínico. Ele trabalha com quatro categorias e não classifica medo, calma ou outras emoções fora dessas classes. O checkpoint do notebook foi substituído porque, na integração testada, sua cabeça de classificação não era carregada pela arquitetura automática do Transformers. O carregamento atual verifica pesos ausentes/incompatíveis para impedir inferência com uma cabeça nova. AST usa rótulos gerais de áudio, não diagnósticos. Os limiares de 0,12 e RMS 0,015 foram herdados do experimento e precisam ser calibrados com dados anotados. Somente inferência não comprova desempenho: registre métricas em um conjunto consentido e separado, conforme [docs/avaliacao.md](docs/avaliacao.md).

Não inferimos paciente pelo primeiro falante nem por expressões como “doutor”. Não geramos consultas sintéticas silenciosamente para substituir arquivos ausentes. Discordância textual/acústica é uma hipótese explicada, nunca uma conclusão de ocultação de sintomas. Menções a sintomas não geram diagnóstico, prescrição nem comunicação externa automática.

## Problemas frequentes

| Problema | Como resolver |
|---|---|
| Import `clinical_audio` não encontrado | Selecione `.venv` e execute `python -m pip install -e .` nesse ambiente. |
| MP3, M4A ou OGG não aceito | Exporte o arquivo como WAV ou FLAC no seu editor de áudio; renomear a extensão não converte o conteúdo. |
| Falha ao carregar SoundFile/libsndfile | Atualize as dependências com `python -m pip install -e .`; no Linux, instale libsndfile1 se necessário. |
| Falta `whisper`, `transformers`, `librosa` ou `pyannote.audio` | Instale as dependências completas com `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` no terminal do VS Code. |
| “Pacotes necessários não instalados” | Na pasta do projeto, execute `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` usando a mesma `.venv` da aplicação e reinicie o Streamlit. |
| Pacote instalado, mas falha ao importar | Confirme o interpretador com `python -c "import sys; print(sys.executable)"` no terminal e compare com o caminho do comando da interface. Verifique também compatibilidade entre torch e torchaudio. |
| Pyannote retorna GatedRepoError/401/403 | Aceite/solicite acesso ao Community-1 e confira a permissão de leitura de HF_TOKEN na mesma conta. Reinicie a aplicação. A análise continua sem identificar falantes enquanto o acesso não estiver disponível. |
| Falha ao obter AST ou SER | Confira o acesso HTTPS ao Hugging Face e o cache dos modelos. A instalação dos pacotes não inclui os pesos. |
| OpenAI indisponível | Confira chave, quota, acesso ao modelo e internet; resultados locais são preservados. |
| Download lento ou sem memória | Comece com Whisper tiny, áudio curto e diarização desligada; AST/SER ainda exigem memória. |
| Erro entre torch e torchaudio | Reinstale versões correspondentes pelo seletor oficial PyTorch em ambiente limpo. |
| Portas ocupadas | Use `--server.port 8502`. |
| Python 3.13 falha ao instalar extras | Use Python 3.11 para a pilha de ML. |

## Referências de implementação

- [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses): `responses.parse`, Pydantic e tratamento de respostas sem objeto estruturado.
- [Pyannote Community-1](https://huggingface.co/pyannote/speaker-diarization-community-1): entrada em memória e `output.speaker_diarization`.
- [Transformers pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines): classificação de áudio e ativação sigmoid para saída multirrótulo.
- [Modelo SER utilizado na aplicação](https://huggingface.co/superb/wav2vec2-base-superb-er): checkpoint compatível com a classificação de áudio do Transformers, com quatro classes.
- [Whisper](https://github.com/openai/whisper), [AST](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) e materiais dos módulos de análise de vídeo/áudio/texto, Textract/Comprehend e OpenAI fornecidos pela equipe.

As dependências têm faixas de versões compatíveis; este repositório não afirma reproduzir bit a bit ambientes de ML. Para uma entrega com inferência validada, congele também versões resolvidas, revisões dos pesos, sistema operacional e hardware usados no experimento.
