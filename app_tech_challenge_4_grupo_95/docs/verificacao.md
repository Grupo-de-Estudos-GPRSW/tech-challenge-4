# Verificação do fluxo da aplicação

## Revisão documental e de regressão — 14/11/2026

Revisados README, arquitetura, escopo, revisão funcional, plano de avaliação, relatórios reais,
CI/configuração e as suítes de serviços/interface. A [matriz de equivalência e testes](equivalencia-e-testes.md)
relaciona as mudanças com os testes e distingue utilitários complementares do fluxo efetivamente ativo.
As auditorias reais anteriores foram preservadas, sem repetir inferência ou chamadas faturadas.

Resultado atual: **199 testes aprovados, sem skips**, **99,29% de cobertura** do pacote.
Ruff lint e verificação de formatação aprovados. São 40 casos adicionais em relação aos 159 anteriores.

Complementos desta revisão:

- Constantes, mapas, payload, prompt e schema verificados em testes controlados.
- Ambiente de comparação recriado por caso de teste, eliminando compartilhamento de mocks entre testes.
- Política de IDs ausentes, desconhecidos e duplicados verificada no pipeline ativo.
- Critério da auditoria real testado contra etapas ausentes, bloqueadas, divergentes, SER com
  pesos novos/ignorados ou evidência desconhecida, falta de OpenAI e paciente vazio/inconsistente.
- Comparação final do script passou a conferir também os eventos produzidos pelo pipeline real.
- Auditoria SER considera tanto avisos de pesos novos quanto de pesos ignorados.

Essas alterações atingem documentação, testes e o critério dos relatórios de auditoria.
Não corrigem nem substituem a arquitetura SER usada pelo aplicativo.
Uma saída final coincidente, isoladamente, não autoriza mais `integral_equivalence_verified=True`.
Um relatório fictício positivo no teste não representa inferência real certificada.
Não foi adicionada dependência. Docker, build Python e CI remota não foram reexecutados nesta
revisão; seus resultados históricos e pendências permanecem descritos abaixo.

## Novo áudio 4r: verificação do pré-requisito SER

A [auditoria do novo áudio de 418,816 segundos](inferencia-real-4r.md) encontrou novamente
pesos de classificação ignorados/novos e emoções diferentes nos três trechos reais comparados.
Áudio pré-processado e prosódia coincidiram. Não foram executadas as demais etapas neste
arquivo após essa contraprova; a equivalência integral permanece não confirmada.

## Auditoria real com o WAV da equipe

A auditoria posterior está em [Inferência real](inferencia-real.md): áudio e 12 segmentos Whisper
iguais, 4 eventos AST iguais, primeira falha RuntimeError na diarização (repetição isolada com
11 turnos concluídos) e divergência entre carregamentos
independentes do SER original com aviso de pesos novos. OpenAI e comparação final não foram
executadas. **A equivalência integral real não está comprovada.** Os resultados de testes de
software abaixo continuam válidos para os cenários controlados, sem substituir essa auditoria.

## Estado atual — 14/11/2026

A revisão funcional substitui a adaptação anterior. O checkpoint ehcalabres,
o filtro automático do paciente, AST top 8/primeiro evento, chamada textual única, fallback Neutro,
regras sem guarda de Não avaliada e relatório clínico foram restaurados.

A suíte atual verifica regras, tabelas, relatórios,
os adaptadores e a interface. Resultado da revisão atual: **199 testes aprovados, sem skips**, com **99,29% de cobertura** do pacote.
Ruff lint e formatação aprovados. Os 72 casos parametrizados de níveis/explicações coincidiram
com os resultados esperados. Markdown e bytes do CSV também coincidiram no caso controlado.
Seleção de paciente, AST/fusão e prosódia também foram verificados em cenários controlados.
Nenhuma chamada autenticada de inferência é necessária nos testes: chaves são fictícias e modelos
substituídos. Os testes de UI isolam cache e impedem carregar credenciais do `.env` como cliente real.

Não foi concluída uma nova inferência completa com o checkpoint original restaurado. O histórico
abaixo registra o problema de pesos observado anteriormente; restaurar a chamada de carregamento não
comprova que esse problema desapareceu. Nenhum modelo substituto será usado automaticamente.

As adaptações de infraestrutura mantidas são upload validado WAV/FLAC, cache, visualização,
download em memória e tratamento de entradas vazias. Docker/CI remota e empacotamento têm validação separada da paridade de funções.

## Nova auditoria de paridade — 14/11/2026

Foi encontrada e corrigida uma divergência: os eventos próximos eram ordenados alfabeticamente,
enquanto `pandas.unique()` preserva a ordem de ocorrência. Agora a lista, o CSV e as
explicações com vários eventos mantêm a ordem da primeira ocorrência. Um teste compara o enriquecimento
completo de voz/eventos, incluindo repetições, e a explicação final. O metadado textual também foi
corrigido para refletir o modelo efetivamente usado pelo adaptador quando diferente do padrão.

A suíte passou de 156 para **159 testes aprovados**, cobertura **99,27%**, Ruff lint e formato aprovados.
O teste integrado verifica DataFrames finais de paciente e eventos usando áudio e saídas
de modelos controlados. Não se trata de inferência
com pesos reais. Foi conferida também a igualdade do prompt, schema OpenAI e constantes dos modelos.

Verificação adicional sem doubles de leitura: WAV PCM16 estéreo sintético em 8, 16, 44,1 e 48 kHz,
convertido para mono/16 kHz. O caminho de upload com reamostragem por librosa e `librosa.load()`
produziram arrays exatamente iguais nos quatro casos (erro máximo zero). Foram usados arquivos
apenas em diretório temporário, sem dados pessoais, rede ou API. Versão de librosa: 0.11.0.

A equivalência comprovada cobre regras, tabelas controladas, relatório, schema e pré-processamento
nesses casos. Permanecem as diferenças operacionais documentadas: interface/upload limitado,
validadores de intervalos/scores, tratamento de vazio e download em memória. Carregadores de arquivos
usados pelo Whisper/Pyannote nas auditorias são substituídos por arrays/tensores no aplicativo.
Isso não comprova igualdade de inferência para todos os formatos ou ambientes. A execução real
integral com HF/OpenAI e a condição histórica do checkpoint SER continuam pendentes de validação.

## Histórico anterior à correção de fidelidade

**Os resultados e decisões a seguir descrevem versões anteriores.** Referências a seleção manual,
SER superb, AST multirrótulo, lotes e Não avaliada não descrevem o fluxo atual. Os totais de testes
anteriores também não devem ser usados como resultado desta revisão.

### Registro da entrega anterior

Este documento reúne verificações históricas de 11/11/2026 e a revisão funcional de 14/11/2026, em Windows com Python 3.13.12 na `.venv` do projeto. A tabela de testes e cobertura abaixo reflete a execução; resultados históricos de build, servidor e inferência são identificados separadamente.

| Verificação | Resultado |
|---|---|
| pytest após revisão dos testes funcionais (14/11/2026) | 107 testes aprovados, sem skips |
| Cobertura do pacote `clinical_audio` | 99,35% de statements; mínimo exigido de 80% |
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

Após identificar os pacotes de IA ausentes, o extra `ml` foi instalado na `.venv` local. `pip check` concluiu sem incompatibilidades declaradas. Os imports de Whisper, torch, torchaudio, librosa e `transformers.pipeline` foram executados com sucesso, sem carregar pesos. Versões verificadas: torch 2.15.0+cpu, torchaudio 2.11.1+cpu e librosa 0.11.0. A diarização é opcional. Naquele momento ainda exigia instalar o extra correspondente, além de configurar HF_TOKEN; a instalação completa atual via `requirements.txt` já inclui essa biblioteca.

O extra `diarization` também foi instalado posteriormente na `.venv` local, com `pyannote.audio` 4.0.7. O import de `pyannote.audio.Pipeline`, a verificação de presença de todos os módulos de IA/diarização e `pip check` concluíram com sucesso. O modelo Community-1 não foi baixado ou executado nessa verificação; seu uso ainda exige HF_TOKEN e aceite dos termos pela conta da equipe.

No diagnóstico do arquivo `START_VIDEO_RECORDING_ky56r.wav` fornecido pela equipe, a leitura confirmou WAV PCM de 16 bits, mono, 16 kHz e duração de 46,232 segundos. A transcrição local com Whisper concluiu. Ao testar a diarização com a configuração local, o Hugging Face retornou `GatedRepoError` para Community-1. O código foi ajustado para continuar sem identificação dos falantes nessa situação. Nenhum áudio ou transcrição foi enviado a serviços de análise externos; houve somente acesso ao Hugging Face para autenticação e obtenção dos modelos.

A detecção AST concluiu no mesmo arquivo. No carregamento do SER original, o Transformers informou que ignorou os pesos `classifier.dense`/`classifier.output` e criou novos pesos de classificação/projeção. Essa execução foi interrompida sem aceitar suas previsões como resultado. O modelo foi substituído por `superb/wav2vec2-base-superb-er`, compatível com a arquitetura automática e com quatro categorias. A implementação passou a validar o relatório de carregamento dos pesos.

O checkpoint substituto carregou com a verificação de pesos aprovada. A classificação de voz e a extração de pitch/RMS concluíram em um trecho de cinco segundos do WAV fornecido. Esse teste manual foi executado em CPU, com duas threads PyTorch, sem OpenAI. A transcrição e o AST foram exercitados no áudio completo; o SER substituto foi exercitado no trecho de cinco segundos, não em uma nova execução completa de ponta a ponta. A diarização real permanece sem validação de inferência enquanto o acesso ao Community-1 não for liberado.

Docker não estava disponível no ambiente local. A imagem e o workflow foram preparados, mas a construção/smoke test do container e a execução remota do GitHub Actions precisam ocorrer no ambiente da equipe. Nenhuma publicação, release ou implantação externa foi realizada.

Para repetir os testes, siga o README. Todos os testes de leitura são executados sem FFmpeg, sem necessidade de instalação ou configuração de PATH para esse programa.


## Revisão funcional — 14/11/2026

Execução inicial da adaptação: 86 testes aprovados, cobertura de 99,35% do pacote.
Foram verificados payload restrito a ID/transcrição, contrato de saída só com emoções,
lotes, recusa da API, alinhamento de IDs, regras locais nos dois sentidos,
limite estrito de score > 0,6, prioridade da oposição direta sobre eventos próximos,
não avaliação de categorias sem polaridade e ausência de dados, relatórios e interface.
A demonstração usa as mesmas regras do pipeline real. Esta validação usa doubles,
sem chamada faturada à API nem nova inferência com pesos de ML.

Ruff check e format --check aprovados (27 arquivos Python).
O empacotamento nesta execução não foi concluído: o build isolado apresentou
UnicodeDecodeError ao reportar falha do subprocesso no Windows; a tentativa
`python -m build --no-isolation` identificou `wheel>=0.45,<1` ausente na venv.
Esses resultados não alteram a aprovação dos testes e não validam um novo pacote distribuível.


## Revisão documental

README, arquitetura, escopo e avaliação foram reconciliados com os contratos e regras
implementados. O documento [Revisão funcional](fluxo-e-regras.md) detalha níveis, campos exportados,
falhas e limites operacionais. Esta revisão altera somente documentação:
os 86 testes e a cobertura acima pertencem à execução da adaptação, não a uma nova execução.
Nenhuma dependência foi adicionada pela revisão funcional. `wheel>=0.45,<1` já consta no `requirements.txt`,
no extra `dev` e nos requisitos de build; a ausência observada era da instalação local.


## Revisão dos testes funcionais — 14/11/2026

Resultado final: **107 testes aprovados, sem skips**, cobertura de **99,35%**.
Ruff lint e verificação de formatação aprovados. A suíte passou de 86 para 107 casos.
Não houve alteração no código de produção nem adição de dependências nesta revisão.

| Área | Verificações complementares |
|---|---|
| Contrato remoto | Payload exato com apenas ID/texto mesmo com segmentos enriquecidos; schema de saída sem resumo ou discordâncias; ausência de chamada com lista vazia |
| IDs | Rejeição de ausentes, duplicados e desconhecidos no adaptador; junção correta quando a resposta vem fora de ordem |
| Falha em lotes | Recusa, timeout e IDs inválidos no segundo lote descartam análise parcial e preservam dados locais |
| Regras | Score zero com emoções reconhecidas fica Não avaliada; score positivo baixo fica Moderada em oposição; limites temporais inclusivos de ±2,5 segundos |
| Resumo | Percentual exclui Não avaliados; zero avaliados não causa divisão por zero |
| Voz secundária | Score do segundo rótulo preservado; fallback sem segunda hipótese usa score zero |
| Exportações | CSV/JSON preservam emoção secundária, score e resultado local; Markdown contém prosódia e eventos; booleano falso mantém distinção entre Nenhuma e Não avaliada |
| Interface | Upload WAV com OpenAI/Pyannote substituídos, resposta fora de ordem e cálculo local; filtro reduz tabela sem modificar resultado completo nem repetir chamada textual |

Os testes anteriores continuam cobrindo prioridade da divergência direta, todos os níveis,
limite estrito de 0,6, processamento local sem OpenAI e proteções de áudio/diarização.
Essa evidência valida as adaptações modulares descritas em `fluxo-e-regras.md`,
sem executar comandos de instalação ou carregar modelos reais.

Limites: modelos e API são substituídos nos testes; não se mede qualidade preditiva,
custo real, desempenho dos pesos ou integração autenticada de ponta a ponta. Aumento
de casos mantendo a mesma cobertura de linhas é esperado: os novos casos verificam
contratos e cenários distintos em caminhos já percorridos pela suíte anterior.
