# Estado da equivalência e rastreabilidade dos testes

Este documento separa o que foi implementado, o que foi testado com entradas controladas
e o que foi observado com modelos reais. Uma aprovação de pytest não certifica inferência integral.

## Estado das alterações

| Alteração | Estado e evidência automatizada |
|---|---|
| Ordem AST → Whisper → Community-1 → filtro → voz → texto → regras | Implementada; `test_pipeline.py` |
| AST top 8, primeiro evento, janelas e fusão de eventos | Implementada; adaptador em `test_backends.py` e testes de comparação funcional |
| Checkpoints/padrões, mapas, inglês/base e prompt originais | Verificados por testes de constantes, mapas e contrato textual |
| Paciente escolhido automaticamente antes de SER/OpenAI | Implementada; paridade da heurística e comparação integrada da tabela do paciente |
| Padding de 0,2 s, voz completa, top 3, pYIN C2–C6 | Implementados; probes controlados de 20, 16.000 e 400.000 amostras, com comparação de prosódia |
| Arredondamento de scores e tempos | Adaptadores e comparação integrada das tabelas |
| Eventos próximos em ordem de primeira ocorrência | Corrigida a ordenação alfabética; teste com Tosse/Espirro/Tosse e explicação resultante |
| Texto somente por IDs/transcrições, chamada única/temperatura zero | Payload, prompt e schema verificados para 0, 1 e 17 segmentos |
| Sem cliente: Neutro; ID ausente: Neutro; duplicado: primeiro | Testes do pipeline, incluindo ID desconhecido e duplicatas com emoções diferentes |
| Falha de API ativa/voz/diarização interrompe execução | Testes de propagação; não se converte erro em Neutro |
| Polaridades, níveis e explicações locais | 72 combinações verificadas em testes controlados |
| CSV/Markdown nativos | Verificação do texto Markdown e dos bytes CSV em diretório temporário |
| Tabela integrada e eventos finais | DataFrames do pipeline verificados com saídas de modelos controladas |
| Interface/cache/configuração/downloads | AppTest com WAV codificado em memória; cache isolado e chaves fictícias |
| Decisão de equivalência integral | `verification.py` e `test_verification.py`: etapas ausentes, falhas, pesos novos/ignorados, divergências ou fallback não aprovam uma auditoria integral com OpenAI |

Os nomes completos dos casos adicionais podem ser listados com `python -m pytest --collect-only -q`.
O ambiente de comparação dos testes é recriado por caso, evitando compartilhamento de mocks entre testes.

## Equivalência implementada versus pendências

A aplicação utiliza as heurísticas descritas no fluxo de análise. Ela não recebeu uma correção
de arquitetura SER: o carregador continua sendo o `pipeline` do checkpoint ehcalabres.
Os dois [relatórios reais](inferencia-real.md) e [do áudio 4r](inferencia-real-4r.md) mostram que
pesos da classificação foram ignorados e novos pesos foram inicializados no ambiente auditado.
Esse problema não está resolvido, e o modelo não foi substituído automaticamente.

Também não foram aplicadas como se estivessem concluídas as propostas discutidas de congelar
versões/revisões de pesos, corrigir o carregamento do classificador ou reutilizar uma resposta
OpenAI gravada para isolar variações do provedor. `requirements.txt` mantém faixas de versões,
não um ambiente de ML reproduzível congelado. Essas são próximas etapas, não resultados existentes.
Fixar apenas uma semente não recupera os pesos treinados descartados pelo carregador.

Diferenças operacionais mantidas: upload validado WAV/FLAC com limites, arrays/tensores no lugar dos
carregadores por arquivo de Whisper/Pyannote, cache, resultado vazio tratado e downloads em memória.
Recursos adicionais incluem JSON, CSV complementar e menções.

## Utilitários complementares e fluxo ativo

Existem utilitários conservados e testados que não definem o algoritmo ativo da aplicação:

| Utilitário | Diferença para o fluxo de avaliação |
|---|---|
| `assign_speakers` | Soma sobreposições; o pipeline usa `filter_patient_segments`, com maior turno individual |
| `merge_events` | Agrupa por categoria; o pipeline faz fusão apenas com o último evento |
| `windows` | Inclui a cauda; o pipeline usa o laço nativo da aplicação, que não adiciona essa cauda |
| `validate_text_alignment` | Validação estrita de IDs; não é usada na junção textual, que aceita ausentes/duplicados com fallback/primeiro |
| `decode_upload` sem opção | Pode usar SciPy; a interface usa explicitamente a reamostragem com librosa |

Os testes desses utilitários verificam seus contratos próprios; não devem ser apresentados como
regra aplicada ao paciente. Os testes do pipeline e de paridade verificam o fluxo efetivamente ativo.

## Critério dos scripts de auditoria

`verification.integral_equivalence_verified` exige conjuntamente:

- Pré-processamento, Whisper, AST, diarização e comparação final concluídos, com igualdade exata.
- SER concluído, sem avisos de pesos novos ou ignorados e com previsões iguais entre carregamentos.
- Eventos produzidos pelo pipeline real iguais nos dois caminhos auditados na comparação final.
- OpenAI habilitada e a etapa final concluída nos dois caminhos.
- Quantidade positiva e igual de segmentos do paciente nos dois resultados.

Ausência de evidência é reprovação, não aprovação. O critério é estrito: ele não aplica tolerância
numérica implícita. Um fluxo com fallback Neutro pode reproduzir a regra da aplicação sem cliente, mas não
certifica uma comparação que inclua inferência real da OpenAI. Temperatura zero não garante que
chamadas independentes ao provedor sejam numericamente ou semanticamente idênticas.

O teste positivo do critério usa um relatório fictício; não significa que exista uma execução
real integral aprovada. As auditorias reais existentes permanecem sem aprovação integral.

## Executar as verificações

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pytest
```

Os testes de software funcionam com a instalação leve `.[dev]`, sem internet, modelos, GPU ou
chaves reais. Não exigem os arquivos privados em `D:\downloads`, nem relatórios em `outputs`.
A CI executa essa suíte em Python 3.11–3.13 quando o workflow é disparado no repositório publicado;
a execução local não comprova que a CI remota ou o container passaram.

As auditorias de inferência são explícitas e separadas de pytest:

```powershell
.\.venv\Scripts\python.exe -u scripts/verify_real_parity.py 'D:\downloads\START_VIDEO_RECORDING_ky56r.wav'
.\.venv\Scripts\python.exe -u scripts/verify_real_parity.py 'D:\downloads\START_VIDEO_RECORDING_ky56r.wav' --diarization-only
.\.venv\Scripts\python.exe -u scripts/verify_audio_checkpoint.py 'D:\downloads\START_VIDEO_RECORDING_4r.wav'
```

A auditoria completa exige dependências ML/diarização, pesos, acesso ao Community-1 e, para incluir
OpenAI, chave válida e autorização para enviar transcrição. Pode baixar modelos, gerar cobrança e
consumir bastante tempo em CPU. O diagnóstico `verify_audio_checkpoint.py` usa somente o cache
local e três trechos reais; nunca certifica sozinho uma execução integral.
O parâmetro `--access-only` do primeiro script testa acesso à configuração do Community-1, sem
provar carregamento de todos os pesos ou execução do modelo.

O primeiro script grava em `outputs/real-parity`; a auditoria SER grava em subdiretório com o nome
do áudio. O primeiro pode sobrescrever seus resultados anteriores: preserve a pasta se precisar
comparar diferentes execuções. A pasta é ignorada pelo Git e pode conter transcrição privada.
O FFmpeg temporário usado na auditoria do carregamento por arquivo não é dependência nova
da aplicação. Ruff exclui `outputs` e `data`, incluindo bibliotecas externas de ferramentas temporárias.


## Resultado da revisão

Na execução local de 14/11/2026: **199 testes aprovados, sem skips**, cobertura **99,29%**,
Ruff lint e formatação aprovados. Esta revisão não repetiu auditorias com modelos reais,
não chamou a OpenAI e não corrigiu o checkpoint. Os dois relatórios reais continuam sendo
evidência dos limites atuais, sem certificação integral.
