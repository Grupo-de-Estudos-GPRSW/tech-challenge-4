# Fluxo e regras da aplicação

Esta revisão substitui a adaptação anterior, que mantinha diferenças de modelo, filtragem e regras.

## Correspondência funcional

| Etapa de análise | Projeto |
|---|---|
| AST top 8, primeiro evento elegível por janela | `LocalModels.classify_events` e `AnalysisPipeline` |
| Laço de janelas, arredondamento, fusão com último evento | Pipeline com os mesmos limiares e ordem |
| Whisper base/en, tempos com duas casas, IDs originais | `LocalModels.transcribe`; inglês inicial na interface |
| Community-1 no áudio completo | Diarização obrigatória, sem fallback de falantes |
| `get_patient_speaker_id_acoustically` | Função do módulo de análise clínica |
| `filter_patient_segments` | Função aplicada antes de SER/OpenAI |
| SER ehcalabres, top 3, padding até 0,2 s | `LocalModels.ser` e `voice_features`, sem corte de duração |
| pYIN de C2 a C6, F0 médio/desvio e RMS | Mesmos limites e arredondamentos |
| OpenAI só texto/IDs, uma chamada, temperatura zero | `OpenAISemantic`, Chat Completions estruturado |
| Sem cliente: emoção Neutro | Pipeline sem cliente usa Neutro em cada fala do paciente |
| ID ausente: Neutro; duplicado: primeiro | Junção local reproduz a escolha |
| Polaridades, prioridade, score > 0,6 e explicações | Comparação local de emoções e eventos |
| Relatório/CSV em português | Markdown e CSV com os campos da análise |

Ordem: áudio → AST → Whisper → diarização → filtro automático → SER/prosódia → emoção textual →
comparação local → relatório. Falhas de diarização, voz ou API ativa interrompem o fluxo.
Ausência do cliente OpenAI é o fallback Neutro; não se aplica esse fallback a uma falha de chamada.

## Regras e arredondamento

Positivos: alegria, calma, neutro, tranquilidade, otimismo, bem-estar, positivo.
Negativos: tristeza, raiva, medo, ansiedade, desconforto, aversão, abatimento, irritação, negativo.
A identificação é por substring em minúsculas.

1. Oposição entre polaridades: Alta com score > 0,6; Moderada nos demais scores.
2. Sem oposição, texto positivo/neutro e eventos próximos: Leve.
3. Demais casos: Nenhuma.

Não há exclusão de score zero nem de rótulos desconhecidos. Voz e eventos têm score arredondado
a três casas, RMS a quatro; tempos a duas; pitch médio/desvio a uma. O percentual usa todos os
segmentos do paciente, sem denominador separado de avaliados.

## Contrato OpenAI

Payload enviado em uma única chamada:

```json
[{"segment_id": 1, "transcricao": "Estou tranquilo."}]
```

Resposta solicitada:

```json
{"segmentos_analisados": [{"segment_id": 1, "emocao_discurso_texto": "Calma"}]}
```

O contrato textual usa os campos do payload e da resposta apresentados acima. A OpenAI não recebe voz, scores ou eventos.
Não há lote de 15, `responses.parse` ou instruções adicionais de classificação na nova execução.
O modelo padrão é gpt-4o-mini, configurável por OPENAI_MODEL. Chamadas reais podem variar mesmo com
temperatura zero. Não é feito preenchimento de Neutro em caso de recusa da API: a execução falha.

## Saídas e interface

`analise_segmentos_clinicos.csv` preserva as colunas e ordem dos dados enriquecidos e da comparação:
segment_id, inicio_s, fim_s, transcricao, emocao_voz_primaria, confianca_voz_primaria,
emocao_voz_secundaria, confianca_voz_secundaria, pitch_medio_hz, pitch_std_hz, energia_rms,
eventos_nao_verbais_proximos, emocao_discurso_texto, interlocutor, ha_discordancia,
nivel_discordancia e explicacao_clinica. `interlocutor` é Paciente, conforme a heurística de seleção.
A tabela de eventos inclui também evento_en e prob_fala_simultanea.
Eventos próximos preservam a ordem da primeira ocorrência, sem ordenação alfabética, como
`pandas.unique()`. Isso também preserva o texto das explicações com vários eventos.

O Markdown inclui resumo quantitativo, tabela de eventos, comparativo e detalhamento
das discordâncias. O código retorna o texto em memória para download. Os testes verificam
o conteúdo do relatório e os bytes do CSV.
CSV complementar e JSON mantêm contratos internos em inglês para compatibilidade com a interface.

A infraestrutura inclui upload WAV/FLAC validado, limites de tamanho, cache, downloads em
memória e interface Streamlit.
Na aplicação, SoundFile lê o upload e librosa reamostra para 16 kHz. Arquivos fora dos
limites continuam recusados; isso é uma restrição da interface, sem trocar o algoritmo de análise.
O caso sem segmentos selecionados produz resultado vazio em vez de falhar na montagem do DataFrame.

## Evidência e limites

Os testes funcionais verificam níveis e explicações em 72 combinações, seleção
de paciente, detector/fusão e relatório. Outros testes verificam adaptadores, UI e falhas. Uma comparação integrada também verifica
a tabela completa do paciente, com transcrição, detecção de eventos, filtragem, enriquecimento
de voz e junção textual controlados.
Os testes não demonstram uma execução real completa do checkpoint restaurado e dos serviços.
O problema de carregamento do SER foi confirmado nas duas auditorias reais, registradas em
`verificacao.md`; não se substitui o modelo pedido. Igualdade numérica em inferência exige controlar ambiente e pesos.
Nenhuma nova dependência foi necessária.


## Estado da inferência real e das correções propostas

As auditorias com os dois WAVs confirmaram o problema de carregamento SER; ele não é somente
um aviso histórico. A arquitetura de classificação ainda não foi corrigida no aplicativo. A igualdade funcional dos casos controlados não garante reprodução das emoções vocais.
A [matriz de equivalência e testes](equivalencia-e-testes.md) identifica cada alteração e distingue
as propostas ainda não implementadas dos resultados já verificados.
