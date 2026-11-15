# Avaliação e apresentação

## Fidelidade funcional

Os testes funcionais verificam as regras e as saídas em ambiente
controlado. Verificam níveis e explicações, inclusive score zero e emoções desconhecidas; seleção
automática do paciente; janelas, primeiro evento e fusão; Markdown idêntico. Os adaptadores são
verificados com doubles para top-k, checkpoint, padding, duração completa, arredondamentos,
limites de pitch, chamada OpenAI única, temperatura zero e nomes do schema/payload.

Testes da UI usam WAV real codificado em memória com modelos substituídos. Não devem usar chaves
reais nem fazer chamadas de rede; caches de modelos são limpos entre casos. A execução real exige
HF_TOKEN válido, acesso ao Community-1 e, quando habilitada, chave OpenAI.

## Roteiro de apresentação

1. Apresentar as etapas de análise e os resultados disponíveis no aplicativo.
2. Usar Whisper base, inglês e OpenAI gpt-4o-mini como configuração padrão.
3. Enviar um WAV autorizado, com credenciais e pesos preparados.
4. Conferir o ID automaticamente selecionado e as falas do paciente antes de interpretar resultados.
5. Conferir eventos, emoções primária/secundária, scores arredondados e pitch/RMS.
6. Mostrar que a OpenAI extrai apenas emoção textual; sem cliente, explicar o fallback Neutro.
7. Baixar `relatorio_discordancias_clinicas.md` e `analise_segmentos_clinicos.csv` e conferir os resultados exportados.
8. Apresentar pytest e os limites da evidência de software.

Registre versões, revisões de pesos, sistema, CPU/GPU, idioma e configuração. Não afirme igualdade
numérica apenas porque a suíte passou. O checkpoint ehcalabres restaurado exige validar o carregamento
real: as duas auditorias confirmaram avisos de pesos ignorados/novos no ambiente local. Não trocar o
modelo silenciosamente para completar a apresentação.

## Qualidade de IA

Separe áudios autorizados, com referências humanas e participantes distintos entre desenvolvimento
e avaliação. A demonstração offline serve apenas para visualizar a interface.

| Etapa | Métrica sugerida |
|---|---|
| Whisper | WER/CER |
| Community-1 | DER e revisão da heurística de papel |
| AST | Precisão, recall, F1 e falsos positivos por minuto |
| Emoção vocal/textual | Macro-F1, matriz de confusão e concordância entre anotadores |
| Discordâncias | Revisão de falsos positivos e erros das polaridades/eventos próximos |
| Execução | Tempo por minuto de áudio, memória e cache frio/quente |

Preserve os limiares configurados ao comparar execuções. Calibração de novos limiares é um
experimento separado. Eventos próximos podem ser produzidos por outro falante; a heurística de
paciente pode errar; score zero ainda pode produzir Moderada; rótulos desconhecidos podem produzir
Nenhuma ou Leve. Essas são características das heurísticas, não garantias clínicas.

Não há avaliação preditiva completa realizada por esta suíte. Consulte [a verificação](verificacao.md).


## Critério de aceite da equivalência

Consulte [a matriz de critérios e testes](equivalencia-e-testes.md). Separe igualdade funcional,
carregamento válido dos pesos e inferência integral. A igualdade dos classificadores não foi
comprovada: os dois áudios auditados apresentaram pesos novos/ignorados no SER. A correção conjunta
do carregamento, o congelamento do ambiente e a validação final permanecem pendentes.
A classificação semântica pela OpenAI deve ser avaliada separadamente da comparação local;
resposta textual gravada pode isolar a regra local, mas não substitui duas inferências reais.
