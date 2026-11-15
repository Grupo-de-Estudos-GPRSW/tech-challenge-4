# Escopo e rastreabilidade

A entrega adota áudio + texto, conforme orientação do professor informada pela equipe.

| Item | Entrega |
|---|---|
| Transcrição | Whisper base, inglês inicial configurável |
| Eventos acústicos | AST, top 8 e um evento por janela |
| Diarização e paciente | Community-1 obrigatório e seleção automática |
| Emoção vocal/prosódia | SER ehcalabres, top 3, segmento completo e padding |
| Texto | OpenAI só emoção textual, ou Neutro sem cliente |
| Fusão | Regras locais de polaridade e eventos próximos |
| Relatórios | Markdown e CSV; JSON complementar |
| Interface e testes | Streamlit, testes de paridade, pytest e CI configurada |
| Vídeo/postura | Fora do escopo áudio + texto informado |
| Sinais vitais/prescrição | Não implementados |
| Monitoramento em tempo real | Não implementado; processamento por arquivo |
| Azure | Não implementado; tecnologias da equipe preservadas |

A correção de fidelidade revoga escolhas anteriores que trocavam o SER, usavam seleção manual de
paciente, AST multirrótulo, lotes de texto e guardas de Não avaliada. Essas decisões alteravam o
fluxo de análise. A versão atual aplica seleção automática, regras locais e percentuais sobre todos os segmentos do paciente.

A interface oferece upload validado e downloads em memória, com limites operacionais
e recursos complementares de visualização. As funções de análise ficam em módulos testáveis.
A [matriz funcional](fluxo-e-regras.md) registra a correspondência e as diferenças de infraestrutura.

As saídas são experimentais. Heurísticas de papel, emoção e contradição não comprovam diagnósticos
ou estados emocionais reais. A equipe deve apresentar essas limitações sem modificar a regra usada
na avaliação. Não há publicação, comunicação externa de alertas ou prescrição automática.


A [matriz de equivalência e testes](equivalencia-e-testes.md) registra o estado de cada mudança.
A correção da arquitetura de carregamento SER foi discutida, mas não implementada. As evidências
de modelos reais não certificam equivalência integral; a entrega mantém o checkpoint original
e explicita sua limitação, sem tratar aprovação de pytest como validação preditiva.
