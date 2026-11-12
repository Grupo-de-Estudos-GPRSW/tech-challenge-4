# Plano de avaliação e roteiro de apresentação

## Validação de software

A suíte automatizada usa fixtures e mocks para executar sem internet, GPU, pesos ou chaves. Ela verifica contratos, regras temporais, fronteiras, falhas e renderização da interface. Testes de adaptadores validam o formato de integração com SDKs; não substituem testes reais com modelos.

Execute `python -m pytest` e registre a saída. CI repete testes em Python 3.11–3.13. O workflow também testa o container por HTTP. A execução da CI remota depende de publicar o projeto no GitHub; a configuração local não prova que houve deploy.

## Avaliação de IA a realizar pela equipe

Separe áudios autorizados com anotação humana e sem sobreposição de participantes entre desenvolvimento e avaliação. Informe quantidade, idioma, duração, hardware, versões e revisões dos modelos. Não use o cenário sintético como evidência de desempenho.

| Etapa | Métrica proposta | Anotação necessária |
|---|---|---|
| Transcrição | WER e CER | Transcrição humana e política de normalização |
| Diarização | DER com parâmetros de tolerância informados | Intervalos e falantes anotados |
| Eventos acústicos | Precisão, recall, F1 e falsos positivos por minuto | Tipo de evento e intervalo temporal |
| Emoção de voz | Macro-F1 e matriz de confusão | Múltiplos avaliadores e categoria indeterminada |
| Comparação voz/texto | Concordância entre avaliadores e revisão de erros | Critérios explícitos, sem interpretar como diagnóstico |
| Execução | Latência, uso de memória e tempo por minuto de áudio | CPU/GPU, tamanho do arquivo e cache frio/quente |

Calibre limiares apenas no conjunto de desenvolvimento. Apresente intervalos de incerteza quando houver amostra suficiente e exemplos de falhas. O SER em inglês exige cautela adicional em português. Não declare sensibilidade clínica ou detecção de condições médicas a partir desses testes de software.

## Roteiro sugerido

1. Apresentar escopo áudio + texto e arquitetura modular.
2. Mostrar demonstração offline, identificando os dados fictícios.
3. Usar um áudio curto autorizado, com modelos previamente baixados, para demonstrar análise real.
4. Conferir transcrição, eventos e falantes; habilitar semântica se houver autorização de envio.
5. Baixar relatórios e explicar scores, erros e limitações.
6. Executar pytest e mostrar os checks de uma execução real do GitHub Actions após publicação.
7. Explicar CD como entrega de imagem e distinguir de implantação em servidor.

## Checklist de inferência real

- SoundFile e SciPy instalados; arquivo de entrada WAV ou FLAC válido. FFmpeg não é necessário.
- Extras de ML instalados em ambiente limpo; versões de torch/torchaudio alinhadas.
- Pesos baixados com antecedência; acesso aos modelos e termos verificados.
- Áudio curto consentido, idioma configurado e referência humana disponível.
- Token de Pyannote e chave OpenAI somente quando as respectivas funções forem usadas.
- Evidências de tempo de execução e avaliação registradas sem expor dados pessoais.

Não há métricas de qualidade de IA medidas neste repositório até a equipe executar e documentar essa avaliação.
