"""Mensagens de recuperação sem expor payloads ou credenciais de exceções."""


def processing_error_message(error: Exception, stage: str) -> str:
    chain = []
    current = error
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    details = " ".join(str(item).lower() for item in chain)
    kinds = {type(item).__name__.lower() for item in chain}
    if isinstance(error, MemoryError) or "out of memory" in details:
        advice = (
            "Memória insuficiente. Use Whisper tiny, feche outros processos e tente um áudio menor."
        )
    elif "gatedrepoerror" in kinds or any(term in details for term in ("401", "403", "gated repo")):
        advice = "Acesso ao modelo negado. Confira HF_TOKEN e o aceite dos termos do modelo na mesma conta do Hugging Face."
    elif any(
        term in details
        for term in ("ssl", "certificate_verify_failed", "certificate verify failed")
    ):
        advice = "Falha no certificado HTTPS ao obter o modelo. Confira os certificados da rede/proxy; não desative a verificação SSL."
    elif any(
        term in details
        for term in ("couldn't connect", "connection", "network", "huggingface.co", "timed out")
    ):
        advice = "Não foi possível obter o modelo. Verifique acesso HTTPS ao Hugging Face, proxy e conexão; os pesos precisam estar disponíveis no cache."
    elif isinstance(error, PermissionError):
        advice = "Sem permissão para acessar o cache dos modelos. Confira as permissões da pasta de cache."
    else:
        advice = "Confira a compatibilidade dos modelos e dependências, memória e configuração."
    return f"Falha na etapa '{stage}' ({type(error).__name__}). {advice}"
