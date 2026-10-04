---
license: apache-2.0
language:
- pt
- multilingual
library_name: laya
pipeline_tag: text-classification
base_model: convaiinnovations/laya-multilingual
tags:
- brier
- mangaba-ai
- typed-decisions
- calibrated-decisions
- system-one
- rlcd
- portuguese
---

# Brier v4

**Modelo aberto de decisão tipada e calibrada, com foco em português.** Recebe um conteúdo (`state`)
e perguntas com respostas definidas (`choice`, `score`, `noul`) e devolve a opção escolhida com a
probabilidade de cada uma, numa única passada e sem gerar texto.

- **Arquitetura:** a do [Laya](https://github.com/NandhaKishorM/laya): encoder bidirecional
  mmBERT-base (322M parâmetros) com cabeça de decisão de 2 camadas; cada opção é um marcador na
  sequência, e a distribuição é o softmax das notas dos marcadores.
- **Pesos de partida:** [`convaiinnovations/laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual)
  (Apache-2.0), ajustados pela Mangaba AI com os dados do Brier.
- **Treino:** o mesmo laço publicado pelo Laya — reforço contra regras de pontuação próprias
  (log + esférica + RPS) mais entropia cruzada contra alvos suaves — sobre:
  - exemplos sintéticos em português, em 46 domínios, rotulados por maioria de votos de vários
    modelos (inclusive o mangaba-titan);
  - exemplos focados em casos difíceis: negação, contagem, implicação textual, ambiguidade;
  - conjuntos públicos com rótulo humano (ASSIN, FaQuAD-NLI, HateBR);
  - exemplos com instruções maliciosas injetadas no texto, com o rótulo certo preservado.
- **Código, servidor, MCP, LangChain e cliente TypeScript:** [github.com/Mangaba-ai/brier](https://github.com/Mangaba-ai/brier)

## Uso

```python
pip install "brier @ git+https://github.com/Mangaba-ai/brier"

from brier import Brier
b = Brier("mangaba-ai/brier-v4")
r = b.decide(
    "minha internet caiu de novo e quero cancelar",
    {"setor": {"type": "choice", "instructions": "Qual time atende?",
               "criteria": {"suporte": "Queda, lentidão", "financeiro": "Boleto", "retencao": "Cancelamento"}},
     "urgente": {"type": "noul", "instructions": "Precisa agir já?",
                 "criteria": {"true": "Serviço parado", "false": "Pode esperar"}}},
    min_confidence=0.7)
print(r["answers"])
```

Também carrega direto pelo pacote do Laya: `laya.load("mangaba-ai/brier-v4")`.

## Resultados

{{RESULTADOS}}

## Limitações

- Perguntas com muitas opções longas podem ser truncadas pela janela da cabeça de decisão.
- Os rótulos sintéticos vêm de modelos de linguagem; onde houve discordância entre eles, o alvo é
  suave, mas erros de rótulo restantes limitam a acurácia.
- Calibrado em português; em outros idiomas, use o roteador do pacote com o Laya multilíngue.

## Licença e créditos

Apache-2.0. Arquitetura, código de treino e pesos de partida: projeto Laya (Apache-2.0), de
NandhaKishorM / convaiinnovations. Ajuste fino, dados e avaliação: Mangaba AI.
