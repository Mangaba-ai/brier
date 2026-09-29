# Brier — decisões tipadas e calibradas, local, no formato do Jev

**Brier** é um modelo de decisão (não de chat) com a mesma API do
[TypeSafe Jev](https://docs.typesafe.ai): recebe um `state` e perguntas tipadas (`choice`,
`score`, `noul`) e devolve escolhas com probabilidade e confiança. O nome vem do
[escore de Brier](https://en.wikipedia.org/wiki/Brier_score) (1950), a família de regras de
pontuação próprias que o treino otimiza: o objetivo é acertar a *probabilidade*, não só a resposta.

Base Qwen3-1.7B + LoRA (MLX, Apple Silicon) · ~0,12 s por pedido · offline · Apache-2.0.

**Estado atual (v1), medido nos mesmos 750 pedidos contra o Jev:** o Jev acerta mais
(76,0% × 70,4% no sintético; 78,2% × 65,0% no ASSIN2; empate no B2W). O Brier é mais bem
calibrado (NLL 0,71 × 1,06 no sintético e 0,73 × 1,86 no B2W; ECE 0,02 × 0,10) e erra com
menos confiança. Detalhes e limitações em [RESULTADOS.md](RESULTADOS.md).

Mesma entrada e saída do TypeSafe Jev: `POST /v1/systemone` com `state` + `questions`
(`choice`, `score`, `noul`) → `answers` com escolha, probabilidades e confiança. Roda local
no Mac (MLX), sem custo por token.

## O que o Jev é (fontes públicas) e o que este modelo reproduz

| Jev (TypeSafe) | Este modelo |
|---|---|
| Transformer que lê o state uma vez e responde tudo na mesma passada ("parallel sampler") | Qwen3-1.7B com **máscara de atenção em blocos**: prefixo (state) causal; cada pergunta vê o prefixo e a si mesma, nunca as outras. Uma passada, respostas independentes. |
| Não gera texto; saída sempre no tipo pedido | **Leitura por logits** só nos tokens-rótulo (`A`…`Z`, `0`…`9`); softmax restrito. 0% de erro de tipo por construção. |
| Treinado só com dados sintéticos | Sintético em PT-BR gerado pelo MiMo: 46 domínios × 9 formatos × 7 "armadilhas" (os pontos fracos que a própria TypeSafe documenta: negação, datas, contagem, duas etapas, ruído, manipulação, ambiguidade). |
| RLCD — RL para decisões calibradas | Perda de **regra de pontuação própria** (entropia cruzada contra alvo suave + termo ordinal). Com leitura por logits a política *é* a distribuição, então o gradiente da recompensa esperada é exato: é o RLCD sem variância de REINFORCE. Referência próxima: RLCR (Damani et al., MIT, 2025) e CARL (2026). |
| Confiança = concentração da distribuição | `confidence = 1 − H(p)/log K` |

**Rótulos que ensinam calibração:** cada state é rotulado duas vezes (quem gerou sabe a
resposta pretendida; um rotulador cego responde sem saber). Concordância → alvo firme (0,95);
discordância → alvo dividido (0,55/0,40). Caso ambíguo de verdade aprende a hesitar.

## Camadas estocásticas (de teses de doutorado)

| Camada | Origem | Onde |
|---|---|---|
| **MC Dropout** nos adaptadores LoRA — dropout ligado na inferência ≈ amostras da posterior | Yarin Gal, tese *Uncertainty in Deep Learning* (Cambridge, 2016) | `decisor.py`, modo `mc+perm` |
| **SWA / SWAG-diagonal** — média e variância dos pesos LoRA no fim do treino; amostra pesos | Pavel Izmailov, tese *Deconstructing Models and Methods in Deep Learning* (NYU, 2023) | `treinar.py` salva `swa_media` e `swag_var`; modo `swag+perm` |
| **Permutação estocástica das opções** — remove viés de posição | Zheng et al., *LLMs Are Not Robust Multiple Choice Selectors* (ICLR 2024) | todos os modos `+perm` |
| **Escala de temperatura** por tipo, ajustada na validação | Guo et al., *On Calibration of Modern Neural Networks* (2017) | `avaliar.py` |
| **Predição conformal (APS)** — conjunto que contém a resposta certa com ≥90% de garantia | Romano, Sesia & Candès (2020); Angelopoulos & Bates (2021) | `conformal.py`; campo `conjunto` |

Com T amostras, a incerteza total H(p̄) se divide em **aleatória** (média das entropias: o caso
é ambíguo) e **epistêmica** (informação mútua: o modelo não sabe). As duas saem em
`incerteza` em cada resposta. Campos extras não quebram clientes do Jev.

## Uso

Rodar o modelo treinado (adaptadores em `execucoes/v1`, o Qwen3-1.7B baixa sozinho do HF):

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/servidor.py --execucao execucoes/v1 --porta 8790
```

Reproduzir do zero (credenciais em variáveis de ambiente ou num `.env`; ver `src/chaves.py`):

```bash
.venv/bin/python src/gerar.py --tarefas 800 --paralelo 12          # dados sintéticos
.venv/bin/python src/externos.py                                    # ASSIN2 e B2W (rótulo humano)
.venv/bin/python src/treinar.py --passos 1100 --camadas 16 --max-prefixo 768 --saida execucoes/v1
.venv/bin/python src/avaliar.py --execucao execucoes/v1 --amostras 6   # ablação + calibração
.venv/bin/python src/servidor.py --execucao execucoes/v1 --porta 8790 # API compatível
TYPESAFE_API_KEY=… .venv/bin/python src/comparar_jev.py              # mesmo teste no Jev
```

```bash
curl -s localhost:8790/v1/systemone -d '{"model":"jev-latest","state":"internet caiu de novo, quero cancelar",
 "questions":{"setor":{"type":"choice","instructions":"Qual time?","criteria":{"suporte":"Queda","retencao":"Cancelamento"}}},
 "amostras":6}'
```

`amostras` (opcional, padrão 1) liga as camadas estocásticas: 1 passada ≈ 0,1 s; 6 ≈ 0,4 s.

## Resultados

Ver [RESULTADOS.md](RESULTADOS.md): ablação das camadas, conformal e comparação pareada com o Jev.

## Conteúdo

- `src/`: código (geração, treino, avaliação, servidor, comparação)
- `dados/sintetico.jsonl`: os 5.085 exemplos sintéticos do treino (gerados com MiMo)
- `execucoes/v1/`: adaptadores LoRA (melhor NLL, SWA e variância SWAG), calibração e métricas

Os conjuntos ASSIN2 e B2W não são redistribuídos: `src/externos.py` baixa e monta. As respostas
cruas do Jev também não: `src/comparar_jev.py` refaz com a sua chave.

## Licença

Apache-2.0, a mesma do Qwen3. Jev e TypeSafe são marcas da TypeSafe AI; este projeto é
independente e só reproduz o formato público da API.
