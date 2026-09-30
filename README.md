# Brier — decisões tipadas e calibradas, locais e independentes

**Brier** é um modelo de decisão (não de chat) com a mesma API do
[TypeSafe Jev](https://docs.typesafe.ai): recebe um `state` e perguntas tipadas (`choice`,
`score`, `noul`) e devolve escolhas com probabilidade e confiança. O nome vem do
[escore de Brier](https://en.wikipedia.org/wiki/Brier_score) (1950), a família de regras de
pontuação próprias que o treino otimiza: o objetivo é acertar a *probabilidade*, não só a resposta.

Brier v2: Qwen3-4B (4 bits) + LoRA em MLX, Apple Silicon · offline · Apache-2.0.
API própria `POST /v1/decide`; `POST /v1/systemone` aceita o mesmo corpo como rota de
compatibilidade para quem migra do Jev. O Brier não usa o Jev em nada: pesos próprios, dados
gerados e rotulados sem o Jev, e nenhuma resposta dele entra no treino.

**Estado atual (v2), pelo critério fixado antes de medir** (vencer em acurácia e escore de Brier,
com IC95%, em ≥2 de 3 conjuntos humanos): **ainda não é melhor que o Jev**, mas a distância caiu muito.
Empata no sintético, empata no B2W (onde ganha no escore de Brier), está à frente no tweetSentBR
(72,0% × 67,6%, IC encostando no zero) e perde só no ASSIN2 (72,8% × 78,2%; no v1 eram 65,0%).
Detalhes em [RESULTADOS.md](RESULTADOS.md).

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

Rodar o modelo treinado (adaptadores em `execucoes/v2`; o Qwen3-4B de 4 bits baixa sozinho do HF):

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/servidor.py --execucao execucoes/v2 --porta 8790
```

Reproduzir do zero (credenciais em variáveis de ambiente ou num `.env`; ver `src/chaves.py`):

```bash
.venv/bin/python src/gerar.py --tarefas 800 --paralelo 12          # dados sintéticos
.venv/bin/python src/externos.py                                    # ASSIN2 e B2W (rótulo humano)
.venv/bin/python src/treinar.py --passos 1100 --camadas 16 --max-prefixo 768 --saida execucoes/v1
.venv/bin/python src/avaliar.py --execucao execucoes/v1 --amostras 6   # ablação + calibração
.venv/bin/python src/servidor.py --execucao execucoes/v2 --porta 8790 # API compatível
TYPESAFE_API_KEY=… .venv/bin/python src/comparar_jev.py              # mesmo teste no Jev
```

```bash
curl -s localhost:8790/v1/systemone -d '{"model":"jev-latest","state":"internet caiu de novo, quero cancelar",
 "questions":{"setor":{"type":"choice","instructions":"Qual time?","criteria":{"suporte":"Queda","retencao":"Cancelamento"}}},
 "amostras":6}'
```

`amostras` (opcional, padrão 1) liga as camadas estocásticas: 1 passada ≈ 0,25 s no v2 (0,12 s no v1); 6 amostras ≈ 6× isso.

## Resultados

Ver [RESULTADOS.md](RESULTADOS.md): ablação das camadas, conformal e comparação pareada com o Jev.

## Conteúdo

- `src/`: código (geração, treino, avaliação, servidor, comparação)
- `dados/sintetico.jsonl`, `dados/sintetico_v2.jsonl`: sintéticos com rótulo de 2 votos e de maioria de 4 votos
- `dados/sintetico_foco*.jsonl`: 1.659 exemplos focados nas armadilhas perdidas pelo v1
- Conjuntos públicos (ASSIN, FaQuAD-NLI, HateBR, tweetSentBR) não são redistribuídos: `src/publicos.py` baixa e converte
- `execucoes/v2/`: adaptadores do v2 (padrão) e calibração
- `execucoes/v1/`: adaptadores do v1 (1,7B; melhor NLL, SWA e SWAG), calibração e métricas
- `execucoes/comparacao_v2.json`: comparação pareada v1 × v2 × Jev com o critério de vitória

Os conjuntos ASSIN2 e B2W não são redistribuídos: `src/externos.py` baixa e monta. As respostas
cruas do Jev também não: `src/comparar_jev.py` refaz com a sua chave.

## Licença

Apache-2.0, a mesma do Qwen3. Jev e TypeSafe são marcas da TypeSafe AI; este projeto é
independente e só reproduz o formato público da API.
