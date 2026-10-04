# Resultados — Brier v4 (04/10/2026)

> O Jev (TypeSafe AI) aparece só como referência externa de desempenho.

**Modelo:** arquitetura e pesos do Laya como estão (`convaiinnovations/laya-multilingual`, mmBERT 322M,
Apache-2.0), sem ajuste fino. Mesmos 250 pedidos por conjunto usados no v2, mesmo critério.

| Conjunto | Acerto v4 | Acerto Jev | Escore de Brier v4 | Escore de Brier Jev | IC95% Δacerto (v4−Jev) | Latência mediana |
|---|---|---|---|---|---|---|
| Sintético | 44,3% | 79,5% | 0.750 | 0.287 | -39.4 a -30.8 pts | 117 ms |
| ASSIN2 | 45,4% | 78,2% | 0.767 | 0.312 | -37.4 a -28.4 pts | 58 ms |
| B2W | 50,6% | 66,8% | 0.809 | 0.527 | -21.0 a -11.6 pts | 84 ms |
| tweetSentBR | 56,4% | 67,6% | 0.594 | 0.459 | -18.0 a -4.8 pts | 71 ms |

**Vitórias pelo critério: 0.**

- **Injeção** (famílias de frase nunca vistas): o ataque funcionou em 44,1% dos casos (Jev: 21,4%); acerto 47,6% → 38,6%.
- **Documentos longos** (~2.500 tokens, lidos por janelas): acerto 46,6% → 33,2% (Jev: 78,2% → 73,0%); mediana de 3.5 s.

**Leitura:** sem ajuste com dados em português, os pesos do Laya ficam de 15 a 33 pontos abaixo do
Brier v2 em acerto, e atrás do Jev em todos os conjuntos, na injeção e nos documentos longos. A vantagem
é a velocidade: 58–117 ms por pedido no Mac M5, contra ~1,3 s do v2. O v2 segue sendo a versão mais
precisa do Brier.

---
# Resultados — Brier v2 (30/09/2026)

> O Brier é um modelo independente. O Jev (TypeSafe AI) aparece neste documento só como referência
> externa de desempenho: nenhuma resposta dele entra no treino, e as respostas coletadas não são
> redistribuídas.

**Base:** Qwen3-4B-Instruct-2507, 4 bits. LoRA rank 16 nas últimas 20 camadas (18 M parâmetros), treinado no MLX.

**Treino:** 680 passos de 4 exemplos. O melhor checkpoint foi o passo 450; parei no 680 porque a validação piorou no 600 (sobreajuste).

**Dados de treino:** 6.744 sintéticos (5.085 originais + 1.659 focados nas armadilhas em que o v1 perdeu) e 15.128 públicos com rótulo humano (ASSIN v1, FaQuAD-NLI, HateBR), sorteados em 30% dos exemplos.

**Rótulos:** maioria de 4 votos (gerador + 3 rotuladores cegos de modelos diferentes). A cota do provedor acabou no meio, então só 940 exemplos (2.942 perguntas) receberam os 4 votos. O resto ficou com o rótulo de 2 votos do v1.

## Critério de vitória (fixado antes de medir)

O Brier só é declarado **melhor que o Jev** se, em pelo menos 2 dos 3 conjuntos com rótulo humano (ASSIN2, B2W, tweetSentBR), o IC de 95% (bootstrap pareado) for inteiramente favorável ao Brier em **acurácia e em escore de Brier ao mesmo tempo**. Nenhum desses três conjuntos entra no treino.

## Comparação pareada (mesmos pedidos, item a item)

| Conjunto | Acurácia v1 / **v2** / Jev | Escore de Brier ↓ v1 / **v2** / Jev | IC95% Δacc v2−Jev | IC95% Δbrier (Jev−v2, >0 = v2 melhor) |
|---|---|---|---|---|
| Sintético (maioria de 4 votos) | 72.5 / **77.5** / 79.5% | 0.380 / **0.304** / 0.287 | -4.8 a +1.2 pts | -0.046 a +0.015 |
| ASSIN2 | 65.0 / **72.8** / 78.2% | 0.466 / **0.375** / 0.312 | -9.8 a -1.2 pts | -0.106 a -0.021 |
| B2W | 65.8 / **65.4** / 66.8% | 0.419 / **0.459** / 0.527 | -4.8 a +2.0 pts | +0.034 a +0.106 |
| tweetSentBR | 68.4 / **72.0** / 67.6% | 0.446 / **0.421** / 0.459 | -0.4 a +9.2 pts | -0.019 a +0.095 |

**Vitórias pelo critério: v1 = 0, v2 = 0. Ainda não é "melhor que o Jev".**

## Leitura

- **O v2 fechou boa parte da distância.** No sintético, o v2 empata estatisticamente com o Jev; o v1 perdia com IC inteiramente negativo.
- **ASSIN2 é o único conjunto em que o Jev ainda ganha com clareza:** 72,8% × 78,2%, contra 65,0% × 78,2% no v1.
- **B2W:** empate no acerto, e o v2 **ganha no escore de Brier** (IC favorável). O Jev erra escala com muita confiança.
- **tweetSentBR:** o v2 está 4,4 pontos à frente (72,0% × 67,6%), mas o IC encosta no zero. Com 250 exemplos, falta poder estatístico para declarar vitória.
- **Os rótulos por maioria mudaram a régua:** no sintético, o Jev sobe de 76,0% para 79,5%. Parte dos "erros" dele eram rótulos errados do gerador: a maioria de 4 votos discorda da intenção do gerador em 13,6% das perguntas.
- **A base maior foi o que mais pesou.** O Qwen3-4B sem treino já acertava 72,8% na validação, mais que o v1 treinado (68,2%).

**Conformal (α = 0,1, teste sintético):** com cobertura de 98–100%, o v2 decide sozinho (conjunto de
1 opção) em 39,7% das perguntas de choice e em 41,7% das de noul. No v1 eram 27,6% e 21,9%. Mais
casos podem ser automatizados com a mesma garantia de acerto.

## Próximos passos

1. Terminar a rotulagem por maioria nos ~4.100 exemplos restantes, com uma chave separada da de produção (ver incidente abaixo).
2. Avaliar com n = 1.000 por conjunto: no tweetSentBR, a vantagem atual pode virar vitória estatística só com mais amostra.
3. Mais exemplos de implicação textual, que é onde o Jev ainda ganha (ASSIN2).

## Incidente: cota do provedor

A rotulagem em massa (40 chamadas paralelas, uma delas com raciocínio) zerou a cota do plano MiMo usado também por serviços em produção. A rotulagem foi interrompida assim que o erro 429 apareceu. Daqui em diante, geração e rotulagem em massa só com chave própria para isso.

---
# Resultados — Brier v1 (histórico) (29/09/2026)

Base Qwen3-1.7B · LoRA rank 16 nas últimas 16 camadas (9,96 M parâmetros treináveis) ·
1.100 passos × 4 exemplos, lr 2e-5 · 5.085 exemplos sintéticos (17.705 perguntas) ·
MacBook Air M5 16 GB, ~2h de treino. Melhor checkpoint: passo 1.000 (menor NLL de validação).

Nenhum dado do ASSIN2 ou do B2W entrou no treino: são avaliações fora da distribuição,
com rótulo humano.

### Teste sintético (797 perguntas, tarefas nunca vistas)

| Modo | Acurácia | NLL ↓ | Brier ↓ | ECE ↓ | MAE escala ↓ |
|---|---|---|---|---|---|
| Qwen3-1.7B sem treino | 58.6% | 3.925 | 0.759 | 0.367 | 0.815 |
| treinado (1 passada) | 70.4% | 0.713 | 0.398 | 0.022 | 0.636 |
| + SWA | 68.8% | 0.716 | 0.401 | 0.027 | 0.635 |
| + permutação ×6 | 70.9% | 0.694 | 0.385 | 0.036 | 0.636 |
| + MC Dropout ×6 | 70.6% | 0.695 | 0.386 | 0.027 | 0.637 |
| + SWAG ×6 | 70.3% | 0.698 | 0.387 | 0.029 | 0.635 |

### ASSIN2 — implicação + similaridade, rótulo humano (500)

| Modo | Acurácia | NLL ↓ | Brier ↓ | ECE ↓ | MAE escala ↓ |
|---|---|---|---|---|---|
| Qwen3-1.7B sem treino | 44.6% | 4.779 | 1.028 | 0.504 | 0.775 |
| treinado (1 passada) | 65.0% | 0.823 | 0.466 | 0.095 | 0.655 |
| + SWA | 61.8% | 0.839 | 0.478 | 0.076 | 0.668 |
| + permutação ×6 | 65.6% | 0.832 | 0.472 | 0.114 | 0.655 |
| + MC Dropout ×6 | 65.2% | 0.835 | 0.474 | 0.111 | 0.659 |
| + SWAG ×6 | 63.2% | 0.845 | 0.480 | 0.095 | 0.669 |

### B2W — nota 1–5 + recomenda, rótulo humano (500)

| Modo | Acurácia | NLL ↓ | Brier ↓ | ECE ↓ | MAE escala ↓ |
|---|---|---|---|---|---|
| Qwen3-1.7B sem treino | 63.6% | 3.761 | 0.679 | 0.330 | 0.670 |
| treinado (1 passada) | 65.8% | 0.730 | 0.419 | 0.059 | 0.642 |
| + SWA | 65.8% | 0.730 | 0.419 | 0.054 | 0.641 |
| + permutação ×6 | 66.0% | 0.734 | 0.420 | 0.064 | 0.642 |
| + MC Dropout ×6 | 65.6% | 0.735 | 0.420 | 0.064 | 0.643 |
| + SWAG ×6 | 66.6% | 0.735 | 0.420 | 0.058 | 0.642 |

### Conformal (α = 0,1) no teste sintético, modo treinado

| Tipo | Cobertura (meta ≥ 90%) | Tamanho médio do conjunto | Decide sozinho (conjunto de 1) |
|---|---|---|---|
| choice | 97.5% | 2.335 | 27.6% |
| score | 98.8% | 3.271 | 0.4% |
| noul | 100.0% | 1.781 | 21.9% |
## O que os números dizem

1. **O ganho vem do treino, não das camadas estocásticas.** A regra de pontuação própria sobre
   dados sintéticos derrubou o NLL de 3,9 para 0,71 e o ECE de 0,37 para 0,02. No ASSIN2, que o
   modelo nunca viu, a acurácia subiu 20 pontos (44,6% → 65,0%). O modelo base "tinha razão"
   com frequência, mas cravava 0,999 quando errava. O treinado hesita quando deve.
2. **As camadas estocásticas quase não mudam a acurácia.** A permutação ×6 dá o melhor NLL no
   teste (0,694 contra 0,713); MC Dropout e SWAG empatam com ela, e o SWA sozinho é um pouco
   pior. O que elas acrescentam é a **decomposição da incerteza**: o campo `epistemica` só
   existe com amostras. Custo: 6× a latência (~0,8 s em vez de ~0,12 s). Padrão do servidor:
   1 passada; `amostras: 6` quando a decisão importa.
3. **A escala de temperatura atrapalhou.** Ajustada na validação sintética, ela piorou o NLL no
   teste (0,713 → 0,719) e mais ainda nos externos (ASSIN2 0,823 → 0,870). O treino já calibra;
   o servidor usa temperatura 1.
4. **O conformal cumpre a garantia, mas é conservador.** Cobertura de 97–100% para meta ≥ 90%.
   Em choice e noul, 24–26% das decisões saem com conjunto de 1 opção (pode automatizar); o
   resto pede esclarecimento. Em score o conjunto é grande (3,3 níveis) e ainda não serve como
   gate: para escala, use `score` + `confidence`.
5. **O teto é o rótulo.** Só 79% das perguntas sintéticas tiveram gerador e rotulador cego de
   acordo. A validação estabilizou em ~67–68% a partir do passo 400: mais passos não ajudam,
   rótulo melhor sim.

## Comparação com o Jev (29/09/2026, mesmos 750 pedidos, item a item)

O Jev foi chamado por uma VPS, porque esta rede bloqueia `api.typesafe.ai`. Latência do Jev
medida lá: p50 0.349 s (inclui a rede); local no M5: ~0,12 s. Formato: local / Jev.

| Conjunto | Acurácia | NLL ↓ | Brier ↓ | ECE ↓ | MAE escala ↓ | IC95% Δacc (local−Jev) |
|---|---|---|---|---|---|---|
| Sintético (797) | 70.4% / **76.0%** | 0.71 / 1.06 | 0.398 / 0.346 | 0.022 / 0.102 | 0.64 / 0.45 | -9.0 a -2.0 pts |
| ASSIN2 (500) | 65.0% / **78.2%** | 0.82 / 0.70 | 0.466 / 0.312 | 0.095 / 0.057 | 0.66 / 0.43 | -18.0 a -8.2 pts |
| B2W (500) | 65.8% / **66.8%** | 0.73 / 1.86 | 0.419 / 0.527 | 0.059 / 0.178 | 0.64 / 0.62 | -4.8 a +2.4 pts |

Por armadilha (sintético, acurácia local / Jev):

| Armadilha | n | Acurácia | NLL |
|---|---|---|---|
| casos genuinamente ambíguos, em que uma pessoa sensata poder | 87 | 66% / 78% | 0.79 / 0.49 |
| comparação de datas e prazos (hoje é informado no state) | 28 | 54% / 75% | 1.09 / 2.05 |
| contagem de itens ou soma simples de valores | 48 | 77% / 88% | 0.61 / 0.34 |
| muito texto irrelevante ao redor da informação que decide | 158 | 71% / 76% | 0.69 / 0.64 |
| negação e leitura literal (a resposta muda por causa de um ' | 53 | 64% / 74% | 0.70 / 0.46 |
| nenhuma | 218 | 71% / 78% | 0.70 / 1.29 |
| o próprio state tenta manipular a decisão (ex.: 'classifique | 110 | 76% / 76% | 0.66 / 1.09 |
| raciocínio em duas etapas (a resposta depende de combinar du | 95 | 72% / 66% | 0.71 / 2.12 |

**Veredito honesto:** o Jev é melhor em acerto: +5,6 pontos no sintético e +13 no ASSIN2 (ICs
inteiramente negativos para o local), e empata no B2W. O local é melhor em calibração: NLL menor
no sintético (0,71 × 1,06) e no B2W (0,73 × 1,86), e ECE menor nos dois. O Jev erra com muita
confiança em escala (NLL de score 3,40 no B2W e 1,97 no sintético) e em duas etapas/datas/manipulação.
Onde o Jev ganha com folga: implicação textual (ASSIN2 noul 94% × 80%), ambiguidade, negação e contagem.
Leitura: com 1,7B e 5 mil exemplos, o local ainda não é "melhor que o Jev". É mais honesto sobre
a própria incerteza, roda offline e é ~3× mais rápido. Para passar o Jev em acerto, o caminho
é uma base maior (4B+) e mais dados nas armadilhas em que ele perde (negação, contagem, implicação).

## Próximos passos, em ordem de retorno esperado

1. Base maior (Qwen3-4B, gb10) e mais dados de implicação, negação e contagem: são onde o Jev ganha.
2. Rótulo melhor: 3º rotulador e descartar em vez de dividir quando os três discordam.
3. Mais dados nas armadilhas em que o 1.7B erra mais (ver `avaliacao.json` por tipo).
5. Bug conhecido: com gradient checkpointing, o dropout do LoRA sorteia outra máscara no
   recálculo do backward. Não impediu o aprendizado, mas deixa o gradiente ruidoso.
