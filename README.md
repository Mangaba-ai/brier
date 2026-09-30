# Brier

**Modelo aberto de decisão tipada e calibrada.** Roda localmente e não depende de nenhum serviço externo.

O Brier recebe um conteúdo (`state`) e uma ou mais perguntas com respostas pré-definidas. Para cada
pergunta, devolve a opção escolhida e a **probabilidade** de cada opção. Ele não gera texto: responde
sempre dentro das opções que você definiu, e diz o quanto tem certeza.

O nome vem do [escore de Brier](https://en.wikipedia.org/wiki/Brier_score) (Glenn Brier, 1950), a regra
de pontuação que mede se uma probabilidade bate com a realidade. É isso que o treino otimiza: acertar a
**probabilidade**, não só a resposta.

- **Código e pesos abertos:** Apache-2.0. Base Qwen3-4B com adaptadores LoRA próprios.
- **Local:** roda em Apple Silicon com MLX, offline, sem custo por chamada. ~0,26 s por pedido.
- **Tipado:** a resposta sempre cabe no tipo pedido, por construção.
- **Calibrado:** quando diz 80%, acerta perto de 80% das vezes. Isso permite decidir o que automatizar
  e o que mandar para uma pessoa.

## Para que serve

Decisões curtas e repetidas dentro de um software, em que errar com convicção custa caro:

- triar mensagens e chamados (qual setor atende? é urgente?);
- classificar documentos, avaliações e comentários (é ofensivo? que nota o cliente deu?);
- decidir se uma ação pode ser automatizada ou precisa de revisão humana.

A diferença para um classificador comum é a confiança: o Brier informa quando **não sabe**, e o campo
`conjunto` dá uma garantia estatística sobre isso (veja "Incerteza e garantia").

## Como funciona uma chamada

```bash
curl -s localhost:8790/v1/decide -d '{
  "state": "minha internet caiu e já é a terceira vez esta semana! quero cancelar",
  "questions": {
    "setor":   {"type": "choice", "instructions": "Qual time atende?",
                "criteria": {"suporte": "Queda, lentidão", "financeiro": "Boleto, cobrança", "retencao": "Cancelamento"}},
    "raiva":   {"type": "score", "instructions": "Quão irritado está o cliente?",
                "criteria": ["Calmo", "Incomodado", "Muito irritado"]},
    "urgente": {"type": "noul", "instructions": "Precisa agir já?",
                "criteria": {"true": "Serviço parado", "false": "Pode esperar"}}
  }
}'
```

Três tipos de pergunta:

| Tipo | Para quê | `criteria` | Resposta |
|---|---|---|---|
| `choice` | escolher 1 entre N opções (até 26) | `{"chave": "descrição"}` | `choice`, `probabilities`, `confidence` |
| `score` | posição numa escala ordenada (2 a 10 níveis) | `["nível 0", "nível 1", …]` | `score` (média ponderada, pode cair entre níveis), `probabilities`, `confidence` |
| `noul` | sim ou não | `{"true": "…", "false": "…"}` | `noul` = probabilidade de "sim" |

Toda resposta também traz `incerteza` e, com calibração carregada, `conjunto`. `confidence` vai de 0
(distribuição uniforme) a 1 (certeza total): `1 − H(p)/log K`.

No exemplo acima, a resposta para `setor` foi `retencao` com 54% e `suporte` com 45%, e o `conjunto`
trouxe as duas opções. É o Brier dizendo que o caso é genuinamente ambíguo (internet caiu **e** o
cliente quer cancelar) e que vale uma pergunta de esclarecimento ou uma revisão humana.

## Como funciona por dentro

**Uma passada para todas as perguntas.** O `state` e as perguntas entram numa única sequência. Uma
máscara de atenção em blocos faz cada pergunta enxergar o `state` e a si mesma, nunca as outras. Assim,
as respostas são independentes, mas custam uma só passada pelo modelo.

**Leitura direta das probabilidades.** No fim de cada pergunta, o Brier lê as probabilidades só dos
tokens que rotulam as opções (`A`…`Z` para choice, `0`…`9` para score) e normaliza entre elas. Nada é
gerado, então não existe resposta fora do tipo. A técnica segue a linha de SALSA (arXiv 2510.22691).

**Treino por regra de pontuação própria.** A perda é a entropia cruzada contra um alvo suave: firme
quando os rotuladores concordam, dividido quando discordam. Nas escalas, um termo extra pune errar por
muitos níveis. Como o modelo devolve a distribuição inteira, otimizar essa regra é otimizar calibração
diretamente, a mesma ideia de RLCR (Damani et al., MIT, 2025) e CARL (arXiv 2601.13284), sem amostragem
por reforço.

**Dados.**
- 6.744 exemplos sintéticos em português, em 46 domínios e 9 formatos. A maioria mira casos difíceis:
  negação, datas, contagem, raciocínio em duas etapas, texto irrelevante, tentativa de manipulação e
  ambiguidade real.
- Cada exemplo é respondido por vários rotuladores independentes, e vale o voto da maioria. Isso
  corrigiu 13,6% dos rótulos originais.
- 15.128 exemplos públicos com rótulo humano (ASSIN v1, FaQuAD-NLI, HateBR), sorteados em 30% do treino.

### Incerteza e garantia

| Camada | O que faz | Origem |
|---|---|---|
| **Predição conformal (APS)** | o campo `conjunto` contém a resposta certa com ≥ 90% de garantia; conjunto de 1 opção = pode decidir sozinho | Romano, Sesia & Candès (2020); Angelopoulos & Bates (2021) |
| **MC Dropout** nos adaptadores | com `"amostras": N`, separa incerteza **aleatória** (o caso é ambíguo) de **epistêmica** (o modelo não sabe) | Yarin Gal, tese *Uncertainty in Deep Learning* (Cambridge, 2016) |
| **SWA / SWAG** | média e variância dos pesos no fim do treino, para amostrar pesos | Pavel Izmailov, tese *Deconstructing Models and Methods in Deep Learning* (NYU, 2023) |
| **Permutação das opções** | cada amostra vê as opções em outra ordem, o que remove viés de posição | Zheng et al., ICLR 2024 |

No v2, o `conjunto` tem uma única opção em ~40% das perguntas de escolha e de sim/não, com cobertura
medida de 98–100%. Essas decisões podem ser automatizadas com a garantia; o restante vai para
esclarecimento.

## Uso

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/servidor.py --execucao execucoes/v2 --porta 8790
```

O Qwen3-4B de 4 bits baixa sozinho do Hugging Face na primeira execução. `"amostras": 6` no corpo liga
as camadas estocásticas (≈ 6× a latência).

**Reproduzir o treino do zero** (credenciais de um endpoint OpenAI-compatível para gerar dados, em
variáveis de ambiente ou num `.env`; ver `src/chaves.py`):

```bash
.venv/bin/python src/gerar.py --tarefas 800 --paralelo 12              # sintéticos
.venv/bin/python src/gerar.py --foco --tarefas 380 --saida dados/sintetico_foco.jsonl
.venv/bin/python src/rotular.py dados/sintetico.jsonl dados/sintetico_v2.jsonl   # maioria de votos
.venv/bin/python src/publicos.py                                        # ASSIN, FaQuAD-NLI, HateBR
.venv/bin/python src/treinar.py --modelo mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --dados dados/sintetico_v2.jsonl dados/sintetico_foco_v2.jsonl \
  --publicos dados/publicos_treino.jsonl --passos 1500 --camadas 20 --lr 2e-5 --avaliar-cada 150 \
  --pular-base --saida execucoes/v2
```

O v2 publicado é o checkpoint de menor perda de validação (passo 450). O treino foi interrompido no
passo 680, quando a validação começou a piorar.

> A geração e a rotulagem fazem milhares de chamadas. Use uma chave dedicada, nunca a de um serviço
> em produção.

## Desempenho

Avaliação em conjuntos com **rótulo humano que não entram no treino**, 250 pedidos por conjunto:

| Conjunto | Tarefa | Acurácia | Escore de Brier ↓ | ECE ↓ |
|---|---|---|---|---|
| ASSIN2 | implicação textual + similaridade | 72,8% | 0,375 | 0,062 |
| B2W | nota de 1 a 5 + recomendação | 65,4% | 0,459 | 0,113 |
| tweetSentBR | sentimento | 72,0% | 0,421 | 0,108 |
| Sintético (teste) | 46 domínios, casos difíceis | 77,5% | 0,304 | 0,041 |

Evolução: o v1 (Qwen3-1.7B) tinha 65,0% no ASSIN2 e 72,5% no sintético. Números completos, ablação das
camadas e histórico em [RESULTADOS.md](RESULTADOS.md).

## Comparação com o Jev (referência externa)

O Brier é independente: pesos próprios, dados próprios e nenhuma dependência de serviço de terceiros
para funcionar. O **Jev**, da TypeSafe AI, é um modelo comercial fechado de decisão tipada. Ele entra
**só como régua de desempenho**, porque é a referência mais próxima do que o Brier faz.

- Nenhuma resposta do Jev foi usada para treinar o Brier.
- As respostas do Jev coletadas na comparação não são redistribuídas neste repositório.

**Como comparamos:** os mesmos pedidos foram enviados aos dois modelos, item a item, com intervalo de
confiança de 95% por bootstrap pareado. O critério de vitória foi fixado **antes** da medição: o Brier só
é "melhor que o Jev" se vencer em acurácia **e** escore de Brier, com IC favorável, em pelo menos 2 dos
3 conjuntos humanos.

| Conjunto | Brier v2 | Jev | Leitura |
|---|---|---|---|
| ASSIN2 | 72,8% · Brier 0,375 | 78,2% · Brier 0,312 | Jev melhor |
| B2W | 65,4% · Brier 0,459 | 66,8% · Brier 0,527 | empate no acerto; **Brier mais bem calibrado** |
| tweetSentBR | 72,0% · Brier 0,421 | 67,6% · Brier 0,459 | Brier à frente, sem significância estatística |
| Sintético | 77,5% · Brier 0,304 | 79,5% · Brier 0,287 | empate estatístico |

**Veredito pelo critério: ainda não é melhor que o Jev.** O Brier não venceu em nenhum dos três
conjuntos humanos com significância nas duas métricas. O v2 fechou boa parte da distância em relação ao
v1: no ASSIN2, ela caiu de 13 para 5,4 pontos.

Vantagens que não dependem desse critério: roda offline, sem custo por chamada, com código e pesos
abertos, e erra com menos convicção quando erra.

**Migração:** quem já usa o formato do Jev pode apontar para `POST /v1/systemone` neste servidor, que
aceita o mesmo corpo. A rota nativa do Brier é `POST /v1/decide`.

Para refazer a comparação com a sua própria chave da TypeSafe, depois de `src/externos.py` e
`src/publicos.py`:

```bash
TYPESAFE_API_KEY=… .venv/bin/python src/comparar_jev.py   # coleta as respostas do Jev (cache local)
.venv/bin/python src/comparar_v2.py                       # compara item a item e aplica o critério
```

## Conteúdo do repositório

- `src/`: geração, rotulagem, treino, avaliação, servidor e comparação
- `execucoes/v2/`: adaptadores do Brier v2 (padrão) e calibração conformal
- `execucoes/v1/`: adaptadores do v1 (Qwen3-1.7B), com SWA e SWAG
- `execucoes/comparacao_v2.json`: métricas da comparação pareada
- `dados/sintetico*.jsonl`: os dados sintéticos de treino e teste, com votos e alvos

Não redistribuímos dados de terceiros: ASSIN, ASSIN2, B2W, FaQuAD-NLI, HateBR e tweetSentBR são baixados
e convertidos por `src/externos.py` e `src/publicos.py`, sob as licenças de cada conjunto.

## Licença

Apache-2.0, a mesma do Qwen3. Jev e TypeSafe são marcas da TypeSafe AI, citadas apenas para fins de
comparação de desempenho. O Brier não tem vínculo com a TypeSafe AI.
