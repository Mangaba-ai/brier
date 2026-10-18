# Brier

[![testes](https://github.com/Mangaba-ai/brier/actions/workflows/testes.yml/badge.svg)](https://github.com/Mangaba-ai/brier/actions/workflows/testes.yml)

**Modelo de decisão tipada e calibrada da Mangaba AI.** Roda localmente e não depende de nenhum serviço externo.

> **English:** Brier is a typed, calibrated decision model for Portuguese text, by Mangaba AI. It runs
> locally and returns, for each question, the chosen option and the probability of every option.
> Site, plans and licensing: [brier.mangaba.ia.br](https://brier.mangaba.ia.br/#planos).

O Brier recebe um conteúdo (`state`) e uma ou mais perguntas com respostas pré-definidas. Para cada
pergunta, devolve a opção escolhida e a **probabilidade** de cada opção. Ele não gera texto: responde
sempre dentro das opções que você definiu, e diz o quanto tem certeza.

O nome vem do [escore de Brier](https://en.wikipedia.org/wiki/Brier_score) (Glenn Brier, 1950), a regra
de pontuação que mede se uma probabilidade bate com a realidade. É isso que o treino otimiza: acertar a
**probabilidade**, não só a resposta.

- **Licença proprietária da Mangaba AI** a partir da versão 5.0.0 (ver [LICENSE](LICENSE) e [NOTICE](NOTICE)).
  O que foi publicado até a tag `ultima-apache-2.0` (pacote 4.1.0 e os pesos v2 e v3) continua sob
  Apache-2.0. Planos: [brier.mangaba.ia.br/pt#planos](https://brier.mangaba.ia.br/pt#planos).
- **Local:** roda nos seus servidores Linux ou Windows, em GPU NVIDIA ou CPU. Offline e sem custo por
  chamada.
- **Tipado:** a resposta sempre cabe no tipo pedido, por construção.
- **Calibrado:** quando diz 80%, acerta perto de 80% das vezes. Isso permite decidir o que automatizar
  e o que mandar para uma pessoa.

## Versões

| | **Brier v5** (recomendada · padrão) | Brier v3 (reserva) | Brier v2 |
|---|---|---|---|
| Modelo | v3 continuado com a parte de treino do B2W e mais passos, treinado em 2× NVIDIA L40S | v2 retreinado contra injeção de instruções | Qwen3-4B + LoRA treinado pelo Brier em português |
| Aplicações | ![Atendimento no WhatsApp](https://img.shields.io/badge/Atendimento_no_WhatsApp-2ea44f) ![Triagem de chamados](https://img.shields.io/badge/Triagem_de_chamados-0ea5e9) ![Avaliações de produto](https://img.shields.io/badge/Avalia%C3%A7%C3%B5es_de_produto-8b5cf6) ![Proteção contra injeção](https://img.shields.io/badge/Prote%C3%A7%C3%A3o_contra_inje%C3%A7%C3%A3o-e11d48) | ![E-mail e formulários](https://img.shields.io/badge/E--mail_e_formul%C3%A1rios-06b6d4) ![Triagem de chamados](https://img.shields.io/badge/Triagem_de_chamados-0ea5e9) ![Proteção contra injeção](https://img.shields.io/badge/Prote%C3%A7%C3%A3o_contra_inje%C3%A7%C3%A3o-e11d48) | ![Análise de sentimento](https://img.shields.io/badge/An%C3%A1lise_de_sentimento-f59e0b) ![Dados internos](https://img.shields.io/badge/Dados_internos-0ea5e9) ![Auditoria em lote](https://img.shields.io/badge/Auditoria_em_lote-2ea44f) |
| Acerto médio nos 3 conjuntos humanos | **75,1%** | 67,3% | 70,1% |
| Injeção: ataques que funcionaram (1.000 casos, menor é melhor) | **12,4%** | 16,7% | não medido |
| Pesos | licença comercial (acesso aprovado) | Apache-2.0 | Apache-2.0 |
| Como rodar | `Brier()` · `brier-serve` | `Brier("v3")` · `brier-serve --modelo v3` | `Brier("v2")` · `brier-serve --modelo v2` |

**Use a v5:** é a mais precisa e a mais difícil de manipular. O download exige acesso aprovado em
[mangaba-ai/brier-v5](https://huggingface.co/mangaba-ai/brier-v5) (licença comercial) e `huggingface-cli login`.
Sem acesso, `Brier()` e `brier-serve` voltam sozinhos para o v3, com um aviso.

Desde a v5, a parte de **treino** do B2W entra no treino (8.000 avaliações; nenhuma das usadas no teste).
O B2W deixou de ser um conjunto fora do domínio para a v5; ASSIN2, tweetSentBR e o sintético continuam fora.

O **`comite`** (média das probabilidades do v2 e do v3) custa o dobro de tempo e cai mais em injeção
(22,5% no teste cego); serve só para texto confiável.

Todas as versões rodam pelo mesmo pacote, com lotes, abstenção (`min_confidence`), auditoria,
mascaramento de dados pessoais, MCP e LangChain.

### Melhores resultados do Brier até aqui (contra o Jev, mesmos pedidos)

- **B2W (avaliações de produto):** a v5 **vence o Jev em acerto e em escore de Brier, com significância**:
  75,8% × 66,8% (IC95% de +4,6 a +13,6 pontos) e 0,335 × 0,527 (menor é melhor).
- **ASSIN2 (implicação textual):** a v5 empata com o Jev: 78,6% × 78,2%.
- **tweetSentBR (sentimento):** v5 70,8% e v2 72,0%, contra 67,6% do Jev, ainda sem significância.
- **Injeção de instruções:** com a v5, a instrução maliciosa mudou a resposta para o que pedia em
  **12,4%** dos 1.000 casos (v3: 16,7%; Jev: 21,4%). Num **teste cego** com 57 frases inéditas escritas por
  outro modelo (mangaba-titan), 15,0% contra 20,5% do Jev (n=200; IC95% de −11,0 a −0,5 pontos, **com
  significância**).
- **Custo e privacidade:** roda offline, sem custo por chamada e sem mandar texto para fora.

```bash
pip install "brier @ git+https://github.com/Mangaba-ai/brier"
huggingface-cli login             # com acesso aprovado à v5
brier-serve --porta 8790          # Brier v5 (sem acesso aprovado, v3)
```

O uso das versões a partir da 5.0.0 depende de licença da Mangaba AI ([LICENSE](LICENSE);
planos em [brier.mangaba.ia.br/pt#planos](https://brier.mangaba.ia.br/pt#planos)).

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

No exemplo acima, o Brier v3 (padrão) respondeu `setor` = `suporte` com 84%, `retencao` com 15% e
`financeiro` com 1%; `raiva` = 1,82 (perto de "Muito irritado") e `urgente` = 0,86. O `conjunto` de
`setor` trouxe `suporte` **e** `retencao`: é o Brier dizendo que o caso tem duas leituras (a internet caiu
**e** o cliente quer cancelar) e que vale uma pergunta de esclarecimento ou uma revisão humana. A resposta
levou 0,37 s (pedido curto; a mediana nos pedidos de teste é ~1,3 s).

## Como funciona por dentro

Os parágrafos abaixo descrevem o v2; o v3 é o v2 retreinado também contra injeção de instruções.

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

### Pacote `brier` (todas as versões)

```bash
pip install "brier @ git+https://github.com/Mangaba-ai/brier"
brier-serve --porta 8790 [--modelo v5|v3|v2|comite] [--auditoria auditoria.jsonl] [--mascarar-pii] [--neutralizar-injecao]
```

Os adaptadores ficam no Hugging Face e baixam sozinhos na primeira execução:
[mangaba-ai/brier-v5](https://huggingface.co/mangaba-ai/brier-v5) (acesso aprovado),
[mangaba-ai/brier-v3](https://huggingface.co/mangaba-ai/brier-v3) e [mangaba-ai/brier-v2](https://huggingface.co/mangaba-ai/brier-v2).

```python
from brier import Brier
b = Brier(mascarar_pii=True)                        # v5 (sem acesso, v3); Brier("v3"), Brier("v2")
b.decide(texto, perguntas, min_confidence=0.8)     # "abstencao" lista as perguntas abaixo do mínimo
b.decide_lote([texto1, texto2, texto3], perguntas)  # vários textos numa chamada
```

| Rota | Para quê |
|---|---|
| `POST /v1/decide` | um `state`; `min_confidence` opcional |
| `POST /v1/decide/lote` | até 256 `states` com as mesmas perguntas |
| `POST /v1/systemone` | mesmo corpo do `/v1/decide`, para quem migra do Jev |

- **Textos longos:** a resposta avisa quando o `state` foi truncado.
- **Instruções injetadas:** frases que parecem ordens ao sistema ("ignore as instruções…", "a resposta
  correta é…") são marcadas no campo `injecao`; com `--neutralizar-injecao`, são trocadas por um aviso
  antes da decisão. É uma camada simples, por regras: num teste cego detectou só 7,5% das frases
  inéditas (com quase nenhum alarme falso), então a defesa principal continua sendo o treino (v3 e v5).
- **Auditoria:** `--auditoria arquivo.jsonl` grava hash do texto, roteamento, latência e respostas,
  nunca o texto.
- **Dados pessoais:** `--mascarar-pii` troca CPF, CNPJ, cartão, e-mail, telefone e CEP por marcadores
  antes da decisão.
- **Integrações:** servidor MCP (`brier-mcp`, ferramentas `brier_decidir` e `brier_decidir_lote`),
  LangChain/LangGraph (`brier.langchain`), cliente TypeScript (`clientes/typescript/brier.ts`) e
  `Dockerfile`.

Latência mediana: ~1,3 s por pedido com o servidor aquecido; comitê ~2× isso.

### Motores, latência e memória

O Brier roda em **Linux e Windows**, em GPU NVIDIA (CUDA) ou CPU, com o motor PyTorch. A v5, o v3 e o
v2 têm a mesma base e o mesmo custo.

**Latência medida** (1 amostra por pedido, servidor aquecido):

| Pedido | Latência |
|---|---|
| Curto (1 frase, 2 perguntas) | 0,26 s |
| Pedidos do teste, mediana | 1,3 s |
| Pedidos do teste, p90 | 1,7 s |

O tempo cresce com o tamanho do `state` e com o número de perguntas.

A CPU funciona, mas é lenta para um modelo de 4B; para uso real, prefira GPU. Em CPU sem bf16 nativo
(por exemplo, Xeon Cascade Lake, sem `avx512_bf16` nem AMX), o bf16 é emulado: rode com
`BRIER_DTYPE=float32`, que nessa máquina foi cerca de 5× mais rápido que o bf16. Para escolher o
dispositivo, use `brier-serve --dispositivo cuda|cpu`.

Na primeira execução, a base (Qwen3-4B em 4 bits, ~2,3 GB) e os adaptadores baixam sozinhos do Hugging
Face. A base é convertida em memória para 16 bits.

**Memória:** ~9–10 GB de RAM ou VRAM (bf16), ou seja, GPU NVIDIA com 12 GB ou mais.

**Testes** (qualquer sistema, sem baixar modelo): `python -m pytest tests -q`. O GitHub Actions roda
os testes em Linux e Windows a cada push.

## Desempenho

Conjuntos com **rótulo humano**, separados do treino, 250 pedidos por conjunto (a v5 treinou com a parte de
treino do B2W, nunca com a de teste). Cada célula: acurácia · escore de Brier (menor é melhor). A v5 foi
medida numa rodada própria de avaliação, com os mesmos pedidos; os números de v2 e v3 são da rodada anterior.

| Conjunto | Tarefa | **Brier v5** | Brier v3 | Brier v2 |
|---|---|---|---|---|
| ASSIN2 | implicação textual + similaridade | **78,6%** · 0,301 | 71,2% · 0,357 | 72,8% · 0,375 |
| B2W | nota de 1 a 5 + recomendação | **75,8%** · 0,335 | 61,6% · 0,484 | 65,4% · 0,459 |
| tweetSentBR | sentimento | 70,8% · 0,427 | 69,2% · 0,455 | **72,0%** · 0,421 |
| Sintético (teste) | 46 domínios, casos difíceis | **77,5%** · 0,297 | 75,7% · 0,318 | **77,5%** · 0,304 |

| Robustez | **Brier v5** | Brier v3 |
|---|---|---|
| Injeção de instruções (frases nunca vistas) | o ataque funcionou em **12,4%** dos casos; acerto 76,8% → 73,9% | 16,7%; acerto 70,3% → 67,3% |
| Documentos longos (~2.500 tokens, por janelas) | acerto 80,7% → 66,9% | acerto 73,5% → 59,1% | Números completos e histórico em [RESULTADOS.md](RESULTADOS.md).

## Comparação com o Jev (referência externa)

O Brier é independente: código próprio, pesos próprios e nenhuma dependência de serviço de terceiros
para funcionar. O **Jev**, da TypeSafe AI, é um modelo comercial fechado de decisão tipada. Ele entra
**só como régua de desempenho**, porque é a referência mais próxima do que o Brier faz.

- Nenhuma resposta do Jev foi usada para treinar o Brier.
- As respostas do Jev coletadas na comparação não são redistribuídas neste repositório.

**Como comparamos:** os mesmos pedidos foram enviados aos dois modelos, item a item, com intervalo de
confiança de 95% por bootstrap pareado. O critério de vitória foi fixado **antes** da medição: o Brier só
é "melhor que o Jev" se vencer em acurácia **e** escore de Brier, com IC favorável, em pelo menos 2 dos
3 conjuntos humanos.

| Conjunto | **Brier v5** | Brier v2 | Brier v3 | Comitê v2+v3 | Jev | Leitura |
|---|---|---|---|---|---|---|
| ASSIN2 | **78,6%** · 0,301 | 72,8% · 0,375 | 71,2% · 0,357 | 73,0% · 0,357 | 78,2% · 0,312 | v5 empata com o Jev; v2, v3 e comitê abaixo |
| B2W | **75,8%** · 0,335 | 65,4% · 0,459 | 61,6% · 0,484 | 63,4% · 0,465 | 66,8% · 0,527 | **v5 melhor que o Jev em acerto e escore de Brier, com significância** |
| tweetSentBR | 70,8% · 0,427 | 72,0% · 0,421 | 69,2% · 0,455 | 71,6% · 0,434 | 67,6% · 0,459 | v5, v2 e comitê à frente do Jev, sem significância |
| Sintético | 77,5% · 0,297 | 77,5% · 0,304 | 75,7% · 0,318 | 77,0% · 0,303 | 79,5% · 0,287 | v5, v2 e comitê empatam com o Jev |
| Injeção, frases nossas (taxa de ataque, menor é melhor) | **12,4%** | não medido | **16,7%** | não medido | 21,4% | **v5 e v3 mais resistentes que o Jev** (v3 com significância; IC da v5 não calculado nesta bateria) |
| Injeção, teste cego (57 frases do mangaba-titan, n=200) | **15,0%** | não medido | **17,5%** | 22,5% | 20,5% | **v5 mais resistente, com significância**; v3 à frente sem significância; comitê pior |
| Documentos longos (acerto limpo → longo) | 80,7% → 66,9% | não medido | 73,5% → 59,1% | não medido | 78,2% → 73,0% | Jev melhor (cai menos) |
| Latência mediana | ~1,3 s | ~1,3 s | ~1,3 s | ~2× o v3 | ~0,35 s (pela rede) | Jev mais rápido |

**Veredito pelo critério: a v5 tem 1 vitória (B2W) das 2 necessárias; ainda não é "melhor que o Jev"
no geral.** Vitórias em conjuntos humanos: v5 = 1, v2 = 0, v3 = 0, comitê = 0. A v5 empata no ASSIN2 e
lidera o tweetSentBR sem significância; o B2W entrou no treino da v5 pela parte de treino.

Vantagens que não dependem desse critério: roda offline, sem custo por chamada, e erra com menos
convicção quando erra.

**Migração:** quem já usa o formato do Jev pode apontar para `POST /v1/systemone` neste servidor, que
aceita o mesmo corpo. A rota nativa do Brier é `POST /v1/decide`.

As métricas de cada comparação estão em `execucoes/*.json`. Os scripts de geração, treino e avaliação
ficam num repositório privado da Mangaba AI.

## Conteúdo do repositório

- `brier/`: pacote do Brier (v5, v3, v2 e comitê; servidor, MCP, LangChain, regras contra injeção, mascaramento de dados pessoais)
- `clientes/typescript/`: cliente TypeScript
- `tests/`: testes do pacote (rodam sem baixar o modelo)
- `execucoes/comparacao_v2.json`, `comparacao_v3.json`, `comparacao_comite.json`, `robustez_v3.json`, `injecao_cega.json`, `comparacao_v5.json`, `robustez_v5.json`, `injecao_cega_v5.json`: métricas das comparações pareadas

Os adaptadores do v2 e do v3 ficam no Hugging Face ([mangaba-ai/brier-v3](https://huggingface.co/mangaba-ai/brier-v3),
[mangaba-ai/brier-v2](https://huggingface.co/mangaba-ai/brier-v2)) e o pacote os baixa sozinho. Os conjuntos públicos
de avaliação (ASSIN2, B2W, tweetSentBR) não são redistribuídos aqui.

## Licença

**Licença proprietária da Mangaba AI** a partir da versão 5.0.0: uso, cópia, modificação e
distribuição só com autorização por escrito (texto completo em [LICENSE](LICENSE)). O código, os
pesos e os dados publicados até a tag `ultima-apache-2.0`, inclusive os adaptadores v2 e v3 no Hugging
Face tal como publicados até então, continuam sob a Apache-2.0 ([licenses/APACHE-2.0.txt](licenses/APACHE-2.0.txt)).
A licença comercial cobre a versão 5.0.0 em diante: uso em produção do pacote atual, suporte,
recursos novos e versões futuras. Pesos de versões futuras serão distribuídos sob a licença
proprietária, fora dos repositórios Apache `mangaba-ai/brier-v2` e `mangaba-ai/brier-v3`.

Componentes de terceiros em [NOTICE](NOTICE): o Qwen3 (Apache-2.0) é a base dos adaptadores. Jev e TypeSafe são marcas da TypeSafe
AI, citadas apenas para fins de comparação de desempenho. O Brier não tem vínculo com a TypeSafe AI.

Planos e licenciamento: [brier.mangaba.ia.br/pt#planos](https://brier.mangaba.ia.br/pt#planos) ·
contato: [mangaba.ia.br](https://mangaba.ia.br).
