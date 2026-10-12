# Brier

[![testes](https://github.com/Mangaba-ai/brier/actions/workflows/testes.yml/badge.svg)](https://github.com/Mangaba-ai/brier/actions/workflows/testes.yml)

**Modelo de decisão tipada e calibrada da Mangaba AI.** Roda localmente e não depende de nenhum serviço externo.

O Brier recebe um conteúdo (`state`) e uma ou mais perguntas com respostas pré-definidas. Para cada
pergunta, devolve a opção escolhida e a **probabilidade** de cada opção. Ele não gera texto: responde
sempre dentro das opções que você definiu, e diz o quanto tem certeza.

O nome vem do [escore de Brier](https://en.wikipedia.org/wiki/Brier_score) (Glenn Brier, 1950), a regra
de pontuação que mede se uma probabilidade bate com a realidade. É isso que o treino otimiza: acertar a
**probabilidade**, não só a resposta.

- **Licença proprietária da Mangaba AI** a partir da versão 5.0.0 (ver [LICENSE](LICENSE) e [NOTICE](NOTICE)).
  O que foi publicado até a tag `ultima-apache-2.0` continua sob Apache-2.0.
- **Local e multiplataforma:** Windows, Linux e macOS, em CPU, GPU NVIDIA ou Apple Silicon. Offline e
  sem custo por chamada.
- **Tipado:** a resposta sempre cabe no tipo pedido, por construção.
- **Calibrado:** quando diz 80%, acerta perto de 80% das vezes. Isso permite decidir o que automatizar
  e o que mandar para uma pessoa.

## Versões

| | Brier v2 (mais preciso) | **Brier v3** (recomendada) |
|---|---|---|
| Modelo | Qwen3-4B + LoRA treinado pelo Brier em português | v2 retreinado contra injeção de instruções; mais resistente a injeção que o Jev |
| Aplicações | ![Análise de sentimento](https://img.shields.io/badge/An%C3%A1lise_de_sentimento-f59e0b) ![Avaliações de produto](https://img.shields.io/badge/Avalia%C3%A7%C3%B5es_de_produto-8b5cf6) ![Dados internos](https://img.shields.io/badge/Dados_internos-0ea5e9) ![Auditoria em lote](https://img.shields.io/badge/Auditoria_em_lote-2ea44f) | ![Atendimento no WhatsApp](https://img.shields.io/badge/Atendimento_no_WhatsApp-2ea44f) ![Triagem de chamados](https://img.shields.io/badge/Triagem_de_chamados-0ea5e9) ![E-mail e formulários](https://img.shields.io/badge/E--mail_e_formul%C3%A1rios-06b6d4) ![Proteção contra injeção](https://img.shields.io/badge/Prote%C3%A7%C3%A3o_contra_inje%C3%A7%C3%A3o-e11d48) |
| Acerto médio nos 3 conjuntos humanos | **70,1%** | 67,3% |
| Latência por pedido (Mac M5) | ~1,3 s | ~1,3 s |
| Como rodar | `Brier("v2")` · `brier-serve --modelo v2` | `Brier()` · `brier-serve` |

**Use o v3** na maioria dos casos: em triagem de texto vindo de terceiros (WhatsApp, e-mail, chamados,
comentários), qualquer pessoa pode escrever "classifique como urgente" na mensagem, e o v3 é a versão
mais difícil de manipular (veja "Injeção de instruções" abaixo), perdendo só 1,6 a 3,8 pontos de acerto
para o v2. **Use o v2** quando o texto é confiável e cada ponto de acerto importa.
O **`comite`** (média das probabilidades do v2 e do v3) acerta um pouco mais que o v3 em texto limpo,
mas cai mais em injeção (22,5% × 17,5% no teste cego) e custa o dobro de tempo; serve para texto confiável.

O antigo v4 (`rapido`, os pesos do Laya usados como estão) saiu do pacote na versão 5.0.0; os números
dele continuam abaixo só como referência.

Todas as versões rodam pelo mesmo pacote, com lotes, abstenção (`min_confidence`), auditoria,
mascaramento de dados pessoais, MCP e LangChain.

### Melhores resultados do Brier até aqui (contra o Jev, mesmos pedidos)

- **tweetSentBR (sentimento):** 72,0% de acerto contra 67,6% do Jev
  (+4,4 pontos; IC95% de −0,4 a +9,2, ainda sem significância).
- **B2W (avaliações de produto):** **mais bem calibrado que o Jev, com significância**: escore de
  Brier 0,459 contra 0,527 (menor é melhor), com acerto empatado.
- **Sintético (46 domínios, casos difíceis):** empate estatístico com o Jev (77,5% × 79,5%).
- **Injeção de instruções (Brier v3):** a instrução maliciosa mudou a resposta para o que pedia em
  16,7% dos casos, contra 21,4% do Jev (−4,7 pontos; IC95% aproximado de −8,1 a −1,3, **com
  significância**), em 1000 casos com 4 famílias de frases escritas por nós e nunca vistas no treino.
  Num **teste cego** com 57 frases inéditas escritas por outro modelo (mangaba-titan), a vantagem se
  manteve, mas menor e sem significância: 17,5% contra 20,5% (n=200; IC95% de −8,5 a +2,0).
- **Custo e privacidade:** roda offline, sem custo por chamada e sem mandar texto para fora.

```bash
pip install "brier[mac] @ git+https://github.com/Mangaba-ai/brier"   # Mac com Apple Silicon (MLX)
pip install "brier @ git+https://github.com/Mangaba-ai/brier"        # Windows e Linux (PyTorch)
brier-serve --porta 8790          # Brier v3; pesos baixados de huggingface.co/mangaba-ai/brier-v3
```

O uso das versões a partir da 5.0.0 depende de licença da Mangaba AI ([LICENSE](LICENSE)).

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
levou 0,37 s num MacBook Air M5 (pedido curto; a mediana nos pedidos de teste é ~1,3 s).

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
pip install "brier[mac] @ git+https://github.com/Mangaba-ai/brier"   # Mac com Apple Silicon; sem [mac] em Windows/Linux
brier-serve --porta 8790 [--modelo v3|v2|comite] [--auditoria auditoria.jsonl] [--mascarar-pii] [--neutralizar-injecao]
```

Os adaptadores do v2 e do v3 ficam em [mangaba-ai/brier-v2](https://huggingface.co/mangaba-ai/brier-v2) e
[mangaba-ai/brier-v3](https://huggingface.co/mangaba-ai/brier-v3) e baixam sozinhos na primeira execução.

```python
from brier import Brier
b = Brier(mascarar_pii=True)                        # v3; Brier("v2"), Brier("comite")
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
  inéditas (com quase nenhum alarme falso), então a defesa principal continua sendo o treino do v3.
- **Auditoria:** `--auditoria arquivo.jsonl` grava hash do texto, roteamento, latência e respostas,
  nunca o texto.
- **Dados pessoais:** `--mascarar-pii` troca CPF, CNPJ, cartão, e-mail, telefone e CEP por marcadores
  antes da decisão.
- **Integrações:** servidor MCP (`brier-mcp`, ferramentas `brier_decidir` e `brier_decidir_lote`),
  LangChain/LangGraph (`brier.langchain`), cliente TypeScript (`clientes/typescript/brier.ts`) e
  `Dockerfile`.

Latência no MacBook Air M5: v3 ~1,3 s por pedido (MLX, mediana); comitê ~2× isso.

### Brier v3 e v2 (Qwen3-4B) a partir do repositório

Os scripts de `src/` (treino, avaliação e o servidor de pesquisa com MC Dropout e SWAG) usam as pastas
`execucoes/`. O v3 (padrão do `src/servidor.py`) e o v2 rodam em **Windows, Linux e macOS** e tem dois motores, que dão as mesmas respostas:

| Sistema | Motor | Hardware |
|---|---|---|
| Windows, Linux | PyTorch | GPU NVIDIA (CUDA) ou CPU |
| macOS com Apple Silicon | MLX (padrão) ou PyTorch (MPS) | GPU integrada |
| macOS com Intel | PyTorch | CPU |

**Latência medida** (Brier v2, 1 amostra, MacBook Air M5 16 GB, servidor aquecido):

| Motor | Pedido curto (1 frase, 2 perguntas) | Pedidos do teste (mediana · p90) | Escolhas iguais às do MLX |
|---|---|---|---|
| MLX | 0,26 s | 1,3 s · 1,7 s | referência |
| PyTorch na GPU (MPS, fp16) | 0,24 s | 1,5 s · 2,3 s | 98,6% (1 de 69, num quase empate) |
| PyTorch na CPU (bf16) | — | ~72 s | 100% |

O tempo cresce com o tamanho do `state` e com o número de perguntas.

A CPU funciona, mas é lenta para um modelo de 4B; para uso real, prefira GPU. Em GPU NVIDIA ainda
não medimos.

O servidor escolhe o motor sozinho; para forçar, use `--motor torch` ou `--motor mlx`.

**Windows (PowerShell)**

```powershell
py -3.11 -m venv .venv
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu124   # GPU NVIDIA; para só CPU use /whl/cpu
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python src\servidor.py --execucao execucoes\v3 --porta 8790
```

**Linux**

```bash
python3 -m venv .venv
.venv/bin/pip install torch            # CUDA incluso; para só CPU: --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
.venv/bin/python src/servidor.py --execucao execucoes/v3 --porta 8790
```

**macOS (Apple Silicon)**

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-mlx.txt
.venv/bin/python src/servidor.py --execucao execucoes/v3 --porta 8790
```

Na primeira execução, a base (Qwen3-4B em 4 bits, ~2,3 GB) baixa sozinha do Hugging Face. No motor
PyTorch, ela é convertida em memória para 16 bits.

**Memória:** no MLX, ~5 GB de RAM livre (a base fica em 4 bits). No PyTorch, ~9–10 GB de RAM ou VRAM
(bf16/fp16 ocupam o mesmo espaço), ou seja, GPU NVIDIA com 12 GB ou mais. Para máquinas menores, o
Brier v1 (Qwen3-1.7B, `--execucao execucoes/v1`) precisa de ~4–5 GB e roda nos dois motores, com
menos acerto (veja Desempenho).

`"amostras": 6` no corpo liga as camadas estocásticas (≈ 6× a latência). No motor PyTorch, as amostras
usam a permutação das opções; o MC Dropout existe só no MLX, porque no PyTorch os adaptadores são
fundidos nos pesos.

**Testes** (qualquer sistema, sem baixar modelo): `python -m pytest tests -q`. O GitHub Actions roda
os testes em Ubuntu, Windows e macOS a cada push.

**Reproduzir o treino do zero** (o treino usa MLX, então exige Mac com Apple Silicon; instale
`requirements-mlx.txt` e `requirements-dados.txt`. Credenciais de um endpoint OpenAI-compatível para gerar dados, em
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

Conjuntos com **rótulo humano que não entram no treino**, 250 pedidos por conjunto. Cada célula:
acurácia · escore de Brier (menor é melhor).

| Conjunto | Tarefa | Brier v2 | Brier v3 | v4 antigo (pesos do Laya, fora do pacote) |
|---|---|---|---|---|
| ASSIN2 | implicação textual + similaridade | **72,8%** · 0,375 | 71,2% · 0,357 | 45,4% · 0,767 |
| B2W | nota de 1 a 5 + recomendação | **65,4%** · 0,459 | 61,6% · 0,484 | 50,6% · 0,809 |
| tweetSentBR | sentimento | **72,0%** · 0,421 | 69,2% · 0,455 | 56,4% · 0,594 |
| Sintético (teste) | 46 domínios, casos difíceis | **77,5%** · 0,304 | 75,7% · 0,318 | 44,3% · 0,750 |

| Robustez | Brier v4 | Brier v3 |
|---|---|---|
| Injeção de instruções (frases nunca vistas) | o ataque funcionou em 44,1% dos casos; acerto 47,6% → 38,6% | o ataque funcionou em 16,7% dos casos; acerto 70,3% → 67,3% |
| Documentos longos (~2.500 tokens, por janelas) | acerto 46,6% → 33,2% | acerto 73,5% → 59,1% |

Usados como estão, sem ajuste com dados em português, os pesos do Laya ficam de 15 a 33 pontos
abaixo do v2 nesses conjuntos. O v3, treinado contra injeção, perde de 1,6 a 3,8 pontos de acerto para o v2 nos conjuntos limpos. O v1 (Qwen3-1.7B) tinha 65,0% no ASSIN2 e 72,5% no
sintético. Números completos e histórico em [RESULTADOS.md](RESULTADOS.md).

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

| Conjunto | Brier v2 | Brier v3 | Comitê v2+v3 | Brier v4 | Jev | Leitura |
|---|---|---|---|---|---|---|
| ASSIN2 | 72,8% · 0,375 | 71,2% · 0,357 | 73,0% · 0,357 | 45,4% · 0,767 | 78,2% · 0,312 | Jev melhor que todas as versões |
| B2W | 65,4% · 0,459 | 61,6% · 0,484 | 63,4% · 0,465 | 50,6% · 0,809 | 66,8% · 0,527 | v2 empata com o Jev no acerto; v2, v3 e comitê mais bem calibrados que o Jev, com significância |
| tweetSentBR | 72,0% · 0,421 | 69,2% · 0,455 | 71,6% · 0,434 | 56,4% · 0,594 | 67,6% · 0,459 | v2 e comitê à frente do Jev, sem significância |
| Sintético | 77,5% · 0,304 | 75,7% · 0,318 | 77,0% · 0,303 | 44,3% · 0,750 | 79,5% · 0,287 | v2 e comitê empatam com o Jev |
| Injeção, frases nossas (taxa de ataque, menor é melhor) | não medido | **16,7%** | não medido | 44,1% | 21,4% | **v3 mais resistente que o Jev, com significância** |
| Injeção, teste cego (57 frases do mangaba-titan, n=200) | não medido | **17,5%** | 22,5% | não medido | 20,5% | v3 à frente do Jev, sem significância; comitê pior |
| Documentos longos (acerto limpo → longo) | não medido | 73,5% → 59,1% | não medido | 46,6% → 33,2% | 78,2% → 73,0% | Jev melhor |
| Latência mediana | ~1,3 s (Mac M5) | ~1,3 s (Mac M5) | ~2× o v3 (Mac M5) | 58–117 ms (Mac M5) | ~0,35 s (pela rede) | v4 mais rápido |

**Veredito pelo critério: nenhuma versão do Brier é melhor que o Jev ainda.** Vitórias em conjuntos
humanos: v2 = 0, v3 = 0, comitê = 0, v4 = 0. O v2 é o mais próximo (empata no sintético e no B2W, onde é mais
bem calibrado, e lidera o tweetSentBR sem significância). O v4 é o mais rápido, mas sem ajuste em
português fica atrás em acerto, em resistência a injeção e em documentos longos.

Vantagens que não dependem desse critério: roda offline, sem custo por chamada, e erra com menos
convicção quando erra.

**Migração:** quem já usa o formato do Jev pode apontar para `POST /v1/systemone` neste servidor, que
aceita o mesmo corpo. A rota nativa do Brier é `POST /v1/decide`.

Para refazer a comparação com a sua própria chave da TypeSafe, depois de `src/externos.py` e
`src/publicos.py`:

```bash
TYPESAFE_API_KEY=… .venv/bin/python src/comparar_jev.py   # coleta as respostas do Jev (cache local)
.venv/bin/python src/comparar_v2.py --execucoes execucoes/v2 execucoes/v3 comite   # critério de vitória
.venv/bin/python src/robustez.py --execucoes execucoes/v3                          # injeção e documentos longos
.venv/bin/python src/injecao_cega.py gera && .venv/bin/python src/injecao_cega.py avalia   # teste cego de injeção
.venv/bin/python src/laya_avaliar.py --modelo convaiinnovations/laya-multilingual --nome brier-v4   # v4 × Jev
```

## Testes práticos

### Corrida da página "Spider" (velocidade × acerto)

Cada modelo percorre os links da página [Spider](https://en.wikipedia.org/wiki/Spider) da Wikipédia em
inglês e decide, link a link, se o link nomeia uma aranha. O gabarito é objetivo: um link nomeia uma
aranha quando o item do Wikidata descende da ordem Araneae pela árvore taxonômica (112 aranhas em 429
links). Mesmo MacBook Air M5, mesma pergunta, um link por chamada, modelos carregados antes do cronômetro.

| | Brier v3 | Laya (antigo `rapido`) |
|---|---:|---:|
| Tempo por link | 256 ms | **26 ms** |
| Achados em 30 s (aranhas marcadas certo) | 17 | **92** |
| Alarmes falsos em 30 s | **28** | 215 |
| Acurácia na página inteira | **84,4%** (89,3% com a pergunta em inglês) | 45,2% (45,5% em inglês) |
| Precisão (quando diz "é aranha") | **62,7%** | 30,0% |
| Cobertura (aranhas encontradas) | **99,1%** | 82,1% |
| Escore de Brier (menor é melhor) | **0,131** | 0,295 |

**Leitura:** contando só achados por tempo, o Laya vence, porque olha dez vezes mais links. Mas erra
mais da metade: marca "Arthropoda" e "fangs" como aranha e deixa passar famílias como Theraphosidae. Os
erros do v3 são termos sobre aranhas (anatomia, taxonomia), quase sempre com probabilidade entre 0,5 e 0,75.
Pôr vários links numa só chamada (uma pergunta por link) não acelerou o v3 e derrubou a precisão.

```bash
python src/spider_gabarito.py                     # execucoes/spider/links.json (Wikipédia + Wikidata)
.venv/bin/python src/spider_corrida.py v3         # 1, 8, 16 e 32 links por chamada
.venv/bin/python src/spider_corrida.py rapido      # pesos do Laya, carregados direto
.venv/bin/python src/spider_corrida.py v3 en      # pergunta em inglês
```

### Tetris (decisão encadeada)

Cada peça vira uma pergunta `choice` com até 26 jogadas, descritas pelo efeito no tabuleiro
(`src/tetris.py`); um erro se acumula nas peças seguintes. Mesmas peças para todos, 150 peças no máximo.

| Semente | Brier v2 | Brier v3 | Regra gulosa | Jev | Aleatório (média de 100) |
|---|---|---|---|---|---|
| 7 | 1.500 pts · 70 peças | 600 · 53 | 5.200 · 145 | 5.500 · 150 | 18 · 26,8 |
| 11 | 300 · 45 | 1.000 · 55 | 6.700 · 150 | 4.800 · 150 | 11 · 26,4 |
| 23 | 1.900 · 69 | 400 · 47 | 6.300 · 150 | 3.300 · 116 | 12 · 27,0 |

O Brier vence as 300 partidas aleatórias de referência (v2 e v3), mas fica bem atrás da regra gulosa (uma
fórmula feita à mão para Tetris) e do Jev. O v3, treinado contra injeção, joga pior que o v2 no total.
Do Jev entram só os placares; as jogadas não são redistribuídas.

```bash
.venv/bin/python src/tetris_brier.py execucoes/v2 7   # o Brier joga (grava execucoes/tetris/brier_7.json)
python src/tetris.py                                   # regra gulosa
python src/tetris_aleatorio.py                         # Brier × 100 partidas aleatórias por semente
```

## Conteúdo do repositório

- `brier/`: pacote do Brier (v3, v2 e comitê; servidor, MCP, LangChain, regras contra injeção, mascaramento de dados pessoais)
- `clientes/typescript/`: cliente TypeScript
- `src/`: geração, rotulagem, treino, avaliação, servidor do v2 e comparação
- `execucoes/v3/`: adaptadores do Brier v3 (padrão do servidor) e calibração conformal
- `execucoes/v2/`: adaptadores do Brier v2 e calibração conformal
- `execucoes/v1/`: adaptadores do v1 (Qwen3-1.7B), com SWA e SWAG
- `execucoes/comparacao_v2.json`, `comparacao_v3.json`, `comparacao_comite.json`, `comparacao_v4.json`, `robustez_v3.json`, `injecao_cega.json`: métricas das comparações pareadas
- `dados/injecao_titan_frases.json` (ajuste das regras) e `dados/injecao_titan_teste.json` (teste cego): frases de injeção geradas pelo mangaba-titan
- `dados/sintetico*.jsonl`: os dados sintéticos de treino e teste, com votos e alvos
- `execucoes/spider/`: gabarito (Wikidata) e resultados da corrida da página "Spider"
- `execucoes/tetris/`: partidas de Tetris do Brier v2 (`brier_*.json`), do v3 (`brier_v3_*.json`) e da regra gulosa

Não redistribuímos dados de terceiros: ASSIN, ASSIN2, B2W, FaQuAD-NLI, HateBR e tweetSentBR são baixados
e convertidos por `src/externos.py` e `src/publicos.py`, sob as licenças de cada conjunto.

## Licença

**Licença proprietária da Mangaba AI** a partir da versão 5.0.0: uso, cópia, modificação e
distribuição só com autorização por escrito (texto completo em [LICENSE](LICENSE)). O código, os
pesos e os dados publicados até a tag `ultima-apache-2.0`, inclusive os adaptadores v2 e v3 no Hugging
Face tal como publicados até então, continuam sob a Apache-2.0 ([licenses/APACHE-2.0.txt](licenses/APACHE-2.0.txt)).

Componentes de terceiros em [NOTICE](NOTICE): o Qwen3 (Apache-2.0) é a base dos adaptadores; o Laya
(Apache-2.0) aparece só nos scripts de pesquisa e comparação. Jev e TypeSafe são marcas da TypeSafe
AI, citadas apenas para fins de comparação de desempenho. O Brier não tem vínculo com a TypeSafe AI.

Licenciamento: [mangaba.ia.br](https://mangaba.ia.br).
