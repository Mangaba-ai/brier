# Brier

[![testes](https://github.com/Mangaba-ai/brier/actions/workflows/testes.yml/badge.svg)](https://github.com/Mangaba-ai/brier/actions/workflows/testes.yml)

**Modelo aberto de decisão tipada e calibrada.** Roda localmente e não depende de nenhum serviço externo.

O Brier recebe um conteúdo (`state`) e uma ou mais perguntas com respostas pré-definidas. Para cada
pergunta, devolve a opção escolhida e a **probabilidade** de cada opção. Ele não gera texto: responde
sempre dentro das opções que você definiu, e diz o quanto tem certeza.

O nome vem do [escore de Brier](https://en.wikipedia.org/wiki/Brier_score) (Glenn Brier, 1950), a regra
de pontuação que mede se uma probabilidade bate com a realidade. É isso que o treino otimiza: acertar a
**probabilidade**, não só a resposta.

- **Código e pesos abertos:** Apache-2.0.
- **Local e multiplataforma:** Windows, Linux e macOS, em CPU, GPU NVIDIA ou Apple Silicon. Offline e
  sem custo por chamada.
- **Tipado:** a resposta sempre cabe no tipo pedido, por construção.
- **Calibrado:** quando diz 80%, acerta perto de 80% das vezes. Isso permite decidir o que automatizar
  e o que mandar para uma pessoa.

## Duas versões: escolha pela prioridade

| | **Brier v4** (rápido) | **Brier v2** (mais preciso) |
|---|---|---|
| Modelo | arquitetura e pesos do [Laya](https://github.com/NandhaKishorM/laya) como estão (`convaiinnovations/laya-multilingual`, mmBERT de 322M), sem ajuste para o português | Qwen3-4B + adaptadores LoRA treinados pelo Brier em português |
| Latência por pedido (Mac M5) | **58–117 ms** | ~1,3 s |
| Acerto em português (média dos 3 conjuntos humanos) | 50,8% | **70,1%** |
| Recursos de produção | lotes, janelas para textos longos, abstenção, roteador por idioma, auditoria, mascaramento de dados pessoais, MCP, LangChain, cliente TypeScript, Docker | servidor HTTP, conjunto conformal, incerteza epistêmica |
| Como rodar | `brier-serve` (pacote `brier/`) | `src/servidor.py --execucao execucoes/v2` |

**Recomendação hoje:** use o **v2** quando o acerto importa e o **v4** quando a velocidade e os recursos
de produção importam mais (triagem de alto volume com abstenção ligada, por exemplo). Sem ajuste fino em
português, os pesos do Laya acertam bem menos que o v2 nos nossos testes (detalhes em "Desempenho").

```bash
pip install "brier @ git+https://github.com/Mangaba-ai/brier"
brier-serve --porta 8790          # Brier v4
```

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

**Brier v4.** Encoder bidirecional (mmBERT de 322M) com uma cabeça de decisão de 2 camadas, do projeto
Laya. Cada opção vira um marcador na sequência; a cabeça dá uma nota a cada marcador e a distribuição é
o softmax dessas notas. Os pesos foram treinados pelo Laya com reforço contra regras de pontuação
próprias; o Brier v4 os usa como estão.

**Brier v2.** Os parágrafos abaixo descrevem o v2, treinado pelo próprio Brier.

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

### Brier v4 (pacote `brier/`)

```bash
pip install "brier @ git+https://github.com/Mangaba-ai/brier"      # Windows, Linux e macOS
brier-serve --porta 8790 [--auditoria auditoria.jsonl] [--mascarar-pii] [--outros-idiomas]
```

```python
from brier import Brier
b = Brier(mascarar_pii=True)
b.decide(texto, perguntas, min_confidence=0.8)     # "abstencao" lista as perguntas abaixo do mínimo
b.decide_lote([texto1, texto2, texto3], perguntas)  # vários textos numa chamada
```

| Rota | Para quê |
|---|---|
| `POST /v1/decide` | um `state`; `min_confidence` e `longo` ("auto", "sim", "nao") opcionais |
| `POST /v1/decide/lote` | até 256 `states` com as mesmas perguntas |
| `POST /v1/systemone` | mesmo corpo do `/v1/decide`, para quem migra do Jev |

- **Textos longos:** quando o `state` não cabe na janela, é lido por janelas sobrepostas e a resposta
  traz um aviso.
- **Auditoria:** `--auditoria arquivo.jsonl` grava hash do texto, roteamento, latência e respostas,
  nunca o texto.
- **Dados pessoais:** `--mascarar-pii` troca CPF, CNPJ, cartão, e-mail, telefone e CEP por marcadores
  antes da decisão.
- **Integrações:** servidor MCP (`brier-mcp`, ferramentas `brier_decidir` e `brier_decidir_lote`),
  LangChain/LangGraph (`brier.langchain`), cliente TypeScript (`clientes/typescript/brier.ts`) e
  `Dockerfile`.

Medido no MacBook Air M5: {c['assin2']['latencia_p50_ms']:.0f}–{c['sintetico']['latencia_p50_ms']:.0f} ms por pedido, ~25 ms por texto em lote, ~{r['longo']['latencia_p50_ms']/1000:.1f} s para um documento de
~2.500 tokens lido por janelas.

### Brier v2 (Qwen3-4B)

O v2 roda em **Windows, Linux e macOS** e tem dois motores, que dão as mesmas respostas:

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
.venv\Scripts\python src\servidor.py --execucao execucoes\v2 --porta 8790
```

**Linux**

```bash
python3 -m venv .venv
.venv/bin/pip install torch            # CUDA incluso; para só CPU: --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
.venv/bin/python src/servidor.py --execucao execucoes/v2 --porta 8790
```

**macOS (Apple Silicon)**

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-mlx.txt
.venv/bin/python src/servidor.py --execucao execucoes/v2 --porta 8790
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

| Conjunto | Tarefa | Brier v4 (pesos do Laya) | Brier v2 |
|---|---|---|---|
| ASSIN2 | implicação textual + similaridade | 45,4% · 0,767 | **72,8%** · 0,375 |
| B2W | nota de 1 a 5 + recomendação | 50,6% · 0,809 | **65,4%** · 0,459 |
| tweetSentBR | sentimento | 56,4% · 0,594 | **72,0%** · 0,421 |
| Sintético (teste) | 46 domínios, casos difíceis | 44,3% · 0,750 | **77,5%** · 0,304 |

| Robustez | Brier v4 |
|---|---|
| Injeção de instruções (frases nunca vistas) | o ataque funcionou em 44,1% dos casos; acerto 47,6% → 38,6% |
| Documentos longos (~2.500 tokens, por janelas) | acerto 46,6% → 33,2% |

Usados como estão, sem ajuste com dados em português, os pesos do Laya ficam de 15 a 33 pontos
abaixo do v2 nesses conjuntos. O v1 (Qwen3-1.7B) tinha 65,0% no ASSIN2 e 72,5% no
sintético. Números completos e histórico em [RESULTADOS.md](RESULTADOS.md).

## Comparação com o Jev (referência externa)

O Brier é independente: código próprio, pesos abertos e nenhuma dependência de serviço de terceiros
para funcionar. O **Jev**, da TypeSafe AI, é um modelo comercial fechado de decisão tipada. Ele entra
**só como régua de desempenho**, porque é a referência mais próxima do que o Brier faz.

- Nenhuma resposta do Jev foi usada para treinar o Brier.
- As respostas do Jev coletadas na comparação não são redistribuídas neste repositório.

**Como comparamos:** os mesmos pedidos foram enviados aos dois modelos, item a item, com intervalo de
confiança de 95% por bootstrap pareado. O critério de vitória foi fixado **antes** da medição: o Brier só
é "melhor que o Jev" se vencer em acurácia **e** escore de Brier, com IC favorável, em pelo menos 2 dos
3 conjuntos humanos.

| Conjunto | Brier v4 | Brier v2 | Jev | Leitura |
|---|---|---|---|---|
| ASSIN2 | 45,4% · 0,767 | 72,8% · 0,375 | 78,2% · 0,312 | Jev melhor que as duas versões |
| B2W | 50,6% · 0,809 | 65,4% · 0,459 | 66,8% · 0,527 | v2 empata com o Jev no acerto e é mais bem calibrado; v4 abaixo |
| tweetSentBR | 56,4% · 0,594 | 72,0% · 0,421 | 67,6% · 0,459 | v2 à frente do Jev, sem significância; v4 abaixo |
| Sintético | 44,3% · 0,750 | 77,5% · 0,304 | 79,5% · 0,287 | v2 empata com o Jev; v4 abaixo |
| Injeção (taxa de ataque, menor é melhor) | 44,1% | não medido | 21,4% | Jev mais resistente |
| Documentos longos (acerto limpo → longo) | 46,6% → 33,2% | não medido | 78,2% → 73,0% | Jev melhor |
| Latência mediana | 58–117 ms (Mac M5) | ~1,3 s (Mac M5) | ~0,35 s (pela rede) | v4 mais rápido |

**Veredito pelo critério: nenhuma versão do Brier é melhor que o Jev ainda.** Vitórias em conjuntos
humanos: v4 = 0, v2 = 0. O v2 é o mais próximo (empata no sintético e no B2W, onde é mais
bem calibrado, e lidera o tweetSentBR sem significância). O v4 é o mais rápido, mas sem ajuste em
português fica atrás em acerto, em resistência a injeção e em documentos longos.

Vantagens que não dependem desse critério: roda offline, sem custo por chamada, com código e pesos
abertos, e erra com menos convicção quando erra.

**Migração:** quem já usa o formato do Jev pode apontar para `POST /v1/systemone` neste servidor, que
aceita o mesmo corpo. A rota nativa do Brier é `POST /v1/decide`.

Para refazer a comparação com a sua própria chave da TypeSafe, depois de `src/externos.py` e
`src/publicos.py`:

```bash
TYPESAFE_API_KEY=… .venv/bin/python src/comparar_jev.py   # coleta as respostas do Jev (cache local)
.venv/bin/python src/comparar_v2.py                       # v1 × v2 × Jev, critério de vitória
.venv/bin/python src/laya_avaliar.py --modelo convaiinnovations/laya-multilingual --nome brier-v4   # v4 × Jev
```

## Conteúdo do repositório

- `brier/`: pacote do Brier v4 (motor, servidor, MCP, LangChain, mascaramento de dados pessoais)
- `clientes/typescript/`: cliente TypeScript
- `src/`: geração, rotulagem, treino, avaliação, servidor do v2 e comparação
- `execucoes/v2/`: adaptadores do Brier v2 (padrão) e calibração conformal
- `execucoes/v1/`: adaptadores do v1 (Qwen3-1.7B), com SWA e SWAG
- `execucoes/comparacao_v2.json`, `execucoes/comparacao_v4.json`: métricas das comparações pareadas
- `dados/sintetico*.jsonl`: os dados sintéticos de treino e teste, com votos e alvos

Não redistribuímos dados de terceiros: ASSIN, ASSIN2, B2W, FaQuAD-NLI, HateBR e tweetSentBR são baixados
e convertidos por `src/externos.py` e `src/publicos.py`, sob as licenças de cada conjunto.

## Licença

Apache-2.0, a mesma do Qwen3 e do Laya. O Brier v4 usa a arquitetura e os pesos do Laya
(NandhaKishorM / convaiinnovations, Apache-2.0), com crédito. Jev e TypeSafe são marcas da TypeSafe AI, citadas apenas para fins de
comparação de desempenho. O Brier não tem vínculo com a TypeSafe AI.
