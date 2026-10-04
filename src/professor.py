"""Titan (mangaba.router) como professor do Brier.

O mangaba-titan é o LLM grande da Mangaba servido pelo mangaba.router (OpenAI-compatível). Aqui ele:
  - joga Tetris raciocinando (para medir se vale a pena imitá-lo);
  - rotula posições de Tetris com a melhor jogada (dados de treino do Brier);
  - dá o voto que falta nos exemplos sintéticos sem maioria (ver rotular.py --professor).
A GPU do Titan é compartilhada com as contas do router: poucas chamadas simultâneas, sempre.

Credenciais no .env (fora do git): MANGABA_ROUTER_KEY=mr-...  [MANGABA_ROUTER_URL, TITAN_MODELO]

Uso:
  python src/professor.py teste                 # 1 chamada: confere acesso e latência
  python src/professor.py tetris --semente 7    # Titan joga uma partida (comparar com Jev e guloso)
  python src/professor.py posicoes --n 800      # rotula posições de Tetris para treino
"""
import argparse
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
import chaves  # noqa: E402
import tetris  # noqa: E402

_C = chaves.carregar()
URL = _C.get("MANGABA_ROUTER_URL", "https://mangabarouter.store/v1").rstrip("/")
CHAVE = _C.get("MANGABA_ROUTER_KEY", "")
MODELO = _C.get("TITAN_MODELO", "mangaba-titan")
_cli = httpx.Client(timeout=300)


def titan(prompt: str, max_tokens: int = 4000, temperatura: float = 0.0) -> tuple[str, float]:
    if not CHAVE:
        sys.exit("defina MANGABA_ROUTER_KEY no .env")
    corpo = {"model": MODELO, "messages": [{"role": "user", "content": prompt}],
             "temperature": temperatura, "max_tokens": max_tokens}
    for t in range(6):
        t0 = time.time()
        try:
            r = _cli.post(f"{URL}/chat/completions", json=corpo, headers={"Authorization": f"Bearer {CHAVE}"})
        except httpx.TransportError:
            time.sleep(5 * (t + 1))
            continue
        if r.status_code in (429, 500, 502, 503, 529):
            time.sleep(10 * (t + 1))
            continue
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        return (msg.get("content") or ""), time.time() - t0
    raise RuntimeError("Titan não respondeu após 6 tentativas")


PROMPT_TETRIS = """Você é um jogador experiente de Tetris. Analise o tabuleiro e escolha a melhor jogada.
Critérios, em ordem: não criar buracos; manter o tabuleiro baixo e plano; deixar um poço para a peça I;
limpar linhas quando possível; considere também a próxima peça.

{estado}

JOGADAS POSSÍVEIS:
{opcoes}

Pense brevemente e termine com uma linha exatamente assim:
RESPOSTA: <letra da melhor jogada>, <letra da segunda melhor>"""


def escolhe_tetris(estado: str, q: dict) -> tuple[dict, float, str]:
    crit = q["jogada"]["criteria"]
    opcoes = "\n".join(f"{k}) {v}" for k, v in crit.items())
    txt, lat = titan(PROMPT_TETRIS.format(estado=estado, opcoes=opcoes))
    m = re.findall(r"resposta:\s*\**\s*([a-z])\b\s*(?:,\s*([a-z])\b)?", txt.lower())
    if not m or m[-1][0] not in crit:
        return {k: 1 / len(crit) for k in crit}, lat, txt  # sem resposta válida: uniforme (registrado)
    a, b = m[-1]
    probs = {k: 0.0 for k in crit}
    if b and b in crit and b != a:
        probs[a], probs[b] = 0.8, 0.2
    else:
        probs[a] = 1.0
    return probs, lat, txt


def cmd_teste():
    txt, lat = titan("Responda só: ok", max_tokens=200)
    print(f"modelo {MODELO} respondeu em {lat:.1f}s: {txt.strip()[:80]!r}")


def cmd_tetris(semente: int, max_pecas: int):
    falhas = []

    def decide(estado, q):
        probs, lat, txt = escolhe_tetris(estado, q)
        if max(probs.values()) < 0.5:
            falhas.append(txt[-300:])
        return probs, lat

    r = tetris.joga(decide, semente=semente, max_pecas=max_pecas,
                    registro=lambda i, p: print(i, p, flush=True) if i % 10 == 0 else None)
    r["modelo"], r["respostas_invalidas"] = "titan", len(falhas)
    Path("execucoes/tetris").mkdir(parents=True, exist_ok=True)
    Path(f"execucoes/tetris/titan_{semente}.json").write_text(json.dumps(r), encoding="utf-8")
    print({k: v for k, v in r.items() if k not in ("passos", "sequencia")})


def _posicoes(n: int, semente: int):
    """Posições variadas: partidas de um jogador guloso com ruído (explora tabuleiros bons e ruins)."""
    rnd, out = random.Random(semente), []
    jogo = 0
    while len(out) < n:
        seq = tetris.sequencia(1000 + jogo, 200)
        tab, ruido = tetris.vazio(), rnd.choice([0.0, 0.15, 0.35])
        for i in range(150):
            js = tetris.jogadas(tab, seq[i])
            if not js:
                break
            out.append({"tab": [l[:] for l in tab], "peca": seq[i], "prox": seq[i + 1]})
            def nota(j):
                return j["linhas"] * 10 - max(0, j["buracos_novos"]) * 5 - j["altura_max"] - 0.3 * j["irregularidade"]
            j = rnd.choice(js) if rnd.random() < ruido else max(js, key=nota)
            forma = tetris.rotacoes(seq[i])[j["rot"]]
            tab, _ = tetris.aplica(tab, forma, tetris.solta(tab, forma, j["col"]), j["col"])
        jogo += 1
    rnd.shuffle(out)
    return out[:n]


def cmd_posicoes(n: int, paralelo: int, saida: str):
    """Titan rotula posições; cada uma vira um exemplo no formato do Brier (alvo 0,8 / 0,2)."""
    destino = Path(saida)
    feitos = sum(1 for _ in destino.open(encoding="utf-8")) if destino.exists() else 0
    pos = _posicoes(n, semente=99)[feitos:]
    print(f"{feitos} já rotuladas, faltam {len(pos)}", flush=True)

    def rotula(item):
        i, p = item
        js = tetris.jogadas(p["tab"], p["peca"])
        estado, q = tetris.pedido(p["tab"], p["peca"], p["prox"], js)
        probs, lat, _ = escolhe_tetris(estado, q)
        if max(probs.values()) < 0.5:
            return None
        chaves_ = list(q["jogada"]["criteria"])
        ouro = max(probs, key=probs.get)
        alvo = [probs[k] * 0.95 + 0.05 / len(chaves_) for k in chaves_]
        return {"id": f"tetris-{feitos + i}", "fonte": "tetris_titan", "state": estado, "questions": q,
                "rotulos": {"jogada": {"ouro": ouro, "alvo": alvo}}, "latencia": round(lat, 2)}

    with destino.open("a", encoding="utf-8") as f, ThreadPoolExecutor(paralelo) as pool:
        for i, ex in enumerate(pool.map(rotula, enumerate(pos)), 1):
            if ex:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                f.flush()
            if i % 25 == 0:
                print(f"{feitos + i} posições", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("comando", choices=["teste", "tetris", "posicoes"])
    ap.add_argument("--semente", type=int, default=7)
    ap.add_argument("--max-pecas", type=int, default=150)
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--paralelo", type=int, default=3)
    ap.add_argument("--saida", default="dados/tetris_titan.jsonl")
    a = ap.parse_args()
    if a.comando == "teste":
        cmd_teste()
    elif a.comando == "tetris":
        cmd_tetris(a.semente, a.max_pecas)
    else:
        cmd_posicoes(a.n, a.paralelo, a.saida)


if __name__ == "__main__":
    main()
