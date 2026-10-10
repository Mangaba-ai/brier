"""Tetris: Brier (registros em execucoes/tetris/) contra um jogador aleatório nas mesmas peças.

O jogador aleatório sorteia uma das mesmas jogadas oferecidas ao Brier. Para não depender de um
sorteio favorável, joga 100 partidas por sequência de peças (sorteios 1000*semente + i).

Uso: python src/tetris_aleatorio.py
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import tetris  # noqa: E402

PASTA = Path(__file__).resolve().parent.parent / "execucoes/tetris"
SEMENTES = (7, 11, 23)
N = 100


def aleatorio(rng):
    def decide(estado, q):
        chaves = list(q["jogada"]["criteria"])
        k = rng.choice(chaves)
        return {c: (1.0 if c == k else 0.0) for c in chaves}, 0.0
    return decide


def main():
    jogos = {s: [tetris.joga(aleatorio(random.Random(1000 * s + i)), semente=s, max_pecas=150) for i in range(N)]
             for s in SEMENTES}
    for versao, prefixo in (("v2", "brier_"), ("v3", "brier_v3_")):
        for s in SEMENTES:
            b = json.loads((PASTA / f"{prefixo}{s}.json").read_text())
            vence = sum(b["pontos"] > j["pontos"] or (b["pontos"] == j["pontos"] and b["pecas"] > j["pecas"])
                        for j in jogos[s])
            media_pts = sum(j["pontos"] for j in jogos[s]) / N
            media_pc = sum(j["pecas"] for j in jogos[s]) / N
            print(f"Brier {versao}, semente {s}: {b['pontos']} pontos em {b['pecas']} peças; vence {vence}/{N} "
                  f"(aleatório: média de {media_pts:.0f} pontos e {media_pc:.1f} peças)")


if __name__ == "__main__":
    main()
