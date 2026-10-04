"""Partida de Tetris jogada pelo Brier (motor MLX), com registro de cada decisão."""
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

sys.path.insert(0, str(Path(__file__).parent))
import tetris  # noqa: E402
from decisor import Decisor  # noqa: E402
from treinar import prepara_lora  # noqa: E402


def main():
    mx.set_cache_limit(1024 ** 3)
    execucao = sys.argv[1] if len(sys.argv) > 1 else "execucoes/v2"
    semente = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    cfg = json.loads(Path(execucao, "config.json").read_text(encoding="utf-8"))
    model, tok = load(cfg["modelo"])
    prepara_lora(model, cfg["rank"], cfg["dropout"], cfg["camadas"])
    model.load_weights(str(Path(execucao, "adaptadores.safetensors")), strict=False)
    model.eval()
    dec = Decisor(model, tok)

    def decide(estado, q):
        t = time.time()
        r = dec.decide({"state": estado, "questions": q})["answers"]["jogada"]
        return r["probabilities"], time.time() - t

    def registro(i, p):
        if i % 10 == 0:
            print(i, p, flush=True)

    r = tetris.joga(decide, semente=semente, max_pecas=150, registro=registro)
    r["modelo"] = f"brier ({Path(execucao).name})"
    Path(f"execucoes/tetris/brier_{semente}.json").write_text(json.dumps(r), encoding="utf-8")
    print({k: v for k, v in r.items() if k not in ("passos", "sequencia")})


if __name__ == "__main__":
    main()
