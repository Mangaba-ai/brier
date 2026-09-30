"""Comparação pareada: decisor local × Jev, nos MESMOS pedidos, item a item.

Lê as respostas do Jev de dados/jev_cache.jsonl (geradas por comparar_jev.py ou pelo relay) e roda o
decisor local nos mesmos pedidos. Métricas por conjunto, por tipo e por armadilha, com IC de 95% por
bootstrap pareado na diferença (local − Jev).

Uso: python src/comparar.py --execucao execucoes/v1 --n 250
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load

sys.path.insert(0, str(Path(__file__).parent))
from avaliar import coleta  # noqa: E402
from comparar_jev import para_itens  # noqa: E402
from dados import carrega, metricas  # noqa: E402
from decisor import Codificador  # noqa: E402
from treinar import prepara_lora  # noqa: E402


def h(ex):
    return hashlib.sha1(json.dumps([ex["state"], ex["questions"]], sort_keys=True,
                                   ensure_ascii=False).encode()).hexdigest()


def bootstrap(a, b, fn, n=2000, semente=0):
    """IC 95% de fn(a) − fn(b) reamostrando os MESMOS índices nos dois (pareado)."""
    rnd = np.random.default_rng(semente)
    idx = np.arange(len(a))
    d = []
    for _ in range(n):
        s = rnd.choice(idx, len(idx))
        d.append(fn([a[i] for i in s]) - fn([b[i] for i in s]))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    mx.set_cache_limit(1024 ** 3)  # sem teto o cache do MLX cresce a cada comprimento novo e leva a swap
    ap = argparse.ArgumentParser()
    ap.add_argument("--execucao", default="execucoes/v1")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--amostras", type=int, default=1)
    a = ap.parse_args()
    d = Path(a.execucao)
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    cache = {}
    for l in Path("dados/jev_cache.jsonl").read_text(encoding="utf-8").splitlines():
        x = json.loads(l)
        cache[x["h"]] = x["r"]
    le = lambda p: [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]  # noqa: E731
    conjuntos = {"teste": carrega(cfg["dados"])["teste"][:a.n], "assin2": le("dados/ext_assin2.jsonl")[:a.n],
                 "b2w": le("dados/ext_b2w.jsonl")[:a.n]}
    model, tok = load(cfg["modelo"])
    prepara_lora(model, cfg["rank"], cfg["dropout"], cfg["camadas"])
    model.load_weights(str(d / "adaptadores.safetensors"), strict=False)
    cod = Codificador(tok)
    saida = {}
    for nome, exs in conjuntos.items():
        exs = [e for e in exs if h(e) in cache]
        jev = [dict(i, grupo=e.get("armadilha") or "nenhuma") for e in exs for i in para_itens(e, cache[h(e)])]
        loc = []
        for e in exs:  # um por vez para manter a ordem idêntica à do Jev
            its = coleta(model, cod, [e], a.amostras, a.amostras > 1, False, None, cfg["max_prefixo"])
            loc += [dict(i, grupo=e.get("armadilha") or "nenhuma") for i in its]
        assert len(loc) == len(jev)
        acc = lambda it: float(np.mean([np.argmax(i["p"]) == i["y"] for i in it]))  # noqa: E731
        nll = lambda it: float(np.mean([-np.log(max(i["p"][i["y"]], 1e-9)) for i in it]))  # noqa: E731
        r = {"local": metricas(loc), "jev": metricas(jev),
             "ic95_acc_local_menos_jev": bootstrap(loc, jev, acc),
             "ic95_nll_local_menos_jev": bootstrap(loc, jev, nll),
             "jev_latencia_p50": round(float(np.median([cache[h(e)]["latencia"] for e in exs])), 3)}
        for t in ("choice", "score", "noul"):
            lt, jt = [i for i in loc if i["tipo"] == t], [i for i in jev if i["tipo"] == t]
            if lt:
                r[t] = {"local": metricas(lt), "jev": metricas(jt)}
        if nome == "teste":
            r["por_armadilha"] = {}
            for g in sorted({i["grupo"] for i in loc}):
                lg, jg = [i for i in loc if i["grupo"] == g], [i for i in jev if i["grupo"] == g]
                r["por_armadilha"][g] = {"n": len(lg), "acc_local": round(acc(lg), 3), "acc_jev": round(acc(jg), 3),
                                         "nll_local": round(nll(lg), 3), "nll_jev": round(nll(jg), 3)}
        saida[nome] = r
        print(nome, json.dumps({k: v for k, v in r.items() if k != "por_armadilha"}, ensure_ascii=False), flush=True)
    (d / "comparacao_jev.json").write_text(json.dumps(saida, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
