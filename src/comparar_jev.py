"""Roda os MESMOS pedidos no Jev e mede com as mesmas métricas do decisor local.

Endpoint: https://api.typesafe.ai/v1/systemone com TYPESAFE_API_KEY, ou o mangaba.router
(JEV_URL=https://…/v1/systemone e MANGABA_ROUTER_KEY). As respostas ficam em cache em
dados/jev_cache.jsonl para não pagar duas vezes.

Uso: TYPESAFE_API_KEY=… python src/comparar_jev.py --n 250
"""
import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from dados import carrega, indice_ouro, metricas  # noqa: E402
from decisor import ordem_canonica  # noqa: E402

URL = os.environ.get("JEV_URL", "https://api.typesafe.ai/v1/systemone")
CHAVE = os.environ.get("TYPESAFE_API_KEY") or os.environ.get("MANGABA_ROUTER_KEY")
CACHE = Path("dados/jev_cache.jsonl")


def chama(ex, cli):
    corpo = {"model": "jev-latest", "state": ex["state"], "questions": ex["questions"]}
    for tentativa in range(3):
        t = time.time()
        r = cli.post(URL, json=corpo, headers={"Authorization": f"Bearer {CHAVE}"})
        if r.status_code in (429, 529):
            time.sleep(2 * (tentativa + 1))
            continue
        r.raise_for_status()
        return {"answers": r.json()["answers"], "latencia": time.time() - t}
    raise RuntimeError("Jev recusou 3 vezes")


def para_itens(ex, resp):
    itens = []
    for k, q in ex["questions"].items():
        a = resp["answers"][k]
        canon = ordem_canonica(q)
        if q["type"] == "noul":
            p = [a["noul"], 1 - a["noul"]]
        elif q["type"] == "choice":
            p = [a["probabilities"].get(o, 0.0) for o in canon]
        else:
            p = [a["probabilities"].get(str(i), 0.0) for i in canon]
        s = sum(p) or 1.0
        itens.append({"tipo": q["type"], "p": [x / s for x in p], "y": indice_ouro(q, ex["rotulos"][k]["ouro"])})
    return itens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--paralelo", type=int, default=8)
    a = ap.parse_args()
    if not CHAVE:
        sys.exit("defina TYPESAFE_API_KEY (ou JEV_URL + MANGABA_ROUTER_KEY)")
    le = lambda p: [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]  # noqa: E731
    conjuntos = {"teste": carrega()["teste"][:a.n], "assin2": le("dados/ext_assin2.jsonl")[:a.n],
                 "b2w": le("dados/ext_b2w.jsonl")[:a.n], "tweetsentbr": le("dados/ext_tweetsentbr.jsonl")[:a.n]}
    cache = {}
    if CACHE.exists():
        for l in CACHE.read_text().splitlines():
            d = json.loads(l)
            cache[d["h"]] = d["r"]
    h = lambda ex: hashlib.sha1(json.dumps([ex["state"], ex["questions"]], sort_keys=True,  # noqa: E731
                                           ensure_ascii=False).encode()).hexdigest()
    cli = httpx.Client(timeout=60)
    resultado = {}
    with CACHE.open("a") as f, ThreadPoolExecutor(a.paralelo) as pool:
        for nome, exs in conjuntos.items():
            faltam = [e for e in exs if h(e) not in cache]
            for e, r in zip(faltam, pool.map(lambda e: chama(e, cli), faltam)):
                cache[h(e)] = r
                f.write(json.dumps({"h": h(e), "r": r}, ensure_ascii=False) + "\n")
            itens = [i for e in exs for i in para_itens(e, cache[h(e)])]
            lat = sorted(cache[h(e)]["latencia"] for e in exs)
            resultado[nome] = {"geral": metricas(itens), "latencia_p50": round(lat[len(lat) // 2], 3),
                               **{t: metricas([i for i in itens if i["tipo"] == t]) for t in ("choice", "score", "noul")}}
            print(nome, resultado[nome], flush=True)
    Path("execucoes").mkdir(exist_ok=True)
    Path("execucoes/jev.json").write_text(json.dumps(resultado, indent=1))


if __name__ == "__main__":
    main()
