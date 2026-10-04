"""Avalia um checkpoint Laya (pronto ou ajustado pelo Brier) nos mesmos pedidos do Brier v2 e do Jev.

Usa o mesmo pareamento e o mesmo critério de vitória de comparar_v2.py, mais injeção e contexto longo
(robustez.py). O formato de resposta do Laya é o mesmo do Jev, então os itens saem por para_itens.

Uso: python src/laya_avaliar.py --modelo convaiinnovations/laya-multilingual [--modelo-pasta execucoes/rapido]
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from comparar_jev import para_itens  # noqa: E402
from comparar_v2 import HUMANOS, acc, brier, conjuntos, ic  # noqa: E402
from dados import metricas  # noqa: E402


def h(e):
    return hashlib.sha1(json.dumps([e["state"], e["questions"]], sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def le(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def roda(agente, exs, longo=False):
    itens, lat = [], []
    for e in exs:
        t = time.time()
        r = agente.predict_long(e["state"], e["questions"]) if longo else agente.predict(e["state"], e["questions"])
        lat.append(time.time() - t)
        itens.append(para_itens(e, r))
    return itens, lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="convaiinnovations/laya-multilingual")
    ap.add_argument("--nome", default=None)
    ap.add_argument("--n", type=int, default=250)
    a = ap.parse_args()
    import laya
    agente = laya.load(a.modelo)
    nome = a.nome or Path(a.modelo).name
    dados = conjuntos(a.n)
    rel = {"modelo": a.modelo, "conjuntos": {}}
    vitorias = 0
    for cj, (exs, resps) in dados.items():
        L, lat = roda(agente, exs)
        J = [para_itens(e, r) for e, r in zip(exs, resps)]
        Lf, Jf = [i for x in L for i in x], [i for x in J for i in x]
        ia, ib = ic(Lf, Jf, acc), ic(Jf, Lf, brier)
        vence = ia[0] > 0 and ib[0] > 0
        vitorias += cj in HUMANOS and vence
        rel["conjuntos"][cj] = {"modelo": metricas(Lf), "jev": metricas(Jf), "ic95_acc": ia, "ic95_brier": ib,
                                "vence": vence, "latencia_p50_ms": round(1000 * float(np.median(lat)), 1)}
        print(nome, cj, "acc", rel["conjuntos"][cj]["modelo"]["acc"], "× jev", rel["conjuntos"][cj]["jev"]["acc"],
              "| brier", rel["conjuntos"][cj]["modelo"]["brier"], "×", rel["conjuntos"][cj]["jev"]["brier"],
              "| ICacc", ia, "| p50", rel["conjuntos"][cj]["latencia_p50_ms"], "ms", flush=True)
    rel["vitorias_humanas"] = vitorias

    # robustez: injeção (famílias nunca vistas) e contexto longo (janelas do predict_long)
    from robustez import avalia, limpos
    base = limpos()
    for tipo, arq in (("injecao", "dados/adv_teste_injecao.jsonl"), ("longo", "dados/adv_teste_longo.jsonl")):
        alt = le(arq)
        chave = "ataque" if tipo == "injecao" else "longo"
        lim = [base[e[chave]["id_limpo"]] for e in alt]
        il, _ = roda(agente, lim)
        ia_, lat = roda(agente, alt, longo=(tipo == "longo"))
        rel[tipo] = avalia(il, ia_, alt)
        rel[tipo]["latencia_p50_ms"] = round(1000 * float(np.median(lat)), 1)
        print(nome, tipo, {k: rel[tipo][k] for k in ("taxa_ataque", "mudou") if k in rel[tipo]},
              "acc", rel[tipo]["acc_limpo"]["acc"], "→", rel[tipo]["acc_alterado"]["acc"], flush=True)
    Path("execucoes/laya").mkdir(parents=True, exist_ok=True)
    Path(f"execucoes/laya/{nome}.json").write_text(json.dumps(rel, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
