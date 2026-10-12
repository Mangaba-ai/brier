"""Avalia um modelo de decisão servido por HTTP (/v1/systemone, formato Jev) nos mesmos pedidos do Brier v2 e do Jev.

Mesmo pareamento e mesmo critério de vitória de comparar_v2.py / laya_avaliar.py: o modelo vence num conjunto
humano quando o IC95% favorece ao mesmo tempo a acurácia e o escore de Brier. As respostas do Jev vêm do cache
(dados/jev_cache.jsonl), sem chamada paga.

Uso: CHAVE=... python src/avaliar_remoto.py --url https://llm-poc.mangaba.ia.br/v1/systemone --modelo laya --n 100
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from comparar_jev import para_itens  # noqa: E402
from comparar_v2 import HUMANOS, acc, brier, conjuntos, ic  # noqa: E402
from dados import metricas  # noqa: E402


def decide(url, modelo, e, chave, tentativas=3):
    corpo = json.dumps({"model": modelo, "state": e["state"], "questions": e["questions"]}).encode()
    # User-Agent de cliente real: a Cloudflare recusa o padrão do urllib com erro 1010
    h = {"content-type": "application/json", "user-agent": "curl/8.7.1"}
    if chave:
        h["authorization"] = "Bearer " + chave
    for i in range(tentativas):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(url, corpo, h), timeout=300))
        except (urllib.error.URLError, TimeoutError):
            if i == tentativas - 1:
                raise
            time.sleep(3 * (i + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    chave = os.environ.get("CHAVE", "")
    rel = {"modelo": a.modelo, "url": a.url, "n": a.n, "conjuntos": {}}
    vitorias = 0
    for cj, (exs, resps) in conjuntos(a.n).items():
        M, lat, falhas = [], [], 0
        for e in exs:
            t = time.time()
            try:
                r = decide(a.url, a.modelo, e, chave)
                M.append(para_itens(e, r))
            except Exception:
                falhas += 1
                M.append(None)
            lat.append(time.time() - t)
        pares = [(m, j) for m, j in zip(M, [para_itens(e, r) for e, r in zip(exs, resps)]) if m is not None]
        if not pares:  # servidor fora do ar: registra a falha e segue, em vez de quebrar nas métricas vazias
            rel["conjuntos"][cj] = {"falhas": falhas, "pedidos": len(exs)}
            print(a.modelo, cj, "sem respostas | falhas", falhas, flush=True)
            continue
        Mf = [i for m, _ in pares for i in m]
        Jf = [i for _, j in pares for i in j]
        ia, ib = ic(Mf, Jf, acc), ic(Jf, Mf, brier)
        vence = ia[0] > 0 and ib[0] > 0
        vitorias += cj in HUMANOS and vence
        rel["conjuntos"][cj] = {"modelo": metricas(Mf), "jev": metricas(Jf), "ic95_acc": ia, "ic95_brier": ib,
                                "vence": vence, "falhas": falhas, "pedidos": len(exs),
                                "latencia_p50_ms": round(1000 * float(np.median(lat)), 1)}
        c = rel["conjuntos"][cj]
        print(a.modelo, cj, "acc", c["modelo"]["acc"], "× jev", c["jev"]["acc"], "| brier", c["modelo"]["brier"],
              "×", c["jev"]["brier"], "| vence", vence, "| p50", c["latencia_p50_ms"], "ms | falhas", falhas, flush=True)
    rel["vitorias_humanas"] = vitorias
    Path("execucoes/remoto").mkdir(parents=True, exist_ok=True)
    Path(f"execucoes/remoto/{a.modelo}.json").write_text(json.dumps(rel, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
