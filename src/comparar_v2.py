"""Comparação pareada Brier × Jev com critério de vitória FIXADO ANTES de medir (v2).

Critério: o Brier só é declarado "melhor que o Jev" se, em pelo menos 2 dos 3 conjuntos com
rótulo humano (ASSIN2, B2W, tweetSentBR), o IC de 95% (bootstrap pareado) da diferença for
inteiramente favorável ao Brier em ACURÁCIA E em ESCORE DE BRIER ao mesmo tempo.
O sintético é reportado, mas não conta para o critério (o gerador é da mesma família do treino).

No sintético, os rótulos são os da MAIORIA v2 (4 votos), nos mesmos 250 exemplos de teste do v1;
as respostas do Jev vêm do cache (mesmo state e perguntas do v1), filtradas para as perguntas
que tiveram maioria.

Uso: python src/comparar_v2.py --execucoes execucoes/v1 execucoes/v2
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

HUMANOS = ("assin2", "b2w", "tweetsentbr")


def h(ex):
    return hashlib.sha1(json.dumps([ex["state"], ex["questions"]], sort_keys=True,
                                   ensure_ascii=False).encode()).hexdigest()


def le(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def acc(it):
    return float(np.mean([np.argmax(i["p"]) == i["y"] for i in it]))


def brier(it):
    return float(np.mean([sum((pk - (k == i["y"])) ** 2 for k, pk in enumerate(i["p"])) for i in it]))


def ic(a, b, fn, n=2000, semente=0):
    rnd = np.random.default_rng(semente)
    d = []
    for _ in range(n):
        s = rnd.integers(0, len(a), len(a))
        d.append(fn([a[i] for i in s]) - fn([b[i] for i in s]))
    return [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]


def conjuntos(n):
    cache = {}
    for l in Path("dados/jev_cache.jsonl").read_text().splitlines():
        x = json.loads(l)
        cache[x["h"]] = x["r"]
    v1_teste = carrega("dados/sintetico.jsonl")["teste"][:n]
    v2 = {e["id"]: e for e in le("dados/sintetico_v2.jsonl")}
    sint, jev_sint = [], []
    for e in v1_teste:
        if e["id"] in v2 and h(e) in cache:
            sint.append(v2[e["id"]])
            jev_sint.append(cache[h(e)])
    out = {"sintetico": (sint, jev_sint)}
    for nome, arq in (("assin2", "dados/ext_assin2.jsonl"), ("b2w", "dados/ext_b2w.jsonl"),
                      ("tweetsentbr", "dados/ext_tweetsentbr.jsonl")):
        exs = [e for e in le(arq)[:n] if h(e) in cache]
        out[nome] = (exs, [cache[h(e)] for e in exs])
    return out


def roda_local(execucao, dados):
    d = Path(execucao)
    cfg = json.loads((d / "config.json").read_text())
    model, tok = load(cfg["modelo"])
    prepara_lora(model, cfg["rank"], cfg["dropout"], cfg["camadas"])
    model.load_weights(str(d / "adaptadores.safetensors"), strict=False)
    cod = Codificador(tok)
    res = {}
    for nome, (exs, _) in dados.items():
        res[nome] = [coleta(model, cod, [e], 1, False, False, None, cfg["max_prefixo"]) for e in exs]
    del model
    return res


def main():
    mx.set_cache_limit(1024 ** 3)  # sem teto o cache do MLX cresce a cada comprimento novo e leva a swap
    ap = argparse.ArgumentParser()
    ap.add_argument("--execucoes", nargs="+", default=["execucoes/v1", "execucoes/v2"])
    ap.add_argument("--n", type=int, default=250)
    a = ap.parse_args()
    dados = conjuntos(a.n)
    jev = {nome: [] for nome in dados}
    for nome, (exs, resps) in dados.items():
        for e, r in zip(exs, resps):
            jev[nome].append(para_itens(e, r))  # para_itens usa só as perguntas presentes em e
    relatorio = {"criterio": "vence se, em >=2 de " + "/".join(HUMANOS) +
                 ", IC95% favorável ao Brier em acurácia E escore de Brier", "modelos": {}}
    for exe in a.execucoes:
        loc = roda_local(exe, dados)
        r = {}
        vitorias = 0
        for nome in dados:
            L = [i for por_ex in loc[nome] for i in por_ex]
            J = [i for por_ex in jev[nome] for i in por_ex]
            assert len(L) == len(J), (nome, len(L), len(J))
            ic_acc, ic_brier = ic(L, J, acc), ic(J, L, brier)  # Brier: positivo = Brier do Jev maior = local melhor
            vence = ic_acc[0] > 0 and ic_brier[0] > 0
            if nome in HUMANOS and vence:
                vitorias += 1
            r[nome] = {"brier_modelo": metricas(L), "jev": metricas(J),
                       "ic95_acc_brier_menos_jev": ic_acc, "ic95_escore_brier_jev_menos_local": ic_brier,
                       "vence": vence}
            print(exe, nome, "acc", r[nome]["brier_modelo"]["acc"], "×", r[nome]["jev"]["acc"],
                  "| brier", r[nome]["brier_modelo"]["brier"], "×", r[nome]["jev"]["brier"],
                  "| ICacc", ic_acc, "ICbrier", ic_brier, "| vence" if vence else "", flush=True)
        r["vitorias_humanas"] = vitorias
        r["melhor_que_jev"] = vitorias >= 2
        relatorio["modelos"][exe] = r
        print(exe, "vitórias em conjuntos humanos:", vitorias, "→ melhor que o Jev" if vitorias >= 2 else "→ ainda não", flush=True)
    Path("execucoes/comparacao_v2.json").write_text(json.dumps(relatorio, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
