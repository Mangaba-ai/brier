"""Leitura do sintético, divisão por TAREFA (sem vazamento entre treino/val/teste) e métricas."""
import json
import zlib
import math
from pathlib import Path

import numpy as np

from decisor import ordem_canonica


def _balde(id_: str) -> int:
    """0 = teste, 1 = val, resto = treino. Sintético: semente da tarefa (tarefas inteiras ficam juntas);
    públicos: crc32 do id."""
    cab = id_.split("-")[0]
    return int(cab) % 10 if cab.isdigit() else zlib.crc32(id_.encode()) % 10


def carrega(caminho="dados/sintetico.jsonl"):
    caminhos = caminho if isinstance(caminho, (list, tuple)) else [caminho]
    exs = [json.loads(l) for c in caminhos for l in Path(c).read_text(encoding="utf-8").splitlines() if l.strip()]
    divisao = {"treino": [], "val": [], "teste": []}
    for e in exs:
        s = _balde(e["id"])
        divisao["teste" if s == 0 else "val" if s == 1 else "treino"].append(e)
    return divisao


def indice_ouro(q, ouro) -> int:
    return ordem_canonica(q).index(ouro)


def metricas(itens):
    """itens: dicts com tipo, p (lista canônica), y (índice ouro). Devolve métricas agregadas."""
    if not itens:
        return {}
    acc = np.mean([int(np.argmax(i["p"]) == i["y"]) for i in itens])
    nll = np.mean([-math.log(max(i["p"][i["y"]], 1e-9)) for i in itens])
    brier = np.mean([sum((pk - (k == i["y"])) ** 2 for k, pk in enumerate(i["p"])) for i in itens])
    conf = np.array([max(i["p"]) for i in itens])
    certo = np.array([int(np.argmax(i["p"]) == i["y"]) for i in itens])
    ece, bordas = 0.0, np.linspace(0, 1, 11)
    for a, b in zip(bordas[:-1], bordas[1:]):
        m = (conf > a) & (conf <= b)
        if m.any():
            ece += m.mean() * abs(conf[m].mean() - certo[m].mean())
    out = {"n": len(itens), "acc": round(float(acc), 4), "nll": round(float(nll), 4),
           "brier": round(float(brier), 4), "ece": round(float(ece), 4)}
    sc = [i for i in itens if i["tipo"] == "score"]
    if sc:
        out["mae_score"] = round(float(np.mean([abs(sum(k * pk for k, pk in enumerate(i["p"])) - i["y"])
                                                for i in sc])), 4)
    return out
