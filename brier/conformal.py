"""Predição conformal com conjuntos adaptativos (APS) — Romano, Sesia & Candès (2020);
Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction" (2021).

Com um conjunto de calibração trocável com os dados de uso, o conjunto devolvido contém a resposta
certa com probabilidade ≥ 1−α, SEM supor que o modelo seja calibrado. É a forma rigorosa do
"não sei": conjunto com 1 opção → decidir; com 2+ → pedir esclarecimento ou humano.
"""
import math

import numpy as np


def _escore_aps(p, y: int, u: float = 1.0) -> float:
    ordem = np.argsort(-np.asarray(p))
    acum = 0.0
    for i in ordem:
        if i == y:
            return acum + u * p[i]
        acum += p[i]
    return acum


def calibra(probs: list, ouros: list, alfa: float = 0.1, semente: int = 0) -> float:
    """q̂ = quantil ⌈(n+1)(1−α)⌉/n dos escores de não-conformidade (versão aleatorizada)."""
    rnd = np.random.default_rng(semente)
    s = [_escore_aps(p, y, rnd.uniform()) for p, y in zip(probs, ouros)]
    n = len(s)
    nivel = min(1.0, math.ceil((n + 1) * (1 - alfa)) / n)
    return float(np.quantile(s, nivel, method="higher"))


def conjunto_aps(p, qhat: float) -> list[int]:
    """Menor prefixo (em ordem de probabilidade) cuja massa acumulada alcança q̂."""
    ordem = np.argsort(-np.asarray(p))
    conj, acum = [], 0.0
    for i in ordem:
        conj.append(int(i))
        acum += p[i]
        if acum >= qhat:
            break
    return conj
