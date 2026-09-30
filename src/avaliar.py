"""Ablação das camadas estocásticas + calibração, em dados sintéticos e externos (rótulo humano).

Modos (cada um acrescenta uma camada):
  base          Qwen3 sem treino (zero-shot, mesma leitura por logits)
  ft            adaptadores de menor NLL de validação, determinístico
  swa           média de pesos SWA (Izmailov)
  ft+perm       T passadas com opções permutadas (remove viés de posição)
  mc+perm       T passadas com MC Dropout ligado (Gal) + permutação
  swag+perm     T amostras de pesos SWAG-diagonal + permutação
Depois, para cada modo: escala de temperatura por tipo e predição conformal, ambas ajustadas SÓ na
validação sintética e aplicadas no teste e nos externos.

Uso: python src/avaliar.py --execucao execucoes/v1 --amostras 6
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm import load

sys.path.insert(0, str(Path(__file__).parent))
from conformal import calibra, conjunto_aps  # noqa: E402
from dados import carrega, indice_ouro, metricas  # noqa: E402
from decisor import (Codificador, dist_restrita, mascara_blocos, para_canonica, passa,  # noqa: E402
                     perm_aleatoria)
from treinar import prepara_lora  # noqa: E402


def le_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def coleta(model, cod, exs, T, permutar, dropout, swag, max_prefixo, semente=0):
    """Devolve itens {tipo, p, y, fonte} com p = média das T amostras (ordem canônica)."""
    rnd = random.Random(semente)
    itens = []
    for ex in exs:
        acum = {}
        for t in range(T):
            if swag is not None:
                media, desvio = swag
                model.update(tree_unflatten([(k, media[k] + desvio[k] * mx.random.normal(media[k].shape))
                                             for k in media]))
            if dropout:
                model.train()
            else:
                model.eval()
            perms = ({k: perm_aleatoria(q, rnd) for k, q in ex["questions"].items()}
                     if permutar and t > 0 else None)
            ids, n_pre, segs, leit, blocos = cod.codifica(ex["state"], ex["questions"], perms, max_prefixo)
            logits = passa(model, mx.array([ids]), mascara_blocos(len(ids), n_pre, segs), leit)[0]
            for i, b in enumerate(blocos):
                p = mx.exp(para_canonica(dist_restrita(logits[i], b), b, ex["questions"][b.chave]))
                acum.setdefault(b.chave, []).append(p)
        model.eval()
        for k, ps in acum.items():
            P = mx.stack(ps)
            mx.eval(P)
            q = ex["questions"][k]
            media = P.mean(axis=0)
            tot = -float((media * mx.log(mx.maximum(media, 1e-12))).sum())
            alea = -float((P * mx.log(mx.maximum(P, 1e-12))).sum(axis=-1).mean())
            itens.append({"tipo": q["type"], "p": media.tolist(), "y": indice_ouro(q, ex["rotulos"][k]["ouro"]),
                          "fonte": ex.get("fonte", "sintetico"), "epist": max(0.0, tot - alea)})
    if swag is not None:
        model.update(tree_unflatten(list(swag[0].items())))
    return itens


def aplica_temp(p, T):
    lp = np.log(np.maximum(np.asarray(p), 1e-12)) / T
    lp -= lp.max()
    e = np.exp(lp)
    return (e / e.sum()).tolist()


def ajusta_temp(itens):
    """Temperatura por tipo que minimiza NLL na validação (busca em grade; 1 parâmetro por tipo)."""
    temps = {}
    for tipo in ("choice", "score", "noul"):
        sub = [i for i in itens if i["tipo"] == tipo]
        if not sub:
            temps[tipo] = 1.0
            continue
        grade = np.exp(np.linspace(math.log(0.3), math.log(4.0), 60))
        nll = [np.mean([-math.log(max(aplica_temp(i["p"], T)[i["y"]], 1e-9)) for i in sub]) for T in grade]
        temps[tipo] = float(grade[int(np.argmin(nll))])
    return temps


def com_temp(itens, temps):
    return [{**i, "p": aplica_temp(i["p"], temps[i["tipo"]])} for i in itens]


def resumo_conformal(val, teste, alfa=0.1):
    out = {}
    for tipo in ("choice", "score", "noul"):
        v = [i for i in val if i["tipo"] == tipo]
        t = [i for i in teste if i["tipo"] == tipo]
        if len(v) < 20 or not t:
            continue
        qhat = calibra([i["p"] for i in v], [i["y"] for i in v], alfa)
        conj = [conjunto_aps(i["p"], qhat) for i in t]
        out[tipo] = {"qhat": round(qhat, 4),
                     "cobertura": round(float(np.mean([i["y"] in c for i, c in zip(t, conj)])), 4),
                     "tamanho_medio": round(float(np.mean([len(c) for c in conj])), 3),
                     "decide_sozinho": round(float(np.mean([len(c) == 1 for c in conj])), 4)}
    return out


def main():
    mx.set_cache_limit(1024 ** 3)  # sem teto o cache do MLX cresce a cada comprimento novo e leva a swap
    ap = argparse.ArgumentParser()
    ap.add_argument("--execucao", default="execucoes/v1")
    ap.add_argument("--amostras", type=int, default=6)
    ap.add_argument("--n-val", type=int, default=250)
    ap.add_argument("--n-teste", type=int, default=250)
    ap.add_argument("--n-ext", type=int, default=250)
    ap.add_argument("--modos", default="base,ft,swa,ft+perm,mc+perm,swag+perm")
    ap.add_argument("--swag-escala", type=float, default=0.5)
    ap.add_argument("--modo-servidor", default="ft", help="modo cuja calibração vai para calibracao.json")
    a = ap.parse_args()
    ex_dir = Path(a.execucao)
    cfg = json.loads((ex_dir / "config.json").read_text())
    div = carrega(cfg["dados"])
    conjuntos = {"val": div["val"][:a.n_val], "teste": div["teste"][:a.n_teste],
                 "assin2": le_jsonl("dados/ext_assin2.jsonl")[:a.n_ext],
                 "b2w": le_jsonl("dados/ext_b2w.jsonl")[:a.n_ext]}
    model, tok = load(cfg["modelo"])
    cod = Codificador(tok)
    mp = cfg["max_prefixo"]
    resultados = {}
    lora_pronto = False
    for modo in a.modos.split(","):
        if modo != "base" and not lora_pronto:
            prepara_lora(model, cfg["rank"], cfg["dropout"], cfg["camadas"])
            lora_pronto = True
        swag = None
        if modo in ("ft", "ft+perm", "mc+perm"):
            model.load_weights(str(ex_dir / "adaptadores.safetensors"), strict=False)
        elif modo in ("swa", "swag+perm"):
            media = dict(mx.load(str(ex_dir / "swa_media.safetensors")))
            model.update(tree_unflatten(list(media.items())))
            if modo == "swag+perm":
                var = mx.load(str(ex_dir / "swag_var.safetensors"))
                swag = (media, {k: a.swag_escala * mx.sqrt(v) for k, v in var.items()})
        T = a.amostras if "+" in modo else 1
        itens = {nome: coleta(model, cod, exs, T, "perm" in modo, modo.startswith("mc"), swag, mp)
                 for nome, exs in conjuntos.items()}
        temps = ajusta_temp(itens["val"])
        val_cal = com_temp(itens["val"], temps)
        qhats = {t: calibra([i["p"] for i in val_cal if i["tipo"] == t], [i["y"] for i in val_cal if i["tipo"] == t])
                 for t in ("choice", "score", "noul") if sum(i["tipo"] == t for i in val_cal) >= 20}
        r = {"temperaturas": temps, "qhat": qhats}
        if modo == a.modo_servidor:
            (ex_dir / "calibracao.json").write_text(json.dumps({"modo": modo, "temperaturas": temps,
                                                                 "conformal": qhats}, indent=1))
        for nome in ("teste", "assin2", "b2w"):
            cru, cal = itens[nome], com_temp(itens[nome], temps)
            r[nome] = {"cru": metricas(cru), "calibrado": metricas(cal),
                       "conformal_90": resumo_conformal(com_temp(itens["val"], temps), cal)}
            for tipo in ("choice", "score", "noul"):
                sub = [i for i in cal if i["tipo"] == tipo]
                if sub:
                    r[nome][f"calibrado_{tipo}"] = metricas(sub)
        resultados[modo] = r
        print(f"\n=== {modo} (T={T}) temps={ {k: round(v, 2) for k, v in temps.items()} }", flush=True)
        for nome in ("teste", "assin2", "b2w"):
            print(f"  {nome:7s} cru {r[nome]['cru']}\n          cal {r[nome]['calibrado']}", flush=True)
        (ex_dir / "avaliacao.json").write_text(json.dumps(resultados, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
