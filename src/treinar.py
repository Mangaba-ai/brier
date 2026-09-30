"""Treino do decisor: LoRA (com dropout) + perda de regra de pontuação própria + SWA/SWAG.

Por que isso é o equivalente do RLCD do Jev: o RLCD recompensa probabilidades que batem com a
realidade. Com leitura por logits, a política É a distribuição sobre as opções, então a recompensa
esperada de uma regra de pontuação própria (log ou Brier) tem gradiente exato — não precisa amostrar
nem de REINFORCE. Minimizar entropia cruzada contra o alvo (firme quando os rotuladores concordam,
dividido quando discordam) é otimizar essa recompensa diretamente. O termo ordinal pune errar a
escala por muitos níveis mais do que por um.

Uso: python src/treinar.py --modelo Qwen/Qwen3-1.7B --passos 6000 --saida execucoes/v1
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from mlx_lm import load
from mlx_lm.tuner.trainer import grad_checkpoint
from mlx_lm.tuner.utils import linear_to_lora_layers

sys.path.insert(0, str(Path(__file__).parent))
from dados import carrega, indice_ouro, metricas  # noqa: E402
from decisor import (Codificador, dist_restrita, mascara_blocos, para_canonica, passa,  # noqa: E402
                     perm_aleatoria)


def prepara_lora(model, rank, dropout, camadas):
    model.freeze()
    linear_to_lora_layers(model, camadas, {"rank": rank, "scale": 20.0, "dropout": dropout,
                                           })
    n = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"parâmetros treináveis: {n / 1e6:.2f} M", flush=True)


def perda_exemplo(model, cod, ex, rnd, max_prefixo, peso_ordinal=0.5):
    qs = list(ex["questions"].items())
    rnd.shuffle(qs)
    questions = dict(qs)
    perms = {k: perm_aleatoria(q, rnd) for k, q in questions.items()}
    ids, n_pre, segs, leit, blocos = cod.codifica(ex["state"], questions, perms, max_prefixo)
    mask = mascara_blocos(len(ids), n_pre, segs)
    alvos = [mx.array(ex["rotulos"][b.chave]["alvo"], dtype=mx.float32) for b in blocos]

    def f(m):
        logits = passa(m, mx.array([ids]), mask, leit)[0]
        tot = 0.0
        for i, b in enumerate(blocos):
            q = questions[b.chave]
            lp = para_canonica(dist_restrita(logits[i], b), b, q)
            tot = tot - (alvos[i] * lp).sum()
            if b.tipo == "score":
                K = len(q["criteria"])
                niveis = mx.arange(K, dtype=mx.float32)
                esp = (mx.exp(lp) * niveis).sum()
                y = float(ex["rotulos"][b.chave]["ouro"])
                tot = tot + peso_ordinal * ((esp - y) / max(K - 1, 1)) ** 2
        return tot / len(blocos)

    return f


def avalia(model, cod, exs, max_prefixo, limite=250):
    model.eval()
    itens = []
    for ex in exs[:limite]:
        ids, n_pre, segs, leit, blocos = cod.codifica(ex["state"], ex["questions"], None, max_prefixo)
        logits = passa(model, mx.array([ids]), mascara_blocos(len(ids), n_pre, segs), leit)[0]
        for i, b in enumerate(blocos):
            q = ex["questions"][b.chave]
            p = mx.exp(para_canonica(dist_restrita(logits[i], b), b, q))
            mx.eval(p)
            itens.append({"tipo": b.tipo, "p": p.tolist(), "y": indice_ouro(q, ex["rotulos"][b.chave]["ouro"])})
    return metricas(itens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--dados", nargs="+", default=["dados/sintetico.jsonl"],
                    help="sintéticos (o 1º define val/teste de referência)")
    ap.add_argument("--publicos", default=None, help="jsonl de conjuntos públicos (só treino)")
    ap.add_argument("--peso-publicos", type=float, default=0.3, help="fração dos exemplos vindos dos públicos")
    ap.add_argument("--pular-base", action="store_true", help="não avaliar o modelo base antes do treino")
    ap.add_argument("--saida", default="execucoes/v1")
    ap.add_argument("--passos", type=int, default=6000)
    ap.add_argument("--acumula", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--camadas", type=int, default=28)
    ap.add_argument("--max-prefixo", type=int, default=1024)
    ap.add_argument("--avaliar-cada", type=int, default=500)
    ap.add_argument("--swa-inicio", type=float, default=0.7, help="fração dos passos em que começa o SWA")
    ap.add_argument("--swa-cada", type=int, default=25)
    a = ap.parse_args()
    # sem teto, o cache de buffers do MLX cresce com cada comprimento de sequência novo e empurra o
    # sistema para swap (o v2 travou assim na 1ª tentativa). 2 GB de cache bastam.
    mx.set_cache_limit(2 * 1024 ** 3)
    saida = Path(a.saida)
    saida.mkdir(parents=True, exist_ok=True)
    (saida / "config.json").write_text(json.dumps(vars(a), indent=1))

    div = carrega(a.dados)
    publicos = carrega(a.publicos)["treino"] if a.publicos else []
    print({k: len(v) for k, v in div.items()}, flush=True)
    model, tok = load(a.modelo)
    cod = Codificador(tok)
    if not a.pular_base:
        print("base (zero-shot) val:", avalia(model, cod, div["val"], a.max_prefixo, 200), flush=True)
    prepara_lora(model, a.rank, a.dropout, a.camadas)
    # máscara própria desliga o kernel de atenção eficiente: sem checkpoint, o backward guardaria
    # uma matriz L×L por camada (≈16 GB com 3k tokens). Recalcular camada a camada cabe em 16 GB.
    grad_checkpoint(model.model.layers[0])

    aquece = min(100, max(1, a.passos // 10))
    sched = optim.join_schedules([optim.linear_schedule(1e-7, a.lr, aquece),
                                  optim.cosine_decay(a.lr, a.passos - aquece, a.lr * 0.1)], [aquece])
    opt = optim.AdamW(learning_rate=sched, weight_decay=0.01)
    rnd = random.Random(0)
    treino = div["treino"]
    melhor, t0, hist = math.inf, time.time(), []
    swa_n, swa_m, swa_m2 = 0, None, None
    for passo in range(1, a.passos + 1):
        model.train()
        grads_acc, perda_acc = None, 0.0
        for _ in range(a.acumula):
            ex = rnd.choice(publicos) if publicos and rnd.random() < a.peso_publicos else rnd.choice(treino)
            f = perda_exemplo(model, cod, ex, rnd, a.max_prefixo)
            perda, g = nn.value_and_grad(model, f)(model)
            grads_acc = g if grads_acc is None else tree_map(mx.add, grads_acc, g)
            mx.eval(grads_acc)  # libera o grafo deste micro-passo antes do próximo
            perda_acc += perda.item()
        grads_acc = tree_map(lambda x: x / a.acumula, grads_acc)
        opt.update(model, grads_acc)
        mx.eval(model.parameters(), opt.state)
        hist.append(perda_acc / a.acumula)

        if passo >= a.swa_inicio * a.passos and passo % a.swa_cada == 0:
            p = {k: np.array(v.astype(mx.float32)) for k, v in tree_flatten(model.trainable_parameters())}
            if swa_m is None:
                swa_m = {k: v.copy() for k, v in p.items()}
                swa_m2 = {k: v ** 2 for k, v in p.items()}
            else:
                for k in p:
                    swa_m[k] += (p[k] - swa_m[k]) / (swa_n + 1)
                    swa_m2[k] += (p[k] ** 2 - swa_m2[k]) / (swa_n + 1)
            swa_n += 1

        if passo % 10 == 0:
            print(f"passo {passo} perda {np.mean(hist[-10:]):.4f} {time.time() - t0:.0f}s", flush=True)
        if passo % a.avaliar_cada == 0 or passo == a.passos:
            m = avalia(model, cod, div["val"], a.max_prefixo)
            print(f"val passo {passo}: {m}", flush=True)
            if m["nll"] < melhor:
                melhor = m["nll"]
                mx.save_safetensors(str(saida / "adaptadores.safetensors"), dict(tree_flatten(model.trainable_parameters())))
                print("  ↳ melhor até agora, salvo", flush=True)

    if swa_m is not None:
        mx.save_safetensors(str(saida / "swa_media.safetensors"), {k: mx.array(v) for k, v in swa_m.items()})
        var = {k: mx.array(np.maximum(swa_m2[k] - swa_m[k] ** 2, 1e-12)) for k in swa_m}
        mx.save_safetensors(str(saida / "swag_var.safetensors"), var)
        model.update(tree_unflatten([(k, mx.array(v)) for k, v in swa_m.items()]))
        print(f"SWA ({swa_n} instantâneos) val: {avalia(model, cod, div['val'], a.max_prefixo)}", flush=True)
    (saida / "fim.json").write_text(json.dumps({"melhor_nll_val": melhor, "swa_n": swa_n}))


if __name__ == "__main__":
    main()
