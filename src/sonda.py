"""Sonda de ativações para detectar instruções injetadas no state (inspirada no monitoramento de
ativações internas descrito pelo Google no Gemini 4 Argon).

O Brier já calcula o estado oculto final na posição de leitura de cada pergunta. Uma regressão
logística sobre esse vetor (2.560 números no Qwen3-4B) estima a probabilidade de o state conter uma
instrução tentando manipular a decisão. Custo na inferência: um produto interno por pergunta.

Treino: pares do adversarial de treino (famílias A–F) e os mesmos exemplos sem injeção.
Teste: famílias G–J (nunca vistas) e os conjuntos humanos limpos (para medir alarme falso).
Saída: <execucao>/sonda_injecao.npz com pesos, normalização e limiar para 5% de alarme falso.

Uso: python src/sonda.py --execucao execucoes/v3
"""
import argparse
import json
import random
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load

sys.path.insert(0, str(Path(__file__).parent))
from dados import carrega  # noqa: E402
from decisor import Codificador, estado_leitura, mascara_blocos  # noqa: E402
from treinar import prepara_lora  # noqa: E402


def le(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def vetores(model, cod, exs, max_state=4096):
    """Um vetor por pergunta de cada exemplo (estado final na posição de leitura)."""
    out = []
    for e in exs:
        ids, n_pre, segs, leit, _ = cod.codifica(e["state"], e["questions"], None, max_state)
        hs = estado_leitura(model, mx.array([ids]), mascara_blocos(len(ids), n_pre, segs), leit)[0]
        out.append(np.array(hs.astype(mx.float32)))
    return out


def treina_logistica(X, y, l2=1e-2, passos=600, lr=0.05):
    """Regressão logística com Adam e L2 (numpy puro, sem dependência extra)."""
    w, b = np.zeros(X.shape[1]), 0.0
    m_w, v_w, m_b, v_b = np.zeros_like(w), np.zeros_like(w), 0.0, 0.0
    for t in range(1, passos + 1):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        gw = X.T @ (p - y) / len(y) + l2 * w
        gb = float(np.mean(p - y))
        m_w, v_w = 0.9 * m_w + 0.1 * gw, 0.999 * v_w + 0.001 * gw ** 2
        m_b, v_b = 0.9 * m_b + 0.1 * gb, 0.999 * v_b + 0.001 * gb ** 2
        w -= lr * (m_w / (1 - 0.9 ** t)) / (np.sqrt(v_w / (1 - 0.999 ** t)) + 1e-8)
        b -= lr * (m_b / (1 - 0.9 ** t)) / (np.sqrt(v_b / (1 - 0.999 ** t)) + 1e-8)
    return w, b


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    ordem = np.argsort(np.concatenate([pos, neg]))
    postos = np.empty(len(ordem))
    postos[ordem] = np.arange(1, len(ordem) + 1)
    return float((postos[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    mx.set_cache_limit(1024 ** 3)
    ap = argparse.ArgumentParser()
    ap.add_argument("--execucao", default="execucoes/v3")
    ap.add_argument("--n-treino", type=int, default=1200)
    a = ap.parse_args()
    d = Path(a.execucao)
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    model, tok = load(cfg["modelo"])
    prepara_lora(model, cfg["rank"], cfg["dropout"], cfg["camadas"])
    model.load_weights(str(d / "adaptadores.safetensors"), strict=False)
    model.eval()
    cod = Codificador(tok)
    rnd = random.Random(5)

    # treino: injetados A–F (só os de injeção, não os longos) + os limpos correspondentes
    sint = carrega(["dados/sintetico_v2.jsonl", "dados/sintetico_foco_v2.jsonl"])
    pubs = {e["id"]: e for e in carrega("dados/publicos_treino.jsonl")["treino"]}
    limpos_por_id = {e["id"]: e for e in sint["treino"]} | pubs
    inj_tr = [e for e in le("dados/adv_treino.jsonl") if "ataque" in e and e["ataque"]["id_limpo"] in limpos_por_id]
    inj_tr = rnd.sample(inj_tr, min(a.n_treino, len(inj_tr)))
    lim_tr = [limpos_por_id[e["ataque"]["id_limpo"]] for e in inj_tr]
    Xp, Xn = vetores(model, cod, inj_tr), vetores(model, cod, lim_tr)
    X = np.concatenate([np.concatenate(Xp), np.concatenate(Xn)])
    y = np.concatenate([np.ones(sum(len(v) for v in Xp)), np.zeros(sum(len(v) for v in Xn))])
    media, desvio = X.mean(0), X.std(0) + 1e-6
    w, b = treina_logistica((X - media) / desvio, y)

    def escore(vs):  # alerta por EXEMPLO = máximo entre as perguntas
        return [float(np.max(1 / (1 + np.exp(-(((v - media) / desvio) @ w + b))))) for v in vs]

    # teste: famílias G–J nunca vistas + os mesmos exemplos limpos
    teste = le("dados/adv_teste_injecao.jsonl")
    base = {}
    v2 = {e["id"]: e for e in le("dados/sintetico_v2.jsonl")}
    for e in carrega("dados/sintetico.jsonl")["teste"][:250]:
        if e["id"] in v2:
            base[e["id"]] = v2[e["id"]]
    for arq in ("dados/ext_assin2.jsonl", "dados/ext_b2w.jsonl", "dados/ext_tweetsentbr.jsonl"):
        base.update({e["id"]: e for e in le(arq)[:250]})
    s_pos = escore(vetores(model, cod, teste))
    s_neg = escore(vetores(model, cod, [base[e["ataque"]["id_limpo"]] for e in teste]))
    limiar = float(np.quantile(s_neg, 0.95))  # 5% de alarme falso nos limpos
    res = {"auc": round(auc(s_pos, s_neg), 4), "limiar_5pct_alarme_falso": round(limiar, 4),
           "deteccao_no_limiar": round(float(np.mean(np.array(s_pos) > limiar)), 4),
           "por_familia": {}}
    for f in sorted({e["ataque"]["familia"] for e in teste}):
        idx = [i for i, e in enumerate(teste) if e["ataque"]["familia"] == f]
        res["por_familia"][f] = round(float(np.mean([s_pos[i] > limiar for i in idx])), 4)
    print(json.dumps(res, ensure_ascii=False), flush=True)
    np.savez(d / "sonda_injecao.npz", w=w.astype(np.float32), b=np.float32(b), media=media.astype(np.float32),
             desvio=desvio.astype(np.float32), limiar=np.float32(limiar))
    (d / "sonda_injecao.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
