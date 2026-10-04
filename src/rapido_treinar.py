"""Brier Rápido: ajuste fino do Laya multilíngue (mmBERT, 322M) com os dados do Brier em português.

Parte dos pesos abertos do Laya (convaiinnovations/laya-multilingual, Apache-2.0) e usa o mesmo laço
de treino publicado por eles (notebooks/laya_finetune_typed_decisions_mps.py, Apache-2.0): RL contra
regras de pontuação próprias (log + esférica + RPS) mais entropia cruzada contra o alvo suave.
O que muda é o dado: os rótulos por maioria de votos do Brier (inclusive do Titan), os exemplos
focados nas armadilhas, conjuntos públicos em PT-BR e exemplos com injeção de instrução.
Só a divisão de TREINO entra; validação e teste ficam intactos.

Uso: python src/rapido_treinar.py --epocas 2 --saida execucoes/rapido
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from laya.agent import _fix_tokenizer_config
from laya.common import QTYPES, build_model, build_sequence, proper_reward, render_options

sys.path.insert(0, str(Path(__file__).parent))
from dados import carrega  # noqa: E402

BASE = "convaiinnovations/laya-multilingual"


def item_laya(tok, cfg, ex, k):
    """Uma pergunta do Brier vira um item de treino do Laya (alvo na ordem de opções do Laya)."""
    q, rot = ex["questions"][k], ex["rotulos"][k]
    alvo = list(rot["alvo"])
    if q["type"] == "noul":          # Brier: [true, false]; Laya: [false, true]
        alvo = alvo[::-1]
    s = sum(alvo)
    alvo = [x / s for x in alvo] if s > 0 else [1 / len(alvo)] * len(alvo)
    qi = {"t": q["type"], "ins": q["instructions"], "crit": q["criteria"]}
    seq, marcadores = build_sequence(tok, ex["state"], qi, cfg["max_len"], cfg["head_max_len"])
    if len(marcadores) != len(render_options(qi)) or len(marcadores) != len(alvo):
        return None
    return {"ids": seq, "markers": marcadores, "qtype": QTYPES[q["type"]], "target": alvo}


def collate(items, pad_id):
    b, L, K = len(items), max(len(i["ids"]) for i in items), max(len(i["markers"]) for i in items)
    ids = torch.full((b, L), pad_id, dtype=torch.long)
    att = torch.zeros((b, L), dtype=torch.long)
    pos = torch.zeros((b, K), dtype=torch.long)
    msk = torch.zeros((b, K), dtype=torch.bool)
    alv = torch.zeros((b, K), dtype=torch.float32)
    for i, it in enumerate(items):
        n, k = len(it["ids"]), len(it["markers"])
        ids[i, :n] = torch.tensor(it["ids"])
        att[i, :n] = 1
        pos[i, :k] = torch.tensor(it["markers"])
        msk[i, :k] = True
        alv[i, :k] = torch.tensor(it["target"])
    return ids, att, pos, msk, alv, torch.tensor([it["qtype"] for it in items])


def monta_itens(tok, cfg, n_publicos, n_injecao, semente=7):
    rnd = random.Random(semente)
    sint = carrega(["dados/sintetico_v4.jsonl", "dados/sintetico_foco_v2.jsonl"])["treino"]
    pubs = carrega("dados/publicos_treino.jsonl")["treino"]
    inj = [e for e in (json.loads(l) for l in open("dados/adv_treino.jsonl", encoding="utf-8")) if "ataque" in e]
    exs = sint + rnd.sample(pubs, min(n_publicos, len(pubs))) + rnd.sample(inj, min(n_injecao, len(inj)))
    itens, pulados = [], 0
    for e in exs:
        for k in e["questions"]:
            it = item_laya(tok, cfg, e, k)
            if it is None:
                pulados += 1
            else:
                itens.append(it)
    rnd.shuffle(itens)
    return itens, pulados


def salva(model, tok, cfg, pasta, temperaturas=None):
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    save_file({n: v.detach().half().cpu().contiguous() for n, v in model.state_dict().items()},
              str(pasta / "model.safetensors"))
    model.encoder.config.save_pretrained(pasta / "encoder")
    tok.save_pretrained(pasta / "tokenizer")
    c = dict(cfg)
    if temperaturas:
        c["temperature"] = temperaturas
    c.update({"model_name": "brier-rapido", "fine_tuned": True, "base": BASE})
    c.pop("temperature_by_options", None)
    (pasta / "rl_agent_config.json").write_text(json.dumps(c, indent=2), encoding="utf-8")


def ajusta_temperatura(amostras):
    if len(amostras) < 10:
        return 1.0
    K = max(len(l) for l, _ in amostras)
    lg = torch.full((len(amostras), K), -1e4)
    tg = torch.zeros((len(amostras), K))
    for i, (l, t) in enumerate(amostras):
        lg[i, :len(l)] = torch.as_tensor(l)
        tg[i, :len(t)] = torch.as_tensor(t)
    lt = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([lt], lr=0.1, max_iter=100)

    def fecho():
        opt.zero_grad()
        perda = -(tg * torch.log_softmax(lg / lt.exp(), -1)).sum(-1).mean()
        perda.backward()
        return perda
    opt.step(fecho)
    return float(torch.clamp(lt.exp(), 0.1, 10.0).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epocas", type=int, default=2)
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--acumula", type=int, default=8)
    ap.add_argument("--publicos", type=int, default=6000)
    ap.add_argument("--injecao", type=int, default=2000)
    ap.add_argument("--max-itens", type=int, default=0, help="limita o total (teste rápido)")
    ap.add_argument("--saida", default="execucoes/rapido")
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    base = snapshot_download(BASE)
    _fix_tokenizer_config(base)
    cfg = json.loads(Path(base, "rl_agent_config.json").read_text(encoding="utf-8"))
    cfg.update({"max_len": 1024, "head_max_len": 256})
    tok = AutoTokenizer.from_pretrained(Path(base) / "tokenizer")
    itens, pulados = monta_itens(tok, cfg, a.publicos, a.injecao)
    if a.max_itens:
        itens = itens[:a.max_itens]
    n_cal = min(400, len(itens) // 20)
    cal, treino = itens[:n_cal], itens[n_cal:]
    # agrupa por comprimento para reduzir padding (lotes embaralhados entre si)
    treino.sort(key=lambda i: len(i["ids"]))
    lotes = [treino[i:i + a.micro] for i in range(0, len(treino), a.micro)]
    print(f"itens: {len(treino)} treino + {len(cal)} calibração (pulados {pulados}); lotes {len(lotes)}; {dev}", flush=True)

    model = build_model(cfg, encoder_dir=Path(base) / "encoder")
    model.load_state_dict(load_file(str(Path(base) / "model.safetensors")), strict=True)
    model.float()
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    model.to(dev).train()
    enc = [p for n, p in model.named_parameters() if "encoder." in n]
    cab = [p for n, p in model.named_parameters() if "encoder." not in n]
    opt = torch.optim.AdamW([{"params": enc, "lr": 2e-5}, {"params": cab, "lr": 8e-5}], weight_decay=0.01)
    passos = max(1, math.ceil(len(lotes) / a.acumula) * a.epocas)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=passos, eta_min=1e-6)
    t0, n = time.time(), 0
    for ep in range(a.epocas):
        random.Random(100 + ep).shuffle(lotes)
        sigma = 0.4 + (0.1 - 0.4) * ep / max(1, a.epocas - 1)
        opt.zero_grad(set_to_none=True)
        soma = 0.0
        for j, lote in enumerate(lotes, 1):
            ids, att, pos, msk, alv, qt = (x.to(dev) for x in collate(lote, tok.pad_token_id))
            logits, act = model(ids, att, pos, msk, qt)
            logits = logits.float()
            k = msk.sum(-1, keepdim=True).float()
            eps = torch.randn((4,) + logits.shape, device=dev) * sigma * msk
            eps = (eps - eps.sum(-1, keepdim=True) / k) * msk
            ruido = logits.detach().unsqueeze(0) + eps
            prob = torch.softmax(ruido.masked_fill(~msk, -1e4), -1)
            with torch.no_grad():
                rec = proper_reward(prob, alv.unsqueeze(0), qt, msk, w_sph=0.75, w_rps=1.0)
                vant = (rec - rec.mean(0, keepdim=True))
                vant = vant / (vant.std() + 1e-6)
            logp = -(((ruido - logits.unsqueeze(0)) ** 2) * msk).sum(-1) / (2 * sigma ** 2)
            perda = (-(vant * logp).mean()
                     - (alv * torch.log_softmax(logits.masked_fill(~msk, -1e4), -1)).sum(-1).mean()
                     + 0.0 * act.sum()) / a.acumula
            perda.backward()
            soma += perda.item() * a.acumula
            if j % a.acumula == 0 or j == len(lotes):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            n += 1
            if j % 200 == 0:
                print(f"época {ep + 1} lote {j}/{len(lotes)} perda {soma / j:.4f} {time.time() - t0:.0f}s", flush=True)
        print(f"época {ep + 1} completa: perda média {soma / len(lotes):.4f}", flush=True)
        salva(model, tok, cfg, Path(a.saida) / "ultima_epoca")  # sobrescreve: disco apertado

    model.eval()
    amostras = [[] for _ in range(3)]
    with torch.no_grad():
        for i in range(0, len(cal), a.micro):
            ch = cal[i:i + a.micro]
            ids, att, pos, msk, alv, qt = (x.to(dev) for x in collate(ch, tok.pad_token_id))
            lg, _ = model(ids, att, pos, msk, qt)
            for r, it in enumerate(ch):
                amostras[it["qtype"]].append((lg[r, :len(it["markers"])].float().cpu(), it["target"]))
    temps = [ajusta_temperatura(g) if g else 1.0 for g in amostras]
    salva(model, tok, cfg, a.saida, temps)
    print(f"salvo em {a.saida}; temperaturas {temps}; {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
