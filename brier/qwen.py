"""Motor dos modelos treinados pelo Brier (v2, v3): Qwen3-4B + adaptadores LoRA, sozinhos ou em comitê.

- MLX em Mac com Apple Silicon (mais rápido); PyTorch nos demais sistemas (ver qwen_torch.py).
- Comitê: a mesma base com os adaptadores de vários membros. No MLX a base é carregada uma vez e os
  adaptadores são trocados em memória; a resposta é a média das probabilidades dos membros.
  Média de modelos tende a melhorar acerto e calibração; aqui junta o v2 (mais preciso) e o v3
  (mais resistente a injeção).
- Cada membro é uma pasta (ou repositório do Hugging Face) com config.json, adaptadores.safetensors e,
  opcionalmente, calibracao.json (conjunto conformal).
"""
from __future__ import annotations

import json
import math
import platform
import sys
from pathlib import Path

import numpy as np

from .nucleo import Codificador, avisos, formata, indices_canonicos, mascara_blocos_np

ESCALA_LORA = 20.0  # a mesma do treino


def _pasta(modelo: str) -> Path:
    p = Path(modelo)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(modelo, allow_patterns=["*.json", "*.safetensors", "*.md"]))


def mlx_disponivel() -> bool:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    try:
        import mlx_lm  # noqa: F401
        return True
    except ImportError:
        return False


class _MembroMLX:
    """Inferência MLX com máscara de atenção em blocos e leitura por logits (mesma do treino)."""

    def __init__(self, cfg: dict):
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.tuner.utils import linear_to_lora_layers
        mx.set_cache_limit(1024 ** 3)  # sem teto o cache cresce a cada comprimento novo e leva a swap
        self.mx = mx
        self.model, self.tok = load(cfg["modelo"])
        self.model.freeze()
        linear_to_lora_layers(self.model, cfg["camadas"], {"rank": cfg["rank"], "scale": ESCALA_LORA,
                                                           "dropout": cfg.get("dropout", 0.0)})
        self.model.eval()
        self.base = cfg["modelo"]

    def carrega_adaptadores(self, arquivo: Path) -> dict:
        return dict(self.mx.load(str(arquivo)))

    def usa(self, pesos: dict):
        from mlx.utils import tree_unflatten
        self.model.update(tree_unflatten(list(pesos.items())))

    def probs(self, ids, n_pre, segs, leituras, blocos, questions) -> list[np.ndarray]:
        mx = self.mx
        ok = mx.array(mascara_blocos_np(len(ids), n_pre, segs))
        inner = self.model.model
        h = inner.embed_tokens(mx.array([ids]))
        m = mx.where(ok, 0.0, -math.inf).astype(h.dtype)
        for layer in inner.layers:
            h = layer(h, m, None)
        h = inner.norm(h[:, mx.array(leituras), :])
        lg = inner.embed_tokens.as_linear(h)[0] if self.model.args.tie_word_embeddings else self.model.lm_head(h)[0]
        out = []
        for i, b in enumerate(blocos):
            sel = lg[i, mx.array(b.ids)].astype(mx.float32)
            p = mx.softmax(sel, axis=-1)[mx.array(indices_canonicos(b, questions[b.chave]))]
            out.append(np.array(p))
        return out


class MotorQwen:
    """Um ou mais membros (pastas/repositórios de adaptadores) sobre a mesma base Qwen3."""

    def __init__(self, membros: list[str] | str, *, dispositivo: str | None = None, max_state: int = 4096,
                 motor: str = "auto"):
        membros = [membros] if isinstance(membros, str) else list(membros)
        self.pastas = [_pasta(m) for m in membros]
        self.nomes = membros
        cfgs = [json.loads((p / "config.json").read_text(encoding="utf-8")) for p in self.pastas]
        if len({c["modelo"] for c in cfgs}) != 1:
            raise ValueError("todos os membros do comitê precisam da mesma base")
        self.max_state = max_state
        self.motor = ("mlx" if mlx_disponivel() else "torch") if motor == "auto" else motor
        if self.motor == "mlx":
            self._mlx = _MembroMLX(cfgs[0])
            self._pesos = [self._mlx.carrega_adaptadores(p / "adaptadores.safetensors") for p in self.pastas]
            self.cod = Codificador(self._mlx.tok)
        else:  # PyTorch: um modelo fundido por membro (memória × número de membros)
            from .qwen_torch import DecisorTorch
            self._torch = [DecisorTorch(p, dispositivo=dispositivo, max_state=max_state) for p in self.pastas]
            self.cod = self._torch[0].cod
        cal = self.pastas[-1] / "calibracao.json"  # o último membro define o conjunto conformal
        self.conformal = json.loads(cal.read_text(encoding="utf-8")).get("conformal") if cal.exists() else None

    @property
    def dispositivo(self) -> str:
        return "mlx" if self.motor == "mlx" else self._torch[0].dispositivo

    def _probs_membros(self, state, questions) -> dict:
        """{chave: [P_membro1, P_membro2, ...]} na ordem canônica das opções."""
        ids, n_pre, segs, leit, blocos = self.cod.codifica(state, questions, None, self.max_state)
        por = {b.chave: [] for b in blocos}
        if self.motor == "mlx":
            for pesos in self._pesos:
                self._mlx.usa(pesos)
                for b, p in zip(blocos, self._mlx.probs(ids, n_pre, segs, leit, blocos, questions)):
                    por[b.chave].append(p)
        else:
            import torch
            for d in self._torch:
                with torch.no_grad():
                    lg = d._passa(ids, n_pre, segs, leit)
                for i, b in enumerate(blocos):
                    p = torch.softmax(lg[i, b.ids], -1)[indices_canonicos(b, questions[b.chave])]
                    por[b.chave].append(p.cpu().numpy())
        return por

    def predict(self, state, questions: dict, min_confidence: float | None = None) -> dict:
        por = self._probs_membros(state, questions)
        # cada membro entra como uma "amostra": a média dá a probabilidade, e a divergência entre
        # membros aparece como incerteza epistêmica
        r = {"answers": {k: formata(questions[k], np.stack(v), self.conformal) for k, v in por.items()}}
        if len(self.nomes) > 1:
            r["membros"] = {k: {n: [round(float(x), 4) for x in p] for n, p in zip(self.nomes, v)} for k, v in por.items()}
        if min_confidence is not None:
            for a in r["answers"].values():
                topo = a["noul"] if a["type"] == "noul" else max(a["probabilities"].values())
                if a["type"] == "noul":
                    topo = max(topo, 1 - topo)
                a["low_confidence"] = bool(topo < min_confidence)
        if avisos(self.cod):
            r["avisos"] = avisos(self.cod)
        r["usage"] = {"truncated": bool(getattr(self.cod, "truncamento", None))}
        return r
