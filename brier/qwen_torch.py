"""Motor PyTorch do Brier: roda em Windows, Linux e macOS (CPU, NVIDIA/CUDA ou Apple/MPS).

Carrega os MESMOS adaptadores treinados no MLX e produz as mesmas respostas:
  1. Base: se o treino usou um checkpoint quantizado do mlx-community (ex.: Qwen3-4B 4 bits), os
     arquivos safetensors são lidos direto e dequantizados em memória (formato afim do MLX:
     w = escala · q + viés por grupo). Assim a base é idêntica à do treino e o download é o de 4 bits.
     Se o treino usou uma base comum do Hugging Face (ex.: Qwen/Qwen3-1.7B), ela é carregada normalmente.
  2. LoRA: cada par (lora_a [in, r], lora_b [r, out]) é fundido no peso: W += escala · (A·B)ᵀ.
  3. Inferência: máscara de atenção em blocos 4D e leitura dos logits só nos tokens-rótulo — o mesmo
     código de montagem e formatação do motor MLX (`nucleo.py`).

A camada de MC Dropout não existe aqui (os adaptadores estão fundidos); com `amostras > 1` o motor
PyTorch usa a permutação estocástica das opções, que foi a camada que mais ajudou na ablação.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .nucleo import (Codificador, aplica_sonda, avisos, carrega_sonda, formata, indices_canonicos,
                    mascara_blocos_np, perm_aleatoria)

ESCALA_LORA = 20.0  # a mesma de treinar.prepara_lora


def dispositivo_padrao() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dequantiza(w: torch.Tensor, escalas: torch.Tensor, vieses: torch.Tensor, bits: int, grupo: int) -> torch.Tensor:
    """Formato afim do MLX: inteiros empacotados em uint32, escala e viés por grupo de colunas."""
    por_palavra = 32 // bits
    w32 = w.view(torch.int32) if w.dtype != torch.int32 else w
    desloc = torch.arange(0, 32, bits, dtype=torch.int32)
    q = (w32.unsqueeze(-1) >> desloc) & ((1 << bits) - 1)          # [out, in/por_palavra, por_palavra]
    q = q.reshape(w.shape[0], w.shape[1] * por_palavra).to(torch.float32)
    esc = escalas.to(torch.float32).repeat_interleave(grupo, dim=-1)
    vie = vieses.to(torch.float32).repeat_interleave(grupo, dim=-1)
    return q * esc + vie


def _carrega_base_mlx_quantizada(repo: str, dtype: torch.dtype):
    pasta = Path(snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"]))
    cfg_bruta = json.loads((pasta / "config.json").read_text(encoding="utf-8"))
    quant = cfg_bruta.get("quantization") or cfg_bruta.get("quantization_config") or {}
    bits, grupo = int(quant.get("bits", 4)), int(quant.get("group_size", 64))
    tensores = {}
    for arq in sorted(pasta.glob("*.safetensors")):
        tensores.update(load_file(str(arq)))
    estado = {}
    for k, v in tensores.items():
        if k.endswith(".scales") or k.endswith(".biases"):
            continue
        base = k[: -len(".weight")] if k.endswith(".weight") else None
        if base and f"{base}.scales" in tensores:
            estado[k] = _dequantiza(v, tensores[f"{base}.scales"], tensores[f"{base}.biases"], bits, grupo).to(dtype)
        else:
            estado[k] = v.to(dtype) if v.is_floating_point() else v
    config = AutoConfig.from_pretrained(str(pasta))
    for campo in ("quantization", "quantization_config"):
        if hasattr(config, campo):
            delattr(config, campo)
    modelo = AutoModelForCausalLM.from_config(config, dtype=dtype)
    faltando, sobrando = modelo.load_state_dict(estado, strict=False)
    faltando = [m for m in faltando if not m.startswith("lm_head")]
    if faltando:
        raise RuntimeError(f"pesos faltando na base: {faltando[:5]}")
    if config.tie_word_embeddings:
        modelo.tie_weights()
    return modelo, AutoTokenizer.from_pretrained(str(pasta))


def _funde_lora(modelo, adaptadores: dict, escala: float = ESCALA_LORA) -> int:
    params = dict(modelo.named_parameters())
    n = 0
    for k, a in adaptadores.items():
        if not k.endswith(".lora_a"):
            continue
        prefixo = k[: -len(".lora_a")]
        b = adaptadores[f"{prefixo}.lora_b"]
        alvo = params.get(f"{prefixo}.weight")
        if alvo is None:
            raise KeyError(f"camada sem peso correspondente para {prefixo}")
        delta = (escala * (a.to(torch.float32) @ b.to(torch.float32))).T      # [out, in]
        with torch.no_grad():
            alvo.add_(delta.to(alvo.dtype))
        n += 1
    return n


class DecisorTorch:
    """Mesma interface do Decisor MLX: decide({"state", "questions"}, amostras) → {"answers": …}."""

    def __init__(self, execucao: str | Path, dispositivo: str | None = None, dtype: str = "auto",
                 max_state: int = 4096):
        d = Path(execucao)
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
        self.dispositivo = dispositivo or dispositivo_padrao()
        if dtype == "auto":
            tipo = torch.float16 if self.dispositivo == "mps" else torch.bfloat16
        else:
            tipo = getattr(torch, dtype)
        repo = cfg["modelo"]
        if repo.startswith("mlx-community/") and ("bit" in repo):
            self.modelo, tok = _carrega_base_mlx_quantizada(repo, tipo)
        else:
            self.modelo = AutoModelForCausalLM.from_pretrained(repo, dtype=tipo)
            tok = AutoTokenizer.from_pretrained(repo)
        adapt = load_file(str(d / "adaptadores.safetensors"))
        self.n_lora = _funde_lora(self.modelo, adapt)
        cal_path = d / "calibracao.json"
        cal = json.loads(cal_path.read_text(encoding="utf-8")) if cal_path.exists() else {}
        self._prepara(self.modelo, tok, tipo, max_state, cal.get("conformal"))
        self.sonda = carrega_sonda(d)

    def _prepara(self, modelo, tok, tipo, max_prefixo, conformal):
        self.modelo = modelo.to(self.dispositivo).eval()
        self.dtype = tipo
        self.cod = Codificador(tok)
        self.max_prefixo = max_prefixo
        self.conformal = conformal

    @classmethod
    def de_objetos(cls, modelo, tokenizer, dispositivo: str = "cpu", conformal: dict | None = None,
                   max_prefixo: int = 4096):
        """Monta o decisor a partir de um modelo já carregado (usado nos testes)."""
        obj = cls.__new__(cls)
        obj.dispositivo, obj.n_lora = dispositivo, 0
        obj._prepara(modelo, tokenizer, next(modelo.parameters()).dtype, max_prefixo, conformal)
        obj.sonda = None
        return obj

    @torch.no_grad()
    def _passa(self, ids, n_pre, segs, leituras) -> torch.Tensor:
        ok = torch.from_numpy(mascara_blocos_np(len(ids), n_pre, segs))
        minimo = torch.finfo(self.dtype).min
        mask = torch.zeros(ok.shape, dtype=self.dtype).masked_fill(~ok, minimo)[None, None].to(self.dispositivo)
        entrada = torch.tensor([ids], device=self.dispositivo)
        saida = self.modelo.model(input_ids=entrada, attention_mask=mask)
        h = saida.last_hidden_state[0, leituras]
        self._ultimo_estado = h.to(torch.float32).cpu().numpy()  # para a sonda de injeção
        return self.modelo.lm_head(h).to(torch.float32)          # [P, V]

    def decide(self, pedido: dict, amostras: int = 1, semente: int = 0) -> dict:
        state, questions = pedido["state"], pedido["questions"]
        rnd = random.Random(semente)
        acumulado = {k: [] for k in questions}
        for t in range(max(1, amostras)):
            perms = {k: perm_aleatoria(q, rnd) for k, q in questions.items()} if t > 0 else None
            ids, n_pre, segs, leit, blocos = self.cod.codifica(state, questions, perms, self.max_prefixo)
            logits = self._passa(ids, n_pre, segs, leit)
            if t == 0:
                estados = (self._ultimo_estado, [b.chave for b in blocos])
            for i, b in enumerate(blocos):
                sel = logits[i, b.ids]
                p = torch.softmax(sel, dim=-1)[indices_canonicos(b, questions[b.chave])]
                acumulado[b.chave].append(p.cpu().numpy())
        r = {"answers": {k: formata(questions[k], np.stack(v), self.conformal) for k, v in acumulado.items()}}
        aplica_sonda(self.sonda, estados[0], estados[1], r)
        if avisos(self.cod):
            r["avisos"] = avisos(self.cod)
        return r
