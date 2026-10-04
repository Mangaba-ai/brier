"""Decisor tipado — mesma entrada e saída do Jev (state + questions → answers).

Arquitetura (o que o Jev descreve como "lê o state uma vez e responde tudo na mesma passada"):
  - Um decoder (Qwen3) recebe [prefixo = system + state] seguido de um bloco por pergunta.
  - Máscara de atenção em blocos: cada pergunta enxerga o prefixo e a si mesma, nunca as outras.
    Assim as respostas são independentes (como perguntas separadas) mas custam UMA passada.
  - Leitura por logits: no último token de cada bloco lemos só os tokens-rótulo das opções
    (" A", " B"… para choice; " 0"…" 9" para score; " A"=sim/" B"=não para noul) e aplicamos softmax
    restrito. Nada é gerado, então a saída sempre cabe no tipo pedido.

Camadas estocásticas (inferência bayesiana aproximada + calibração):
  1. MC Dropout nos adaptadores LoRA — Gal, tese de doutorado "Uncertainty in Deep Learning"
     (Cambridge, 2016): dropout ligado na inferência ≈ amostras da posterior. As T amostras vão no
     eixo de lote, numa chamada só.
  2. SWA / SWAG-diagonal sobre os pesos LoRA — Izmailov, tese "Deconstructing Models and Methods in
     Deep Learning" (NYU, 2023): média dos pesos no fim do treino (SWA) e Gaussiana diagonal para
     amostrar pesos (SWAG).
  3. Permutação estocástica das opções no teste: cada amostra vê as opções em outra ordem e a média
     remove o viés de posição (A costuma ganhar).
  4. Escala de temperatura por tipo de pergunta, ajustada na validação.
  5. Predição conformal (conjunto com cobertura garantida) — em `conformal.py`.
  A média das amostras dá a probabilidade; a informação mútua separa incerteza epistêmica
  (o modelo não sabe) de aleatória (o caso é ambíguo).
"""
from __future__ import annotations

import math
import random

import mlx.core as mx
import numpy as np

# tudo que não depende do MLX vem do núcleo portátil (o motor PyTorch usa o mesmo código)
from nucleo import (LETRAS, SISTEMA, Bloco, Codificador, _txt, aplica_sonda, avisos, confianca, formata,  # noqa: F401
                    indices_canonicos, mascara_blocos_np, monta_bloco, ordem_canonica,
                    perm_aleatoria, valida_pergunta)


def mascara_blocos(L: int, n_pre: int, segs) -> mx.array:
    """Máscara aditiva (0 ou −inf) a partir da máscara booleana do núcleo."""
    ok = mx.array(mascara_blocos_np(L, n_pre, segs))
    return mx.where(ok, 0.0, -math.inf).astype(mx.float32)


def estado_leitura(model, ids_lote: mx.array, mask: mx.array, leituras: list[int]) -> mx.array:
    """Estado oculto final (após a norma) nas posições de leitura. [B, P, D] — usado pela sonda."""
    inner = model.model
    h = inner.embed_tokens(ids_lote)
    m = mask.astype(h.dtype)
    for layer in inner.layers:
        h = layer(h, m, None)
    return inner.norm(h[:, mx.array(leituras), :])


def para_logits(model, h: mx.array) -> mx.array:
    if model.args.tie_word_embeddings:
        return model.model.embed_tokens.as_linear(h)
    return model.lm_head(h)


def passa(model, ids_lote: mx.array, mask: mx.array, leituras: list[int]) -> mx.array:
    """Uma passada com máscara própria; devolve logits só nas posições de leitura. [B, P, V]"""
    h = estado_leitura(model, ids_lote, mask, leituras)
    inner = model.model
    if model.args.tie_word_embeddings:
        return inner.embed_tokens.as_linear(h)
    return model.lm_head(h)


def dist_restrita(logits_pos: mx.array, b: Bloco, temperatura: float = 1.0) -> mx.array:
    """log-probabilidades sobre os rótulos da pergunta, na ORDEM CANÔNICA das opções."""
    sel = logits_pos[..., mx.array(b.ids)].astype(mx.float32) / temperatura
    return sel - mx.logsumexp(sel, axis=-1, keepdims=True)




def para_canonica(logp: mx.array, b: Bloco, q: dict) -> mx.array:
    """Reordena as colunas apresentadas (permutadas) para a ordem canônica da pergunta."""
    return logp[..., mx.array(indices_canonicos(b, q))]






class Decisor:
    """Inferência no formato Jev, com amostragem estocástica opcional."""

    def __init__(self, model, tokenizer, temperaturas: dict | None = None, swag: dict | None = None,
                 conformal: dict | None = None, max_state: int = 4096, sonda: dict | None = None):
        self.model, self.cod = model, Codificador(tokenizer)
        self.temp = temperaturas or {"choice": 1.0, "score": 1.0, "noul": 1.0}
        self.swag = swag            # {"media": {...}, "var": {...}} dos pesos LoRA, se houver
        self.conformal = conformal  # {"choice": qhat, "score": qhat, "noul": qhat}
        self.max_state = max_state
        self.sonda = sonda          # sonda linear de injeção (nucleo.carrega_sonda), opcional

    def _amostras(self, state, questions, T: int, estocastico: bool, semente: int):
        """Devolve {chave: [T, K] log-probs canônicas}. Com T>1 e permutação, cada amostra é uma
        passada com outra ordem de opções; dropout (se ligado) sorteia máscara nova por amostra."""
        rnd = random.Random(semente)
        acumulado = {k: [] for k in questions}
        for t in range(T):
            perms = {k: perm_aleatoria(q, rnd) for k, q in questions.items()} if (estocastico and t > 0) else None
            ids, n_pre, segs, leit, blocos = self.cod.codifica(state, questions, perms, self.max_state)
            mask = mascara_blocos(len(ids), n_pre, segs)
            h = estado_leitura(self.model, mx.array([ids]), mask, leit)
            logits = para_logits(self.model, h)[0]
            if t == 0:
                self._estados = (np.array(h[0].astype(mx.float32)), [b.chave for b in blocos])
            for i, b in enumerate(blocos):
                q = questions[b.chave]
                lp = para_canonica(dist_restrita(logits[i], b, self.temp[b.tipo]), b, q)
                acumulado[b.chave].append(lp)
        return {k: mx.stack(v) for k, v in acumulado.items()}

    def decide(self, pedido: dict, amostras: int = 1, semente: int = 0) -> dict:
        state, questions = pedido["state"], pedido["questions"]
        estoc = amostras > 1
        if estoc and self.swag is None:
            self.model.train()  # liga o dropout do LoRA (MC Dropout)
        try:
            lps = self._amostras(state, questions, amostras, estoc, semente)
        finally:
            self.model.eval()
        mx.eval(lps)
        r = {"answers": {k: self._formata(questions[k], lps[k]) for k in questions}}
        aplica_sonda(self.sonda, self._estados[0], self._estados[1], r)
        if avisos(self.cod):
            r["avisos"] = avisos(self.cod)
        return r

    def _formata(self, q: dict, lp: mx.array) -> dict:
        return formata(q, np.array(mx.exp(lp)), self.conformal)
