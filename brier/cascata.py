"""Brier-0: cascata aluno → professor.

O aluno (Qwen3-4B + LoRA do Brier-0) decide sozinho. Nas perguntas de escolha e de sim/não em que a
confiança dele (probabilidade da opção mais provável) fica abaixo do limiar, consulta um professor grande
por um endpoint compatível com a API da OpenAI (`/v1/completions` com logprobs). Por padrão é o
mangaba-titan (Qwen3.5-122B) no servidor GPU da Mangaba. O professor responde nas duas ordens de opções
(a apresentada e a invertida) e a média tira o viés de posição dele. Perguntas de escala nunca vão ao
professor: nelas o aluno é melhor.

O limiar 0,9 foi escolhido só na validação do sintético (src/zero/cascata.py). Medido no teste:
sintético limpo 89,3% → 92,0%, consultando o professor em ~36% das perguntas; detalhes em
execucoes/zero/cascata_r2.json.

Configuração (variáveis de ambiente):
  BRIER_PROFESSOR_URL     padrão https://llm-gpu.mangaba.ia.br/v1
  BRIER_PROFESSOR_CHAVE   chave do endpoint (sem ela, a cascata fica desligada e só o aluno responde)
  BRIER_PROFESSOR_MODELO  padrão mangaba-qwen35-122b
  BRIER_LIMIAR_CASCATA    padrão 0.9
"""
from __future__ import annotations

import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .nucleo import SISTEMA, _txt, avisos, formata, indices_canonicos, monta_bloco

LIMIAR_PADRAO = float(os.environ.get("BRIER_LIMIAR_CASCATA", "0.9"))


class Professor:
    """Distribuição do professor sobre os rótulos de uma pergunta, pela média de duas ordens de opções."""

    def __init__(self, url: str | None = None, chave: str | None = None, modelo: str | None = None,
                 timeout: float = 60.0, max_chars: int = 12000):
        self.url = (url or os.environ.get("BRIER_PROFESSOR_URL", "https://llm-gpu.mangaba.ia.br/v1")).rstrip("/")
        self.chave = chave or os.environ.get("BRIER_PROFESSOR_CHAVE")
        self.modelo = modelo or os.environ.get("BRIER_PROFESSOR_MODELO", "mangaba-qwen35-122b")
        self.timeout, self.max_chars = timeout, max_chars

    @property
    def ativo(self) -> bool:
        return bool(self.chave)

    def _top(self, texto: str) -> dict:
        corpo = json.dumps({"model": self.modelo, "prompt": texto, "max_tokens": 1, "temperature": 0,
                            "logprobs": 20}).encode()
        # User-Agent explícito: o Cloudflare recusa (403) o padrão do urllib
        req = urllib.request.Request(self.url + "/completions", corpo, {
            "Content-Type": "application/json", "Authorization": f"Bearer {self.chave}", "User-Agent": "brier/5.2"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)["choices"][0]["logprobs"]["top_logprobs"][0]

    def distribuicao(self, state, q: dict) -> list[float]:
        s = _txt(state)
        if len(s) > self.max_chars:
            s = s[: self.max_chars * 2 // 3] + "\n[…]\n" + s[-self.max_chars // 3:]
        n = len(q["criteria"]) if q["type"] == "choice" else 2
        acum = np.zeros(n)
        for perm in (list(range(n)), list(range(n))[::-1]):
            txt, b = monta_bloco("q", q, perm, pensa=True)
            prompt = f"<|im_start|>system\n{SISTEMA}<|im_end|>\n<|im_start|>user\nESTADO:\n{s}<|im_end|>\n" + txt
            # o tokenizador do Qwen3.5 funde espaço e letra (" A"): o prompt termina em "Resposta:"
            top = self._top(prompt[:-1])
            piso = min(top.values()) - 2.0
            lps = np.array([top.get(" " + r, piso) for r in b.rotulos])
            p = np.exp(lps - lps.max())
            p /= p.sum()
            acum += p[indices_canonicos(b, q)] / 2
        return acum.tolist()


class MotorCascata:
    """Envolve um MotorQwen (o aluno) e consulta o professor nas perguntas incertas."""

    def __init__(self, aluno, professor: Professor | None = None, limiar: float = LIMIAR_PADRAO):
        self.aluno = aluno
        self.professor = professor or Professor()
        self.limiar = limiar

    @property
    def dispositivo(self) -> str:
        return self.aluno.dispositivo

    def predict(self, state, questions: dict, min_confidence: float | None = None) -> dict:
        por = {k: np.stack(v).mean(axis=0) for k, v in self.aluno._probs_membros(state, questions).items()}
        incertas = [k for k, p in por.items()
                    if questions[k]["type"] != "score" and float(p.max()) < self.limiar]
        info = {"limiar": self.limiar, "professor": self.professor.modelo, "consultadas": [], "falhas": []}
        if incertas and self.professor.ativo:
            with ThreadPoolExecutor(min(8, len(incertas))) as pool:
                futuros = {k: pool.submit(self.professor.distribuicao, state, questions[k]) for k in incertas}
            for k, f in futuros.items():
                try:
                    por[k] = np.asarray(f.result(), dtype=np.float64)
                    info["consultadas"].append(k)
                except Exception as erro:  # professor fora do ar: fica a resposta do aluno
                    info["falhas"].append({"pergunta": k, "erro": repr(erro)[:200]})
        elif incertas:
            info["falhas"].append({"erro": "BRIER_PROFESSOR_CHAVE não definida: só o aluno respondeu"})
        r = {"answers": {k: formata(questions[k], p[None], self.aluno.conformal) for k, p in por.items()},
             "cascata": info}
        if min_confidence is not None:
            for a in r["answers"].values():
                topo = a["noul"] if a["type"] == "noul" else max(a["probabilities"].values())
                if a["type"] == "noul":
                    topo = max(topo, 1 - topo)
                a["low_confidence"] = bool(topo < min_confidence)
        if avisos(self.aluno.cod):
            r["avisos"] = avisos(self.aluno.cod)
        r["usage"] = {"truncated": bool(getattr(self.aluno.cod, "truncamento", None))}
        return r
