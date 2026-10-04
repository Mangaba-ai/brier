"""Motor do Brier v4: arquitetura e pesos do Laya (encoder mmBERT de 322M + cabeça de decisão),
usados como estão, sem ajuste fino. Pesos: convaiinnovations/laya-multilingual (Apache-2.0).

Recursos:
  - uma passada para todas as perguntas; ~50 ms por pedido em GPU de Apple Silicon;
  - documentos longos: se o state não cabe na janela, varre por janelas sobrepostas (predict_long);
  - lotes: vários states com as mesmas perguntas numa chamada (decide_lote);
  - abstenção: min_confidence marca respostas abaixo da certeza mínima;
  - roteador por idioma: português (e o que estiver em `idiomas`) vai para o Brier; os demais, para
    um modelo multilíngue geral, se configurado;
  - ganchos: log de auditoria em JSONL (sem o texto do state, só o hash) e mascaramento de dados
    pessoais (CPF, CNPJ, cartão, e-mail, telefone, CEP), mais funções próprias antes/depois.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

from .pii import mascarar

MODELO_PADRAO = os.environ.get("BRIER_MODELO", "convaiinnovations/laya-multilingual")
MODELO_OUTROS_IDIOMAS = "convaiinnovations/laya-multilingual"


def _resumo(resposta: dict) -> dict:
    out = {}
    for k, a in resposta.get("answers", {}).items():
        out[k] = {c: a.get(c) for c in ("type", "choice", "score", "noul", "confidence") if c in a}
    return out


class Brier:
    """Decisões tipadas (choice, score, noul) com probabilidade calibrada."""

    def __init__(self, modelo: str = MODELO_PADRAO, *, dispositivo: Optional[str] = None,
                 idiomas: tuple = ("pt",), modelo_outros_idiomas: Optional[str] = None,
                 mascarar_pii: bool = False, auditoria: Optional[str] = None,
                 antes: Optional[Callable] = None, depois: Optional[Callable] = None):
        import laya  # importado aqui: o pacote carrega rápido sem o motor
        self._laya = laya
        self.nome_modelo = modelo
        self.agente = laya.load(modelo, device=dispositivo)
        self.idiomas = set(idiomas)
        self.modelo_outros = modelo_outros_idiomas
        self._outros = None
        self.mascarar_pii = mascarar_pii
        self.auditoria = Path(auditoria) if auditoria else None
        self.antes, self.depois = antes, depois
        self._trava = threading.Lock()

    # ---------- roteador por idioma
    def _escolhe(self, state) -> tuple:
        info = self._laya.detect_language(state)
        idioma = info.get("language") or "indefinido"
        if self.modelo_outros and idioma not in self.idiomas and not info.get("language_undecided"):
            if self._outros is None:
                self._outros = self._laya.load(self.modelo_outros)
            return self._outros, {"idioma": idioma, "modelo": self.modelo_outros}
        return self.agente, {"idioma": idioma, "modelo": self.nome_modelo}

    # ---------- ganchos
    def _registra(self, ctx: dict, resposta: dict):
        if self.depois:
            self.depois(ctx, resposta)
        if self.auditoria:
            linha = {"id": ctx["id"], "ts": ctx["ts"], "latencia_ms": resposta.get("latencia_ms"),
                     "state_sha256": ctx["state_sha256"], "state_chars": ctx["state_chars"],
                     "roteamento": resposta.get("roteamento"), "respostas": _resumo(resposta)}
            self.auditoria.parent.mkdir(parents=True, exist_ok=True)
            with self.auditoria.open("a", encoding="utf-8") as f:
                f.write(json.dumps(linha, ensure_ascii=False) + "\n")

    def _contexto(self, state) -> dict:
        bruto = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, sort_keys=True)
        ctx = {"id": uuid.uuid4().hex[:12], "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "state_sha256": hashlib.sha256(bruto.encode("utf-8")).hexdigest(), "state_chars": len(bruto)}
        if self.antes:
            self.antes(ctx, state)
        return ctx

    # ---------- decisões
    def decide(self, state, questions: dict, *, min_confidence: Optional[float] = None,
               longo: str = "auto") -> dict:
        """longo: "auto" (janelas só se o state não couber), "sim" ou "nao"."""
        ctx = self._contexto(state)
        entrada = mascarar(state) if self.mascarar_pii else state
        agente, rota = self._escolhe(entrada)
        t = time.time()
        with self._trava:
            if longo == "sim":
                r = agente.predict_long(entrada, questions)
            else:
                r = agente.predict(entrada, questions, min_confidence=min_confidence)
                if longo == "auto" and r.get("usage", {}).get("truncated"):
                    r = agente.predict_long(entrada, questions)
                    r.setdefault("avisos", []).append("state longo: lido por janelas sobrepostas")
        r["model"], r["roteamento"] = "brier-4", rota
        r["latencia_ms"] = round((time.time() - t) * 1000, 1)
        if min_confidence is not None:
            abst = [k for k, a in r["answers"].items() if a.get("low_confidence")]
            r["abstencao"] = {"min_confidence": min_confidence, "perguntas": abst}
        self._registra(ctx, r)
        return r

    def decide_lote(self, states: list, questions: dict, *, min_confidence: Optional[float] = None) -> list:
        ctxs = [self._contexto(s) for s in states]
        entradas = [mascarar(s) for s in states] if self.mascarar_pii else states
        t = time.time()
        with self._trava:
            rs = self.agente.predict_batch(entradas, questions, sort_by_length=True, min_confidence=min_confidence)
        ms = round((time.time() - t) * 1000 / max(1, len(states)), 1)
        for ctx, r in zip(ctxs, rs):
            r["model"], r["latencia_ms"] = "brier-4", ms
            r["roteamento"] = {"modelo": self.nome_modelo, "idioma": "lote"}
            self._registra(ctx, r)
        return rs
