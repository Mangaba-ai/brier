"""Fachada do Brier: escolhe a versão e acrescenta os recursos de produção.

Versões (parâmetro `modelo`, ou variável de ambiente BRIER_MODELO):
  - "v3"     (padrão)  Qwen3-4B + LoRA do Brier, treinado também contra injeção de instruções;
  - "v2"               Qwen3-4B + LoRA do Brier, o mais preciso em texto confiável;
  - "comite"           v2 + v3 com a média das probabilidades;
  - "rapido"           arquitetura e pesos do Laya como estão (~50 ms, bem menos preciso em português);
  - uma pasta local ou um repositório do Hugging Face com adaptadores do Brier ou pesos no formato Laya.

Recursos para todas as versões:
  - lotes (decide_lote), abstenção por min_confidence, aviso de truncamento de textos longos;
  - defesa por regras contra instruções injetadas no texto (regras.py): marca e, se pedido, neutraliza;
  - log de auditoria em JSONL (sem o texto do state, só o hash) e mascaramento de dados pessoais;
  - no "rapido": textos longos por janelas sobrepostas e roteador por idioma.
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
from .regras import neutraliza_injecao, procura_injecao

VERSOES = {
    "v3": ["mangaba-ai/brier-v3"],
    "v2": ["mangaba-ai/brier-v2"],
    "comite": ["mangaba-ai/brier-v2", "mangaba-ai/brier-v3"],
    "rapido": ["convaiinnovations/laya-multilingual"],
}
MODELO_PADRAO = os.environ.get("BRIER_MODELO", "v3")
MODELO_OUTROS_IDIOMAS = "convaiinnovations/laya-multilingual"


def _resolve(nome: str) -> list[str]:
    """Nome de versão → pastas/repositórios. Dentro do repositório do Brier, usa execucoes/<versão> local."""
    repos = VERSOES.get(nome, [nome])
    out = []
    for r in repos:
        local = Path("execucoes") / r.split("/")[-1].replace("brier-", "")
        out.append(str(local) if r.startswith("mangaba-ai/brier-") and (local / "adaptadores.safetensors").exists() else r)
    return out


def _eh_laya(caminho: str) -> bool:
    if caminho.startswith("convaiinnovations/"):
        return True
    return (Path(caminho) / "rl_agent_config.json").exists()


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
                 neutralizar_injecao: bool = False, max_state: int = 4096,
                 antes: Optional[Callable] = None, depois: Optional[Callable] = None):
        self.versao = modelo
        caminhos = _resolve(modelo)
        self.nome_modelo = "+".join(caminhos)
        if len(caminhos) == 1 and _eh_laya(caminhos[0]):
            import laya  # importado só quando usado: o pacote carrega rápido sem o motor
            self._laya = laya
            self.agente = laya.load(caminhos[0], device=dispositivo)
            self.tipo = "laya"
        else:
            from .qwen import MotorQwen
            self.agente = MotorQwen(caminhos, dispositivo=dispositivo, max_state=max_state)
            self.tipo = "qwen"
            self._laya = None
        self.idiomas = set(idiomas)
        self.modelo_outros = modelo_outros_idiomas
        self._outros = None
        self.mascarar_pii = mascarar_pii
        self.neutralizar_injecao = neutralizar_injecao
        self.auditoria = Path(auditoria) if auditoria else None
        self.antes, self.depois = antes, depois
        self._trava = threading.Lock()

    @property
    def rotulo(self) -> str:
        return {"v3": "brier-3", "v2": "brier-2", "comite": "brier-comite", "rapido": "brier-4"}.get(self.versao, "brier")

    # ---------- roteador por idioma (só no "rapido", que tem o detector do Laya)
    def _escolhe(self, state) -> tuple:
        if self._laya is None or not self.modelo_outros:
            return self.agente, {"modelo": self.nome_modelo}
        info = self._laya.detect_language(state)
        idioma = info.get("language") or "indefinido"
        if idioma not in self.idiomas and not info.get("language_undecided"):
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
                     "roteamento": resposta.get("roteamento"), "injecao": resposta.get("injecao"),
                     "respostas": _resumo(resposta)}
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

    def _prepara(self, state):
        """Mascara dados pessoais e trata instruções injetadas. Devolve (entrada, info de injeção)."""
        entrada = mascarar(state) if self.mascarar_pii else state
        achados = procura_injecao(entrada)
        info = None
        if achados:
            info = {"suspeita": True, "trechos": achados[:5], "neutralizado": self.neutralizar_injecao}
            if self.neutralizar_injecao:
                entrada = neutraliza_injecao(entrada)
        return entrada, info

    # ---------- decisões
    def decide(self, state, questions: dict, *, min_confidence: Optional[float] = None,
               longo: str = "auto") -> dict:
        """longo (só no "rapido"): "auto" (janelas se não couber), "sim" ou "nao"."""
        ctx = self._contexto(state)
        entrada, injecao = self._prepara(state)
        agente, rota = self._escolhe(entrada)
        t = time.time()
        with self._trava:
            if self.tipo == "laya" and longo == "sim":
                r = agente.predict_long(entrada, questions)
            else:
                r = agente.predict(entrada, questions, min_confidence=min_confidence)
                if self.tipo == "laya" and longo == "auto" and r.get("usage", {}).get("truncated"):
                    r = agente.predict_long(entrada, questions)
                    r.setdefault("avisos", []).append("state longo: lido por janelas sobrepostas")
        r["model"], r["roteamento"] = self.rotulo, rota
        r["latencia_ms"] = round((time.time() - t) * 1000, 1)
        if injecao:
            r["injecao"] = injecao
        if min_confidence is not None:
            abst = [k for k, a in r["answers"].items() if a.get("low_confidence")]
            r["abstencao"] = {"min_confidence": min_confidence, "perguntas": abst}
        self._registra(ctx, r)
        return r

    def decide_lote(self, states: list, questions: dict, *, min_confidence: Optional[float] = None) -> list:
        if self.tipo == "qwen":
            return [self.decide(s, questions, min_confidence=min_confidence) for s in states]
        ctxs = [self._contexto(s) for s in states]
        preparados = [self._prepara(s) for s in states]
        t = time.time()
        with self._trava:
            rs = self.agente.predict_batch([p[0] for p in preparados], questions, sort_by_length=True,
                                           min_confidence=min_confidence)
        ms = round((time.time() - t) * 1000 / max(1, len(states)), 1)
        for ctx, (_, injecao), r in zip(ctxs, preparados, rs):
            r["model"], r["latencia_ms"] = self.rotulo, ms
            r["roteamento"] = {"modelo": self.nome_modelo, "idioma": "lote"}
            if injecao:
                r["injecao"] = injecao
            self._registra(ctx, r)
        return rs
