"""Fachada do Brier: escolhe a versão e acrescenta os recursos de produção.

Versões (parâmetro `modelo`, ou variável de ambiente BRIER_MODELO):
  - "v5"     (padrão)  o v3 continuado com dados do domínio; o mais preciso e o mais resistente a injeção.
                       Licença comercial: o download exige acesso aprovado no Hugging Face. Sem acesso,
                       o padrão volta para o v3 com um aviso;
  - "v3"               Qwen3-4B + LoRA do Brier, treinado também contra injeção de instruções;
  - "v2"               Qwen3-4B + LoRA do Brier, o mais preciso em texto confiável;
  - "comite"           v2 + v3 com a média das probabilidades;
  - uma pasta local ou um repositório do Hugging Face com adaptadores do Brier.

Recursos para todas as versões:
  - lotes (decide_lote), abstenção por min_confidence, aviso de truncamento de textos longos;
  - defesa por regras contra instruções injetadas no texto (regras.py): marca e, se pedido, neutraliza;
  - log de auditoria em JSONL (sem o texto do state, só o hash) e mascaramento de dados pessoais.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
import warnings
from pathlib import Path
from typing import Callable, Optional

from .pii import mascarar
from .regras import neutraliza_injecao, procura_injecao

VERSOES = {
    "v5": ["mangaba-ai/brier-v5"],
    "v3": ["mangaba-ai/brier-v3"],
    "v2": ["mangaba-ai/brier-v2"],
    "comite": ["mangaba-ai/brier-v2", "mangaba-ai/brier-v3"],
}
MODELO_PADRAO = os.environ.get("BRIER_MODELO", "v5")
RESERVA = "v3"  # usado quando o padrão é o v5 e a conta não tem acesso aos pesos dele


def _resolve(nome: str) -> list[str]:
    """Nome de versão → pastas/repositórios. Dentro do repositório do Brier, usa execucoes/<versão> local."""
    repos = VERSOES.get(nome, [nome])
    out = []
    for r in repos:
        local = Path("execucoes") / r.split("/")[-1].replace("brier-", "")
        out.append(str(local) if r.startswith("mangaba-ai/brier-") and (local / "adaptadores.safetensors").exists() else r)
    return out


def _sem_acesso(erro: Exception) -> bool:
    """Erro de repositório com acesso controlado ou de autenticação no Hugging Face."""
    nome = type(erro).__name__
    texto = str(erro)
    return nome in ("GatedRepoError", "RepositoryNotFoundError") or "401" in texto or "403" in texto or "gated" in texto.lower()


def _resumo(resposta: dict) -> dict:
    out = {}
    for k, a in resposta.get("answers", {}).items():
        out[k] = {c: a.get(c) for c in ("type", "choice", "score", "noul", "confidence") if c in a}
    return out


class Brier:
    """Decisões tipadas (choice, score, noul) com probabilidade calibrada."""

    def __init__(self, modelo: str = MODELO_PADRAO, *, dispositivo: Optional[str] = None,
                 mascarar_pii: bool = False, auditoria: Optional[str] = None,
                 neutralizar_injecao: bool = False, max_state: int = 4096,
                 antes: Optional[Callable] = None, depois: Optional[Callable] = None):
        from .qwen import MotorQwen  # importado só quando usado: o pacote carrega rápido sem o motor
        self.versao = modelo
        caminhos = _resolve(modelo)
        try:
            self.agente = MotorQwen(caminhos, dispositivo=dispositivo, max_state=max_state)
        except Exception as erro:
            if modelo != "v5" or not _sem_acesso(erro):
                raise
            warnings.warn("Sem acesso aos pesos do Brier v5 (licença comercial; peça acesso em "
                          "https://huggingface.co/mangaba-ai/brier-v5 e faça `huggingface-cli login`). "
                          f"Usando o {RESERVA}.", stacklevel=2)
            self.versao = RESERVA
            caminhos = _resolve(RESERVA)
            self.agente = MotorQwen(caminhos, dispositivo=dispositivo, max_state=max_state)
        self.nome_modelo = "+".join(caminhos)
        self.mascarar_pii = mascarar_pii
        self.neutralizar_injecao = neutralizar_injecao
        self.auditoria = Path(auditoria) if auditoria else None
        self.antes, self.depois = antes, depois
        self._trava = threading.Lock()

    @property
    def rotulo(self) -> str:
        return {"v5": "brier-5", "v3": "brier-3", "v2": "brier-2", "comite": "brier-comite"}.get(self.versao, "brier")

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
    def decide(self, state, questions: dict, *, min_confidence: Optional[float] = None) -> dict:
        ctx = self._contexto(state)
        entrada, injecao = self._prepara(state)
        t = time.time()
        with self._trava:
            r = self.agente.predict(entrada, questions, min_confidence=min_confidence)
        r["model"], r["roteamento"] = self.rotulo, {"modelo": self.nome_modelo}
        r["latencia_ms"] = round((time.time() - t) * 1000, 1)
        if injecao:
            r["injecao"] = injecao
        if min_confidence is not None:
            abst = [k for k, a in r["answers"].items() if a.get("low_confidence")]
            r["abstencao"] = {"min_confidence": min_confidence, "perguntas": abst}
        self._registra(ctx, r)
        return r

    def decide_lote(self, states: list, questions: dict, *, min_confidence: Optional[float] = None) -> list:
        return [self.decide(s, questions, min_confidence=min_confidence) for s in states]
