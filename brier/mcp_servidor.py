"""Servidor MCP do Brier: expõe decisões tipadas como ferramentas para agentes (Claude, IDEs etc.).

Ferramentas: brier_decidir (um state) e brier_decidir_lote (vários states, mesmas perguntas).
Instale com:  pip install "brier[mcp]"
Configure no cliente MCP:  {"command": "brier-mcp", "env": {"BRIER_MODELO": "v3"}}
"""
from typing import Any, Optional

try:  # SDK MCP 2.x
    from mcp.server.mcpserver import MCPServer as _Servidor
except ImportError:  # SDK MCP 1.x
    from mcp.server.fastmcp import FastMCP as _Servidor

from .motor import MODELO_PADRAO, Brier

app = _Servidor("brier")
_brier: Optional[Brier] = None


def _motor() -> Brier:
    global _brier
    if _brier is None:
        _brier = Brier(MODELO_PADRAO)
    return _brier


@app.tool()
def brier_decidir(state: Any, questions: dict, min_confidence: Optional[float] = None) -> dict:
    """Decide perguntas tipadas sobre um conteúdo. questions: {"chave": {"type": "choice"|"score"|"noul",
    "instructions": "...", "criteria": ...}}. Devolve escolha, probabilidades e confiança por pergunta."""
    return _motor().decide(state, questions, min_confidence=min_confidence)


@app.tool()
def brier_decidir_lote(states: list, questions: dict, min_confidence: Optional[float] = None) -> list:
    """Mesmas perguntas sobre vários conteúdos de uma vez (mais rápido que um por um)."""
    return _motor().decide_lote(states, questions, min_confidence=min_confidence)


def main():
    app.run()


if __name__ == "__main__":
    main()
