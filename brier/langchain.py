"""Integração com LangChain / LangGraph: o Brier como ferramenta e como roteador.

    from brier import Brier
    from brier.langchain import ferramenta_brier, roteador_brier
    b = Brier()
    tool = ferramenta_brier(b)                          # StructuredTool para agentes
    rota = roteador_brier(b, "Qual time atende?",       # função para add_conditional_edges
                          {"suporte": "Queda", "financeiro": "Boleto"}, chave_estado="mensagem")

Instale com:  pip install "brier[langchain]"
"""
from typing import Any, Callable, Optional

from langchain_core.tools import StructuredTool


def ferramenta_brier(brier, nome: str = "brier_decidir") -> StructuredTool:
    def decidir(state: Any, questions: dict, min_confidence: Optional[float] = None) -> dict:
        return brier.decide(state, questions, min_confidence=min_confidence)
    return StructuredTool.from_function(
        decidir, name=nome,
        description="Decide perguntas tipadas (choice, score, noul) sobre um texto e devolve probabilidades calibradas.")


def roteador_brier(brier, instrucao: str, opcoes: dict, chave_estado: str = "input",
                   min_confidence: Optional[float] = None, se_incerto: Optional[str] = None) -> Callable[[dict], str]:
    """Função de roteamento para LangGraph: lê state[chave_estado] e devolve a opção escolhida.
    Com min_confidence e se_incerto, devolve se_incerto quando a certeza fica abaixo do mínimo."""
    def rota(estado: dict) -> str:
        r = brier.decide(estado[chave_estado], {"rota": {"type": "choice", "instructions": instrucao, "criteria": opcoes}},
                         min_confidence=min_confidence)["answers"]["rota"]
        if se_incerto and r.get("low_confidence"):
            return se_incerto
        return r["choice"]
    return rota
