"""Brier: decisões tipadas e calibradas (choice, score, noul), locais e abertas.

    from brier import Brier
    b = Brier()                       # v3; também "v2", "comite", "rapido" ou uma pasta/repositório
    b.decide("internet caiu de novo, quero cancelar",
             {"setor": {"type": "choice", "instructions": "Qual time atende?",
                        "criteria": {"suporte": "Queda", "retencao": "Cancelamento"}}})
"""
from .motor import Brier, MODELO_PADRAO  # noqa: F401
from .pii import mascarar  # noqa: F401

__version__ = "4.1.0"
