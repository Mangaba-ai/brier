"""Brier: decisões tipadas e calibradas (choice, score, noul), locais.

    from brier import Brier
    b = Brier()                       # v5 (sem acesso aprovado, v3); também "v3", "v2", "comite"
    b.decide("internet caiu de novo, quero cancelar",
             {"setor": {"type": "choice", "instructions": "Qual time atende?",
                        "criteria": {"suporte": "Queda", "retencao": "Cancelamento"}}})
"""
from .motor import Brier, MODELO_PADRAO  # noqa: F401
from .pii import mascarar  # noqa: F401

__version__ = "4.1.0"
