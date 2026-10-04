"""Brier: decisões tipadas e calibradas (choice, score, noul), locais e abertas.

    from brier import Brier
    b = Brier()                       # pesos do Laya multilíngue (ou outra pasta/repositório)
    b.decide("internet caiu de novo, quero cancelar",
             {"setor": {"type": "choice", "instructions": "Qual time atende?",
                        "criteria": {"suporte": "Queda", "retencao": "Cancelamento"}}})
"""
from .motor import Brier, MODELO_PADRAO  # noqa: F401
from .pii import mascarar  # noqa: F401

__version__ = "4.0.0"
