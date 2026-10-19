"""Cascata do Brier-0 sem rede nem modelo: aluno e professor falsos."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brier.cascata import MotorCascata  # noqa: E402

Q = {
    "setor": {"type": "choice", "instructions": "Qual time?", "criteria": {"suporte": "Queda", "retencao": "Cancelar"}},
    "urgente": {"type": "noul", "instructions": "É urgente?", "criteria": {"true": "sim", "false": "não"}},
    "nota": {"type": "score", "instructions": "Nota", "criteria": ["ruim", "ok", "bom"]},
}


class AlunoFalso:
    conformal = None
    dispositivo = "cpu"
    cod = None

    def __init__(self, probs):
        self.probs = probs

    def _probs_membros(self, state, questions):
        return {k: [np.array(p)] for k, p in self.probs.items()}


class ProfessorFalso:
    modelo = "titan-falso"

    def __init__(self, ativo=True, falha=False):
        self.ativo, self.falha, self.chamadas = ativo, falha, []

    def distribuicao(self, state, q):
        self.chamadas.append(q["instructions"])
        if self.falha:
            raise ConnectionError("fora do ar")
        return [0.05, 0.95]


def test_consulta_so_incertas_e_nunca_escala():
    aluno = AlunoFalso({"setor": [0.6, 0.4], "urgente": [0.97, 0.03], "nota": [0.4, 0.3, 0.3]})
    prof = ProfessorFalso()
    r = MotorCascata(aluno, prof, limiar=0.9).predict("x", Q)
    assert prof.chamadas == ["Qual time?"]          # sim/não confiante e escala ficam com o aluno
    assert r["answers"]["setor"]["choice"] == "retencao"
    assert r["cascata"]["consultadas"] == ["setor"]
    assert r["answers"]["urgente"]["noul"] == 0.97


def test_professor_fora_do_ar_mantem_aluno():
    aluno = AlunoFalso({"setor": [0.6, 0.4], "urgente": [0.5, 0.5], "nota": [1, 0, 0]})
    r = MotorCascata(aluno, ProfessorFalso(falha=True), limiar=0.9).predict("x", Q)
    assert r["answers"]["setor"]["choice"] == "suporte"
    assert len(r["cascata"]["falhas"]) == 2 and not r["cascata"]["consultadas"]


def test_sem_chave_avisa_e_responde_so_com_aluno():
    aluno = AlunoFalso({"setor": [0.6, 0.4], "urgente": [0.5, 0.5], "nota": [1, 0, 0]})
    prof = ProfessorFalso(ativo=False)
    r = MotorCascata(aluno, prof, limiar=0.9).predict("x", Q, min_confidence=0.7)
    assert not prof.chamadas and "BRIER_PROFESSOR_CHAVE" in r["cascata"]["falhas"][0]["erro"]
    assert r["answers"]["setor"]["low_confidence"] is True
