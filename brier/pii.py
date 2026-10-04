"""Mascaramento de dados pessoais brasileiros (opcional, antes da inferência e sempre nos logs)."""
import re

PADROES = [
    ("CPF", re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")),
    ("CNPJ", re.compile(r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b")),
    ("CARTAO", re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("TELEFONE", re.compile(r"(?:\+?55\s?)?\(?\b\d{2}\)?\s?9?\d{4}[-\s]?\d{4}\b")),
    ("CEP", re.compile(r"\b\d{5}-\d{3}\b")),
]


def mascarar_texto(texto: str) -> str:
    for nome, rx in PADROES:  # CNPJ antes de telefone, cartão antes de CPF: a ordem importa
        texto = rx.sub(f"[{nome}]", texto)
    return texto


def mascarar(valor):
    """Mascara strings dentro de str, dict ou list (o state pode ser estruturado)."""
    if isinstance(valor, str):
        return mascarar_texto(valor)
    if isinstance(valor, dict):
        return {k: mascarar(v) for k, v in valor.items()}
    if isinstance(valor, list):
        return [mascarar(v) for v in valor]
    return valor
