"""Credenciais por variável de ambiente (ou um .env na raiz do projeto, que o .gitignore exclui).

  TYPESAFE_API_KEY   chave do Jev (só para comparar_jev.py)
  LLM_BASE_URL       endpoint OpenAI-compatível do gerador de dados (ex.: MiMo)
  LLM_API_KEY        chave desse endpoint
"""
import os
from pathlib import Path

_ENV = Path(__file__).resolve().parent.parent / ".env"


def carregar() -> dict:
    vals = {}
    if _ENV.exists():
        for linha in _ENV.read_text(encoding="utf-8").splitlines():
            if "=" in linha and not linha.lstrip().startswith("#"):
                k, v = linha.split("=", 1)
                vals[k.strip()] = v.strip().strip('"').strip("'")
    vals.update(os.environ)
    return vals


def jev_key() -> str:
    c = carregar()
    return c.get("TYPESAFE_API_KEY") or c["LLM_JEV_API_KEY"]


def mimo() -> tuple[str, str]:
    c = carregar()
    return c["LLM_BASE_URL"], c["LLM_API_KEY"]
