"""Atalho: o código vive em brier/conformal.py (pacote instalável). Mantido para os scripts de src/."""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from brier import conformal as _m  # noqa: E402

globals().update({_k: getattr(_m, _k) for _k in dir(_m) if not _k.startswith("__")})
