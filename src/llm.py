"""Cliente mínimo para o MiMo (OpenAI-compat), usado como gerador e rotulador de dados."""
import json
import re
import time

import httpx

import chaves

_BASE, _KEY = chaves.mimo()
_cli = httpx.Client(timeout=180)


def chat(prompt: str, model: str = "mimo-v2.6-pro", temperature: float = 0.9, max_tokens: int = 6000,
         system: str | None = None) -> str:
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    body = {"model": model, "messages": msgs, "temperature": temperature, "max_tokens": max_tokens,
            "reasoning_effort": "none"}
    for tentativa in range(4):
        try:
            r = _cli.post(f"{_BASE}/chat/completions", json=body, headers={"Authorization": f"Bearer {_KEY}"})
            if r.status_code in (429, 500, 502, 503, 529):
                time.sleep(3 * (tentativa + 1))
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"].get("content") or ""
        except (httpx.TimeoutException, httpx.TransportError):
            time.sleep(3 * (tentativa + 1))
    raise RuntimeError("MiMo não respondeu após 4 tentativas")


def json_de(texto: str):
    """Extrai o primeiro bloco JSON (objeto ou lista) de uma resposta."""
    texto = re.sub(r"<think>.*?</think>", "", texto, flags=re.S)
    m = re.search(r"```(?:json)?\s*(.*?)```", texto, flags=re.S)
    if m:
        texto = m.group(1)
    ini = min([i for i in (texto.find("{"), texto.find("[")) if i >= 0], default=-1)
    if ini < 0:
        raise ValueError("sem JSON")
    return json.JSONDecoder().raw_decode(texto[ini:])[0]
