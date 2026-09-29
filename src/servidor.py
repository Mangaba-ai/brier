"""Servidor compatível com o Jev: POST /v1/systemone e GET /v1/models.

Mesmo corpo (model, state, questions) e mesma resposta ({"answers": {...}}). Campos a mais em cada
resposta, que clientes do Jev simplesmente ignoram: `incerteza` (total/aleatória/epistêmica) e,
com calibração conformal, `conjunto` (opções que cobrem a resposta certa com 90% de garantia).
Parâmetro opcional `amostras` (padrão 1): >1 liga as camadas estocásticas.

Uso: python src/servidor.py --execucao execucoes/v1 --porta 8790
"""
import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from mlx.utils import tree_unflatten
import mlx.core as mx
from mlx_lm import load

sys.path.insert(0, str(Path(__file__).parent))
from decisor import Decisor  # noqa: E402
from gerar import valida_pergunta  # noqa: E402
from treinar import prepara_lora  # noqa: E402

NOME = "brier-1"


def carrega_decisor(execucao: str, usar_swa: bool) -> Decisor:
    d = Path(execucao)
    cfg = json.loads((d / "config.json").read_text())
    model, tok = load(cfg["modelo"])
    prepara_lora(model, cfg["rank"], cfg["dropout"], cfg["camadas"])
    if usar_swa and (d / "swa_media.safetensors").exists():
        model.update(tree_unflatten(list(mx.load(str(d / "swa_media.safetensors")).items())))
    else:
        model.load_weights(str(d / "adaptadores.safetensors"), strict=False)
    model.eval()
    cal = json.loads((d / "calibracao.json").read_text()) if (d / "calibracao.json").exists() else {}
    return Decisor(model, tok, cal.get("temperaturas"), conformal=cal.get("conformal"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execucao", default="execucoes/v1")
    ap.add_argument("--porta", type=int, default=8790)
    ap.add_argument("--swa", action="store_true")
    a = ap.parse_args()
    dec = carrega_decisor(a.execucao, a.swa)

    class H(BaseHTTPRequestHandler):
        def _json(self, cod, obj):
            b = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(cod)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path.rstrip("/") == "/v1/models":
                return self._json(200, {"models": [{"name": NOME}, {"name": "jev-latest"}]})
            self._json(404, {"detail": "rota inexistente"})

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/systemone":
                return self._json(404, {"detail": "rota inexistente"})
            try:
                corpo = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                qs = corpo.get("questions")
                if "state" not in corpo or not isinstance(qs, dict) or not qs:
                    raise ValueError("corpo precisa de state e questions")
                ruins = [k for k, q in qs.items() if not valida_pergunta(q)]
                if ruins:
                    raise ValueError(f"perguntas inválidas: {ruins} (confira type e criteria)")
            except (ValueError, json.JSONDecodeError) as e:
                return self._json(422, {"detail": [{"msg": str(e)}]})
            t = time.time()
            r = dec.decide(corpo, amostras=int(corpo.get("amostras", 1)))
            r["model"], r["latencia_ms"] = NOME, round((time.time() - t) * 1000, 1)
            self._json(200, r)

        def log_message(self, *args):
            pass

    print(f"{NOME} em http://127.0.0.1:{a.porta}/v1/systemone", flush=True)
    # thread única: o MLX amarra o stream à thread que carregou o modelo, e as passadas já são sequenciais
    HTTPServer(("127.0.0.1", a.porta), H).serve_forever()


if __name__ == "__main__":
    main()
