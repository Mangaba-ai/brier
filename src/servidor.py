"""Servidor do Brier: POST /v1/decide (rota própria) e GET /v1/models.

Roda em macOS, Linux e Windows. O motor é escolhido sozinho:
  - MLX em Mac com Apple Silicon, se o mlx-lm estiver instalado (mais rápido);
  - PyTorch em qualquer outro caso (CPU, NVIDIA/CUDA ou Apple/MPS).
Force com --motor mlx|torch. Os dois produzem as mesmas respostas (ver tests/).

POST /v1/systemone aceita o MESMO corpo, só como rota de compatibilidade para quem migra de outro
serviço com esse formato. Cada resposta traz `incerteza` e, com calibração conformal, `conjunto`
(opções que cobrem a resposta certa com 90% de garantia). `amostras` (padrão 1) > 1 liga as camadas
estocásticas.

Uso: python src/servidor.py --execucao execucoes/v2 --porta 8790
"""
import argparse
import json
import platform
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nucleo import valida_pergunta  # noqa: E402

NOME = "brier-2"


def motor_padrao() -> str:
    if sys.platform == "darwin" and platform.machine() == "arm64":
        try:
            import mlx_lm  # noqa: F401
            return "mlx"
        except ImportError:
            pass
    return "torch"


def carrega_mlx(execucao: str, usar_swa: bool):
    import mlx.core as mx
    from mlx.utils import tree_unflatten
    from mlx_lm import load

    from decisor import Decisor
    from treinar import prepara_lora

    mx.set_cache_limit(1024 ** 3)  # sem teto o cache do MLX cresce a cada comprimento novo e leva a swap
    d = Path(execucao)
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    model, tok = load(cfg["modelo"])
    prepara_lora(model, cfg["rank"], cfg["dropout"], cfg["camadas"])
    if usar_swa and (d / "swa_media.safetensors").exists():
        model.update(tree_unflatten(list(mx.load(str(d / "swa_media.safetensors")).items())))
    else:
        model.load_weights(str(d / "adaptadores.safetensors"), strict=False)
    model.eval()
    cal_path = d / "calibracao.json"
    cal = json.loads(cal_path.read_text(encoding="utf-8")) if cal_path.exists() else {}
    return Decisor(model, tok, cal.get("temperaturas"), conformal=cal.get("conformal"))


def carrega_torch(execucao: str, dispositivo: str | None, dtype: str):
    from motor_torch import DecisorTorch
    return DecisorTorch(execucao, dispositivo=dispositivo, dtype=dtype)


def cria_handler(dec):
    class H(BaseHTTPRequestHandler):
        def _json(self, cod, obj):
            b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(cod)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path.rstrip("/") == "/v1/models":
                return self._json(200, {"models": [{"name": NOME}]})
            self._json(404, {"detail": "rota inexistente"})

        def do_POST(self):
            if self.path.rstrip("/") not in ("/v1/decide", "/v1/systemone"):
                return self._json(404, {"detail": "rota inexistente"})
            try:
                corpo = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))).decode("utf-8"))
                qs = corpo.get("questions") if isinstance(corpo, dict) else None
                if not isinstance(corpo, dict) or "state" not in corpo or not isinstance(qs, dict) or not qs:
                    raise ValueError("corpo precisa de state e questions")
                ruins = [k for k, q in qs.items() if not valida_pergunta(q)]
                if ruins:
                    raise ValueError(f"perguntas inválidas: {ruins} (confira type e criteria)")
                amostras = int(corpo.get("amostras", 1))
                if not 1 <= amostras <= 16:
                    raise ValueError("amostras deve ficar entre 1 e 16")
            except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as e:
                return self._json(422, {"detail": [{"msg": str(e)}]})
            t = time.time()
            r = dec.decide(corpo, amostras=amostras)
            r["model"], r["latencia_ms"] = NOME, round((time.time() - t) * 1000, 1)
            self._json(200, r)

        def log_message(self, *args):
            pass

    return H


def main():
    if hasattr(sys.stdout, "reconfigure"):  # console do Windows não é UTF-8 por padrão
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--execucao", default="execucoes/v2")
    ap.add_argument("--porta", type=int, default=8790)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--motor", choices=["auto", "mlx", "torch"], default="auto")
    ap.add_argument("--dispositivo", default=None, help="torch: cuda, mps ou cpu (padrão: o melhor disponível)")
    ap.add_argument("--dtype", default="auto", help="torch: bfloat16, float16 ou float32")
    ap.add_argument("--swa", action="store_true", help="mlx: usa a média SWA em vez do melhor checkpoint")
    a = ap.parse_args()
    motor = motor_padrao() if a.motor == "auto" else a.motor
    dec = carrega_mlx(a.execucao, a.swa) if motor == "mlx" else carrega_torch(a.execucao, a.dispositivo, a.dtype)
    onde = motor if motor == "mlx" else f"torch/{dec.dispositivo}"
    print(f"{NOME} ({onde}) em http://{a.host}:{a.porta}/v1/decide (compatível: /v1/systemone)", flush=True)
    # thread única: o MLX amarra o stream à thread que carregou o modelo, e as passadas já são sequenciais
    HTTPServer((a.host, a.porta), cria_handler(dec)).serve_forever()


if __name__ == "__main__":
    main()
