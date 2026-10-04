"""Servidor HTTP do Brier v4 (biblioteca padrão; Windows, Linux e macOS).

  POST /v1/decide        {"state", "questions", "min_confidence"?, "longo"?: "auto"|"sim"|"nao"}
  POST /v1/decide/lote   {"states": [...], "questions", "min_confidence"?}
  POST /v1/systemone     mesmo corpo do /v1/decide (compatibilidade para quem migra)
  GET  /v1/models        GET /saude

Uso: brier-serve --modelo execucoes/v4 --porta 8790 [--auditoria auditoria.jsonl] [--mascarar-pii]
"""
import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .motor import MODELO_PADRAO, MODELO_OUTROS_IDIOMAS, Brier


def valida_pergunta(q) -> bool:
    if not isinstance(q, dict) or not isinstance(q.get("instructions"), (str, dict, list)):
        return False
    t, c = q.get("type"), q.get("criteria")
    if t == "choice":
        return isinstance(c, dict) and 2 <= len(c) <= 255
    if t == "score":
        return isinstance(c, list) and 2 <= len(c) <= 10
    if t == "noul":
        return isinstance(c, dict) and {"true", "false"} <= set(map(str, c))
    return False


def _valida(questions):
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions precisa ser um objeto com ao menos uma pergunta")
    ruins = [k for k, q in questions.items() if not valida_pergunta(q)]
    if ruins:
        raise ValueError(f"perguntas inválidas: {ruins} (confira type e criteria)")


def cria_handler(brier: Brier):
    class H(BaseHTTPRequestHandler):
        def _json(self, cod, obj):
            b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(cod)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            caminho = self.path.rstrip("/")
            if caminho == "/v1/models":
                return self._json(200, {"models": [{"name": "brier-4", "pesos": brier.nome_modelo}]})
            if caminho == "/saude":
                return self._json(200, {"ok": True})
            self._json(404, {"detail": "rota inexistente"})

        def do_POST(self):
            caminho = self.path.rstrip("/")
            if caminho not in ("/v1/decide", "/v1/systemone", "/v1/decide/lote"):
                return self._json(404, {"detail": "rota inexistente"})
            try:
                corpo = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))).decode("utf-8"))
                if not isinstance(corpo, dict):
                    raise ValueError("o corpo precisa ser um objeto JSON")
                _valida(corpo.get("questions"))
                mc = corpo.get("min_confidence")
                if mc is not None and not 0 <= float(mc) <= 1:
                    raise ValueError("min_confidence deve ficar entre 0 e 1")
                if caminho == "/v1/decide/lote":
                    states = corpo.get("states")
                    if not isinstance(states, list) or not states or len(states) > 256:
                        raise ValueError("states precisa ser uma lista com 1 a 256 itens")
                elif "state" not in corpo:
                    raise ValueError("corpo precisa de state")
            except (ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as e:
                return self._json(422, {"detail": [{"msg": str(e)}]})
            if caminho == "/v1/decide/lote":
                return self._json(200, {"resultados": brier.decide_lote(corpo["states"], corpo["questions"], min_confidence=mc)})
            self._json(200, brier.decide(corpo["state"], corpo["questions"], min_confidence=mc,
                                         longo=corpo.get("longo", "auto")))

        def log_message(self, *args):
            pass

    return H


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Servidor do Brier v4")
    ap.add_argument("--modelo", default=MODELO_PADRAO, help="pasta local ou repositório no Hugging Face")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--porta", type=int, default=8790)
    ap.add_argument("--dispositivo", default=None, help="cuda, mps ou cpu (padrão: o melhor disponível)")
    ap.add_argument("--outros-idiomas", action="store_true",
                    help=f"roteia textos fora do português para {MODELO_OUTROS_IDIOMAS}")
    ap.add_argument("--auditoria", default=None, help="arquivo JSONL de auditoria (sem o texto do state)")
    ap.add_argument("--mascarar-pii", action="store_true", help="mascara CPF, CNPJ, cartão, e-mail, telefone e CEP")
    a = ap.parse_args()
    b = Brier(a.modelo, dispositivo=a.dispositivo, mascarar_pii=a.mascarar_pii, auditoria=a.auditoria,
              modelo_outros_idiomas=MODELO_OUTROS_IDIOMAS if a.outros_idiomas else None)
    print(f"brier-4 em http://{a.host}:{a.porta}/v1/decide (pesos: {a.modelo})", flush=True)
    ThreadingHTTPServer((a.host, a.porta), cria_handler(b)).serve_forever()


if __name__ == "__main__":
    main()
