"""Partida de Tetris jogada pelo Jev (roda na VPS; só biblioteca padrão)."""
import json, os, sys, time, urllib.request, urllib.error
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tetris
CHAVE = open(os.path.expanduser("~/jevtetris/.chave")).read().strip()

def decide(estado, q):
    corpo = json.dumps({"model": "jev-latest", "state": estado, "questions": q}).encode()
    for t in range(6):
        req = urllib.request.Request("https://api.typesafe.ai/v1/systemone", data=corpo,
                                     headers={"Authorization": f"Bearer {CHAVE}", "Content-Type": "application/json"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                a = json.load(r)["answers"]["jogada"]
                return a["probabilities"], time.time() - t0
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 529):
                time.sleep(2 * (t + 1)); continue
            raise
        except Exception:
            time.sleep(2 * (t + 1))
    raise RuntimeError("Jev não respondeu")

semente = int(sys.argv[1]) if len(sys.argv) > 1 else 7
r = tetris.joga(decide, semente=semente, max_pecas=150,
                registro=lambda i, p: print(i, p, flush=True) if i % 10 == 0 else None)
r["modelo"] = "jev"
json.dump(r, open(os.path.expanduser(f"~/jevtetris/jev_{semente}.json"), "w"))
print({k: v for k, v in r.items() if k not in ("passos", "sequencia")})
