"""Relay stdlib: lê pedidos.jsonl, chama o Jev e grava respostas.jsonl no formato do cache."""
import json, os, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
CHAVE = open(os.path.expanduser("~/jevcmp/.chave")).read().strip()
URL = "https://api.typesafe.ai/v1/systemone"
def chama(linha):
    d = json.loads(linha)
    corpo = json.dumps(d["corpo"]).encode()
    for t in range(4):
        req = urllib.request.Request(URL, data=corpo, headers={"Authorization": f"Bearer {CHAVE}", "Content-Type": "application/json"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return {"h": d["h"], "r": {"answers": json.load(r)["answers"], "latencia": time.time() - t0}}
        except urllib.error.HTTPError as e:
            if e.code in (429, 529, 500, 502, 503):
                time.sleep(2 * (t + 1)); continue
            return {"h": d["h"], "erro": f"{e.code} {e.read()[:300].decode(errors='replace')}"}
        except Exception as e:
            time.sleep(2 * (t + 1)); err = str(e)
    return {"h": d["h"], "erro": "falhou 4x"}
linhas = open(os.path.expanduser("~/jevcmp/pedidos.jsonl")).read().splitlines()
with ThreadPoolExecutor(8) as ex, open(os.path.expanduser("~/jevcmp/respostas.jsonl"), "w") as f:
    for i, r in enumerate(ex.map(chama, linhas), 1):
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if i % 100 == 0: print(i, flush=True)
print("fim", len(linhas))
