"""Rotulagem por maioria (v2): mais dois rotuladores cegos e independentes sobre cada exemplo.

Votos por pergunta: intenção do gerador (mimo-v2.6-pro) + rotulador cego v1 (mimo-v2.6-pro, T=0)
+ mimo-v2.5-pro + mimo-v2.6-flash com raciocínio. Modelos diferentes erram de jeitos diferentes,
então a maioria limpa o ruído que capava o v1 (só 79% de concordância entre 2 votos).

Agregação (em `alvo_maioria`):
  - sem maioria estrita (empate no topo) → pergunta descartada;
  - alvo = contagem de votos + prior 0,3 por opção, normalizada (4/4 → ~0,85–0,9; 3/4 → ~0,65);
    em score o prior vai só para os vizinhos do nível vencedor (erro de 1 nível é plausível);
  - `ouro` passa a ser a resposta da maioria (inclusive no teste: mede contra rótulo melhor).

Uso: python src/rotular.py dados/sintetico.jsonl dados/sintetico_v2.jsonl --paralelo 12
"""
import argparse
import json
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gerar import PROMPT_ROTULADOR, normaliza  # noqa: E402
from llm import chat, json_de  # noqa: E402
from decisor import ordem_canonica  # noqa: E402

ROTULADORES = [("mimo-v2.5-pro", "none", 800), ("mimo-v2.6-flash", None, 6000)]


def votos_extras(ex):
    st = ex["state"] if isinstance(ex["state"], str) else json.dumps(ex["state"], ensure_ascii=False, indent=1)
    perg = json.dumps(ex["questions"], ensure_ascii=False, indent=1)
    out = {}
    for modelo, rac, mt in ROTULADORES:
        try:
            r = json_de(chat(PROMPT_ROTULADOR.format(state=st, perguntas=perg), model=modelo, temperature=0.0,
                             max_tokens=mt, raciocinio=rac))
        except Exception:
            r = {}
        out[modelo] = {k: normaliza(q, r.get(k)) if isinstance(r, dict) else None for k, q in ex["questions"].items()}
    return out


def alvo_maioria(q, votos, prior=0.3):
    validos = [v for v in votos if v is not None]
    if len(validos) < 2:
        return None
    c = Counter(validos).most_common()
    if len(c) > 1 and c[0][1] == c[1][1]:
        return None
    venc = c[0][0]
    opcoes = ordem_canonica(q)
    massa = {o: float(sum(1 for v in validos if v == o)) for o in opcoes}
    if q["type"] == "score":
        for o in opcoes:
            if abs(o - venc) == 1:
                massa[o] += prior
        massa[venc] += prior
    else:
        for o in opcoes:  # massa total 2×prior repartida: não achata choice com muitas opções
            massa[o] += 2 * prior / len(opcoes)
    tot = sum(massa.values())
    return venc, [massa[o] / tot for o in opcoes], c[0][1] / len(validos)


def agrega(ex):
    if not ex.get("votos_extras"):  # sem votos extras (cota, falha): mantém o rótulo de 2 votos do v1
        return {k: v for k, v in ex.items() if k != "votos_extras"}
    novos, questions = {}, {}
    for k, r in ex["rotulos"].items():
        q = ex["questions"][k]
        votos = [r["ouro"], r.get("cega")] + [ex.get("votos_extras", {}).get(m, {}).get(k) for m, _, _ in ROTULADORES]
        a = alvo_maioria(q, votos)
        if a is None:
            continue
        venc, alvo, frac = a
        novos[k] = {"ouro": venc, "alvo": alvo, "votos": votos, "fracao_maioria": frac,
                    "ouro_gerador": r["ouro"], "concordou": frac == 1.0}
        questions[k] = q
    if not novos:
        return None
    return {**{k: v for k, v in ex.items() if k not in ("rotulos", "questions")}, "questions": questions,
            "rotulos": novos}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entrada")
    ap.add_argument("saida")
    ap.add_argument("--paralelo", type=int, default=12)
    ap.add_argument("--so-agregar", action="store_true", help="não chama API; agrega os votos já gravados")
    a = ap.parse_args()
    votos_path = Path(a.entrada).with_suffix(".votos.jsonl")
    feitos = {}
    if votos_path.exists():
        for l in votos_path.read_text(encoding="utf-8").splitlines():
            d = json.loads(l)
            feitos[d["id"]] = d["votos_extras"]
    exs = [json.loads(l) for l in Path(a.entrada).read_text(encoding="utf-8").splitlines() if l.strip()]
    faltam = [] if a.so_agregar else [e for e in exs if e["id"] not in feitos]
    print(f"{len(exs)} exemplos, {len(faltam)} a rotular", flush=True)
    trava = threading.Lock()
    with ThreadPoolExecutor(a.paralelo) as pool, votos_path.open("a", encoding="utf-8") as f:
        futs = {pool.submit(votos_extras, e): e["id"] for e in faltam}
        for i, fu in enumerate(as_completed(futs), 1):
            v = fu.result()
            with trava:
                feitos[futs[fu]] = v
                f.write(json.dumps({"id": futs[fu], "votos_extras": v}, ensure_ascii=False) + "\n")
                f.flush()
            if i % 200 == 0:
                print(f"{i}/{len(faltam)}", flush=True)
    n_perg = n_desc = 0
    with open(a.saida, "w", encoding="utf-8") as f:
        for e in exs:
            e["votos_extras"] = feitos.get(e["id"], {})
            n_perg += len(e["rotulos"])
            g = agrega(e)
            if g is None:
                n_desc += len(e["rotulos"])
                continue
            n_desc += len(e["rotulos"]) - len(g["rotulos"])
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    print(f"fim: {n_perg} perguntas, {n_desc} descartadas sem maioria ({n_desc / max(n_perg, 1):.1%})")


if __name__ == "__main__":
    main()
