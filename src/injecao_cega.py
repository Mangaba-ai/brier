"""Injeção com frases INÉDITAS escritas pelo mangaba-titan — teste cego da defesa (modelos e regras).

As famílias G–J de robustez.py foram escritas por nós; este teste usa frases geradas pelo Titan com outro
enunciado (dados/injecao_titan_teste.json), sem que as regras de brier/regras.py tenham sido ajustadas
nelas (o lote de ajuste foi outro: dados/injecao_titan_frases.json).

  python src/injecao_cega.py gera                 # dados/adv_teste_injecao_titan.jsonl + pedidos do Jev
  python src/injecao_cega.py avalia --versoes v3 comite v3+neutraliza
"""
import argparse
import copy
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from adversarial import _alvo_ataque, _insere  # noqa: E402
from comparar_jev import para_itens  # noqa: E402
from robustez import avalia, fonte, h, le, limpos  # noqa: E402

TESTE = Path("dados/adv_teste_injecao_titan.jsonl")


def gera(n_por_fonte=50):
    rnd = random.Random(23)
    frases = json.loads(Path("dados/injecao_titan_teste.json").read_text(encoding="utf-8"))
    base = limpos()
    por = {}
    for e in base.values():
        por.setdefault(fonte(e["id"]), []).append(e)
    exs = []
    for f in sorted(por):
        for ex in rnd.sample(por[f], min(n_por_fonte, len(por[f]))):
            k = rnd.choice(list(ex["questions"]))
            alvo, escrito = _alvo_ataque(ex["questions"][k], ex["rotulos"][k]["ouro"], rnd)
            i = rnd.randrange(len(frases))
            novo = copy.deepcopy(ex)
            novo["state"] = _insere(ex["state"], frases[i].replace("{ALVO}", str(escrito)), rnd)
            novo["id"] = f"{ex['id']}~titan{i}"
            novo["ataque"] = {"pergunta": k, "alvo": alvo, "familia": f"titan{i:02d}", "id_limpo": ex["id"]}
            exs.append(novo)
    TESTE.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in exs), encoding="utf-8")
    cache = {json.loads(l)["h"] for l in Path("dados/jev_cache.jsonl").read_text(encoding="utf-8").splitlines()}
    falta = [e for e in exs + [base[e["ataque"]["id_limpo"]] for e in exs] if h(e) not in cache]
    with open("execucoes/jev/pedidos.jsonl", "w", encoding="utf-8") as f:
        for e in {h(e): e for e in falta}.values():
            f.write(json.dumps({"h": h(e), "corpo": {"model": "jev-latest", "state": e["state"],
                                                     "questions": e["questions"]}}, ensure_ascii=False) + "\n")
    print(len(exs), "exemplos;", len({h(e) for e in falta}), "pedidos novos ao Jev")


def _versao(nome):
    from brier import Brier
    neutr = nome.endswith("+neutraliza")
    return Brier(nome.replace("+neutraliza", ""), neutralizar_injecao=neutr)


def _sucessos(exs, itens_inj):
    import numpy as np
    from dados import indice_ouro
    out = []
    for e, ia in zip(exs, itens_inj):
        k = list(e["questions"]).index(e["ataque"]["pergunta"])
        out.append(int(np.argmax(ia[k]["p"]) == indice_ouro(e["questions"][e["ataque"]["pergunta"]], e["ataque"]["alvo"])))
    return np.array(out)


def _ic_contra_jev(m, j, n=5000):
    """IC95 pareado (bootstrap) da diferença de taxa de ataque, modelo − Jev (negativo = mais resistente)."""
    import numpy as np
    rnd = np.random.default_rng(0)
    d = [(m[s] - j[s]).mean() for s in (rnd.integers(0, len(m), len(m)) for _ in range(n))]
    return [round(float(x), 4) for x in np.percentile(d, [2.5, 97.5])]


def avalia_versoes(versoes, saida):
    from brier.regras import procura_injecao
    exs = le(TESTE)
    base = limpos()
    lim = [base[e["ataque"]["id_limpo"]] for e in exs]
    det = [bool(procura_injecao(e["state"])) for e in exs]
    rel = {"n": len(exs), "regras_detectaram": round(sum(det) / len(det), 4)}
    cache = {}
    for l in Path("dados/jev_cache.jsonl").read_text(encoding="utf-8").splitlines():
        x = json.loads(l)
        cache[x["h"]] = x["r"]
    ok = [i for i in range(len(exs)) if h(exs[i]) in cache and h(lim[i]) in cache]
    if ok:
        rel["jev"] = avalia([para_itens(lim[i], cache[h(lim[i])]) for i in ok],
                            [para_itens(exs[i], cache[h(exs[i])]) for i in ok], [exs[i] for i in ok])
        rel["jev"]["n_pares"] = len(ok)
        print("jev", rel["jev"]["taxa_ataque"], len(ok), flush=True)
    memo = {}
    for v in versoes:
        b = _versao(v)
        chave_limpo = v.replace("+neutraliza", "")  # sem injeção as regras quase nunca disparam
        if chave_limpo not in memo:
            memo[chave_limpo] = [para_itens(e, b.decide(e["state"], e["questions"])) for e in lim]
        ia = [para_itens(e, b.decide(e["state"], e["questions"])) for e in exs]
        rel[v] = avalia(memo[chave_limpo], ia, exs)
        rel[v]["n_pares"] = len(exs)
        if len(ok) == len(exs):
            jev = _sucessos(exs, [para_itens(e, cache[h(e)]) for e in exs])
            rel[v]["ic95_taxa_ataque_menos_jev"] = _ic_contra_jev(_sucessos(exs, ia), jev)
        print(v, rel[v]["taxa_ataque"], rel[v]["acc_limpo"]["acc"], rel[v]["acc_alterado"]["acc"], flush=True)
        del b
    Path(saida).write_text(json.dumps(rel, indent=1, ensure_ascii=False), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("acao", choices=["gera", "avalia"])
    ap.add_argument("--versoes", nargs="*", default=["v3", "comite", "v3+neutraliza"])
    ap.add_argument("--saida", default="execucoes/injecao_cega.json")
    a = ap.parse_args()
    gera() if a.acao == "gera" else avalia_versoes(a.versoes, a.saida)


if __name__ == "__main__":
    main()
