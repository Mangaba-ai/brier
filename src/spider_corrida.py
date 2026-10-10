"""Corrida da página Spider, passo 2: cada modelo decide, link a link, se o link nomeia uma aranha.

Modos: um link por chamada, ou vários links por chamada (uma pergunta noul por link, que o Brier
responde numa única passada). Para cada modo: corrida de 30 s e página inteira (todos os links).
Gabarito: execucoes/spider/links.json (src/spider_gabarito.py, Wikidata). Achado = aranha decidida como aranha (p >= 0,5).

Uso: python src/spider_corrida.py v3|v2|rapido [en]   ("rapido" = pesos do Laya, carregados direto)   ("en" = pergunta em inglês, só 1 link por chamada)
"""
import json
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from brier import Brier  # noqa: E402

AQUI = RAIZ / "execucoes/spider"
LINKS = json.load(open(AQUI / "links.json"))
LIMITE = 30.0
CRITERIOS = {"true": "Sim: é uma aranha ou um grupo de aranhas (espécie, gênero, família, subordem)",
             "false": "Não: outro animal, outro grupo, conceito, lugar, pessoa ou termo"}
INSTRUCAO = "O link nomeia uma aranha?"
INGLES = len(sys.argv) > 2 and sys.argv[2] == "en"
if INGLES:
    CRITERIOS = {"true": "Yes: it is a spider or a group of spiders (species, genus, family, suborder)",
                 "false": "No: another animal, another group, a concept, place, person or term"}
    INSTRUCAO = "Does the link name a spider?"


def pergunta_unica(l):
    estado = (f"Link on the Wikipedia page “Spider”. Link text: “{l['texto']}”. Article: “{l['titulo']}”." if INGLES else
              f"Link na página “Spider” da Wikipédia em inglês. Texto do link: “{l['texto']}”. Artigo: “{l['titulo']}”.")
    return estado, {"aranha": {"type": "noul", "instructions": INSTRUCAO, "criteria": CRITERIOS}}


def perguntas_varias(lote):
    estado = "Links da página “Spider” da Wikipédia em inglês. Cada pergunta cita um link."
    qs = {f"l{i}": {"type": "noul", "instructions": f"O link “{l['texto']}” (artigo “{l['titulo']}”) nomeia uma aranha?",
                    "criteria": CRITERIOS} for i, l in enumerate(lote)}
    return estado, qs


def roda(b, k, limite=None):
    """Decide os links em ordem, k por chamada. Devolve [(índice, p_sim, t_fim)]."""
    out, t0 = [], time.perf_counter()
    for i in range(0, len(LINKS), k):
        if limite and time.perf_counter() - t0 >= limite:
            break
        lote = LINKS[i:i + k]
        if k == 1:
            estado, qs = pergunta_unica(lote[0])
            r = b.decide(estado, qs)
            ps = [r["answers"]["aranha"]["noul"]]
        else:
            estado, qs = perguntas_varias(lote)
            r = b.decide(estado, qs)
            ps = [r["answers"][f"l{j}"]["noul"] for j in range(len(lote))]
        t = time.perf_counter() - t0
        out += [(i + j, p, t) for j, p in enumerate(ps)]
    return out


def placar(res, limite=None):
    if limite:
        res = [x for x in res if x[2] <= limite]
    tp = sum(1 for i, p, _ in res if p >= 0.5 and LINKS[i]["aranha"])
    fp = sum(1 for i, p, _ in res if p >= 0.5 and not LINKS[i]["aranha"])
    fn = sum(1 for i, p, _ in res if p < 0.5 and LINKS[i]["aranha"])
    acertos = sum(1 for i, p, _ in res if (p >= 0.5) == LINKS[i]["aranha"])
    brier = sum((p - LINKS[i]["aranha"]) ** 2 for i, p, _ in res) / max(1, len(res))
    return {"checados": len(res), "achados": tp, "alarmes_falsos": fp, "aranhas_perdidas": fn,
            "acuracia": round(acertos / max(1, len(res)), 3), "precisao": round(tp / max(1, tp + fp), 3),
            "cobertura": round(tp / max(1, tp + fn), 3), "escore_brier": round(brier, 3),
            "ms_por_link": round(1000 * res[-1][2] / len(res), 1) if res else None}


class Laya:
    """Pesos do Laya, carregados direto (não fazem parte do pacote brier); mesma interface de decide()."""

    def __init__(self):
        import laya
        self.agente = laya.load("convaiinnovations/laya-multilingual")

    def decide(self, state, questions):
        return self.agente.predict(state, questions)


def main():
    modelo = sys.argv[1]
    t = time.time()
    b = Laya() if modelo == "rapido" else Brier(modelo)
    print(f"{modelo}: carregado em {time.time() - t:.1f}s", flush=True)
    ks = [1] if INGLES else [1, 8, 16, 32]
    saida = {}
    for k in ks:
        roda(b, k, limite=0.001)  # aquecimento (primeira chamada)
        corrida = roda(b, k, limite=LIMITE)
        pagina = roda(b, k)
        saida[f"k{k}"] = {"corrida": placar(corrida, LIMITE), "pagina": placar(pagina),
                          "decisoes": [(i, round(p, 4), round(tt, 3)) for i, p, tt in corrida]}
        c, pg = saida[f"k{k}"]["corrida"], saida[f"k{k}"]["pagina"]
        print(f"  {k:2d} por chamada | 30 s: {c['achados']} achados, {c['alarmes_falsos']} falsos, {c['checados']} checados "
              f"({c['ms_por_link']} ms/link) | página: acurácia {pg['acuracia']}, precisão {pg['precisao']}, "
              f"cobertura {pg['cobertura']}, Brier {pg['escore_brier']}", flush=True)
    json.dump(saida, open(AQUI / f"resultado_{modelo}{'_en' if INGLES else ''}.json", "w"), indent=1)


main()
