"""Corrida da página Spider, passo 1: links da página "Spider" (Wikipédia em inglês), na ordem, com gabarito pelo Wikidata.

Um link "nomeia uma aranha" quando o item do Wikidata descende de Araneae (Q1357) pela árvore
taxonômica (P171 = táxon superior), excluindo a própria Araneae.

Uso: python src/spider_gabarito.py   (grava execucoes/spider/links.json)
"""
import json
from pathlib import Path
import re
import time
import urllib.parse
import urllib.request

UA = {"User-Agent": "brier-spider/1.0 (https://github.com/Mangaba-ai/brier)"}


def get(url):
    req = urllib.request.Request(url, headers=UA)
    return json.load(urllib.request.urlopen(req, timeout=60))


html = get("https://en.wikipedia.org/w/api.php?action=parse&page=Spider&prop=text&format=json&formatversion=2&redirects=1")["parse"]["text"]
# corta referências e caixas de navegação do fim
html = re.split(r'<h2[^>]*id="(References|Notes|Further_reading|External_links)"', html)[0]
links, vistos = [], set()
for m in re.finditer(r'<a href="/wiki/([^"#?]+)"[^>]*>(.*?)</a>', html):
    titulo = urllib.parse.unquote(m.group(1)).replace("_", " ")
    texto = re.sub(r"<[^>]+>", "", m.group(2)).strip()
    if ":" in titulo or not texto or titulo in vistos:
        continue
    vistos.add(titulo)
    links.append({"titulo": titulo, "texto": texto})

# QIDs (seguindo redirecionamentos)
qid = {}
for i in range(0, len(links), 50):
    lote = [l["titulo"] for l in links[i:i + 50]]
    q = get("https://en.wikipedia.org/w/api.php?action=query&prop=pageprops&ppprop=wikibase_item&redirects=1&format=json&formatversion=2&titles="
            + urllib.parse.quote("|".join(lote)))["query"]
    destino = {t: t for t in lote}
    for r in q.get("normalized", []) + q.get("redirects", []):
        for t, d in list(destino.items()):
            if d == r["from"]:
                destino[t] = r["to"]
    por_titulo = {p["title"]: p.get("pageprops", {}).get("wikibase_item") for p in q["pages"]}
    for t in lote:
        qid[t] = por_titulo.get(destino[t])
    time.sleep(0.2)

ids = sorted({v for v in qid.values() if v})
pais = {}


def busca_pais(lista):
    falta = [x for x in lista if x not in pais]
    for k in range(0, len(falta), 50):
        lote = falta[k:k + 50]
        r = get("https://www.wikidata.org/w/api.php?action=wbgetentities&props=claims&format=json&ids=" + "|".join(lote))
        for x in lote:
            ent = r["entities"].get(x, {})
            pais[x] = [c["mainsnak"]["datavalue"]["value"]["id"] for c in ent.get("claims", {}).get("P171", [])
                       if c["mainsnak"].get("datavalue")]
        time.sleep(0.1)


# sobe a árvore (P171) até Araneae; memoriza o veredito de cada item
veredito = {"Q1357": True}
fronteira = set(ids)
for _ in range(40):
    busca_pais(sorted(fronteira))
    fronteira = {p for x in fronteira for p in pais[x] if p not in pais and p not in veredito}
    if not fronteira:
        break


def eh_aranha(x, vistos=()):
    if x in veredito:
        return veredito[x]
    if x in vistos or x not in pais:
        return False
    v = any(eh_aranha(p, vistos + (x,)) for p in pais[x])
    veredito[x] = v
    return v


aranhas = {x for x in ids if x != "Q1357" and eh_aranha(x)}

for l in links:
    l["qid"] = qid.get(l["titulo"])
    l["aranha"] = l["qid"] in aranhas
SAIDA = Path(__file__).resolve().parent.parent / "execucoes/spider/links.json"
SAIDA.parent.mkdir(parents=True, exist_ok=True)
json.dump(links, open(SAIDA, "w"), ensure_ascii=False, indent=1)
n = sum(l["aranha"] for l in links)
print(f"{len(links)} links únicos, {n} nomeiam aranhas")
print("aranhas:", [l["texto"] for l in links if l["aranha"]][:40])
print("primeiros links:", [l["texto"] for l in links[:25]])
