"""Conjuntos de avaliação com rótulo HUMANO, fora da distribuição do gerador (nunca entram no treino).

- ASSIN2 (NILC): pares de frases → implicação (noul) + similaridade 1–5 (score de 5 níveis).
- B2W-Reviews01 (Americanas): avaliação de produto → nota 1–5 (score) + recomendaria (noul).
Cada linha vira um pedido no formato Jev com `rotulos[chave].ouro`, igual ao sintético.
"""
import io
import json
import random
from pathlib import Path

import httpx
import pandas as pd

URLS = {
    "assin2": "https://huggingface.co/datasets/nilc-nlp/assin2/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet",
    "b2w": "https://huggingface.co/datasets/ruanchaves/b2w-reviews01/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
}

Q_ASSIN = {
    "implica": {"type": "noul",
                "instructions": "Se o texto A for verdadeiro, o texto B também é necessariamente verdadeiro?",
                "criteria": {"true": "A implica B: B decorre de A", "false": "B não decorre de A (é neutro ou contradiz)"}},
    "similaridade": {"type": "score", "instructions": "Quão parecidos em significado são os textos A e B?",
                     "criteria": ["Completamente diferentes, sem relação de sentido",
                                  "Pouco relacionados: compartilham só um detalhe ou tema",
                                  "Moderadamente parecidos: mesmo tema, mas diferem em informação importante",
                                  "Muito parecidos: diferem só em detalhes secundários",
                                  "Equivalentes: dizem a mesma coisa com outras palavras"]},
}

Q_B2W = {
    "nota": {"type": "score", "instructions": "Que nota (de 1 a 5 estrelas) o cliente deu ao produto?",
             "criteria": ["1 estrela: péssimo, muito insatisfeito", "2 estrelas: ruim, insatisfeito",
                          "3 estrelas: regular, satisfação mista", "4 estrelas: bom, satisfeito com ressalvas",
                          "5 estrelas: excelente, muito satisfeito"]},
    "recomenda": {"type": "noul", "instructions": "O cliente recomendaria este produto a um amigo?",
                  "criteria": {"true": "Recomendaria", "false": "Não recomendaria"}},
}


def _parquet(nome):
    cache = Path(f"dados/{nome}.parquet")
    if not cache.exists():
        r = httpx.get(URLS[nome], follow_redirects=True, timeout=300)
        r.raise_for_status()
        cache.write_bytes(r.content)
    return pd.read_parquet(io.BytesIO(cache.read_bytes()))


def assin2(n=400, semente=0):
    df = _parquet("assin2").sample(n, random_state=semente)
    out = []
    for _, r in df.iterrows():
        nivel = int(round(float(r.relatedness_score))) - 1
        out.append({"id": f"assin2-{r.sentence_pair_id}", "fonte": "assin2",
                    "state": {"texto_A": r.premise, "texto_B": r.hypothesis}, "questions": Q_ASSIN,
                    "rotulos": {"implica": {"ouro": bool(r.entailment_judgment == 1)},
                                "similaridade": {"ouro": max(0, min(4, nivel))}}})
    return out


def b2w(n=400, semente=0):
    df = _parquet("b2w")
    df = df[df.review_text.notna() & df.recommend_to_a_friend.isin(["Yes", "No"])]
    # estratifica por nota para não virar "tudo 5 estrelas"
    df = pd.concat([df[df.overall_rating == k].sample(n // 5, random_state=semente) for k in range(1, 6)])
    out = []
    for i, r in df.iterrows():
        texto = f"{r.review_title}\n{r.review_text}" if r.review_title else r.review_text
        out.append({"id": f"b2w-{i}", "fonte": "b2w", "state": texto, "questions": Q_B2W,
                    "rotulos": {"nota": {"ouro": int(r.overall_rating) - 1},
                                "recomenda": {"ouro": r.recommend_to_a_friend == "Yes"}}})
    random.Random(semente).shuffle(out)
    return out


if __name__ == "__main__":
    Path("dados").mkdir(exist_ok=True)
    for nome, fn in (("assin2", assin2), ("b2w", b2w)):
        exs = fn()
        with open(f"dados/ext_{nome}.jsonl", "w", encoding="utf-8") as f:
            for e in exs:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(nome, len(exs))
