"""Conjuntos públicos em PT-BR com rótulo HUMANO, convertidos para o formato Brier (v2).

TREINO (nunca usados na avaliação):
  - ASSIN v1 (NILC, notícias PT-BR/PT-PT): implicação/paráfrase (choice) + similaridade 1–5 (score)
  - FaQuAD-NLI: a frase responde à pergunta? (noul)
  - HateBR (Vargas et al.): comentário ofensivo? (noul) — 3 anotadores → alvo suave de verdade
AVALIAÇÃO (zero-shot para Brier e Jev): tweetSentBR (sentimento), além de ASSIN2 e B2W.

Cada exemplo recebe uma formulação sorteada entre várias (instrução, chaves, descrições), para o
modelo aprender a tarefa e não uma frase fixa. Os ids começam com o nome do conjunto e o número
da divisão segue a mesma regra do sintético (id % 10: 0 = teste, 1 = val, resto = treino).

Uso: python src/publicos.py
"""
import io
import json
import random
from pathlib import Path

import httpx
import pandas as pd

HF = "https://huggingface.co/datasets/{}/resolve/refs%2Fconvert%2Fparquet/{}/0000.parquet"


def _pq(repo, caminho, nome):
    cache = Path(f"dados/pub_{nome}.parquet")
    if not cache.exists():
        r = httpx.get(HF.format(repo, caminho), follow_redirects=True, timeout=300)
        r.raise_for_status()
        cache.write_bytes(r.content)
    return pd.read_parquet(io.BytesIO(cache.read_bytes()))


def _suave(k_opcoes, idx, p=0.9):
    return [p if i == idx else (1 - p) / (k_opcoes - 1) for i in range(k_opcoes)]


# ---------- ASSIN v1 ----------
Q_IMPLICA = [
    ("Qual a relação lógica entre o texto A e o texto B?",
     {"implica": "A garante que B é verdadeiro, mas não o contrário", "parafrase": "A e B dizem a mesma coisa (um implica o outro)",
      "nenhuma": "B não decorre de A: é neutro, fala de outra coisa ou contradiz"}),
    ("Classifique o par de frases.",
     {"acarreta": "Se A é verdade, B também é", "equivalentes": "As duas frases têm o mesmo sentido",
      "sem_relacao": "A não permite concluir B"}),
    ("O que o texto A permite afirmar sobre o texto B?",
     {"b_decorre": "B é consequência de A", "mesmo_sentido": "A e B são paráfrases",
      "nao_decorre": "Não dá para concluir B a partir de A"}),
]
Q_SIMIL = [
    ("Quão parecidos em significado são os textos A e B?",
     ["Sem relação de sentido", "Pouca relação, só o tema", "Relacionados, mas com diferenças importantes",
      "Muito parecidos, diferem em detalhes", "Mesmo significado"]),
    ("Grau de similaridade semântica entre as duas frases (1 a 5).",
     ["1: completamente diferentes", "2: compartilham um elemento", "3: parcialmente equivalentes",
      "4: quase equivalentes", "5: equivalentes"]),
]
ORDEM_ASSIN = ["nenhuma", "implica", "parafrase"]  # entailment_judgment 0,1,2


def assin(rnd):
    df = _pq("nilc-nlp/assin", "full/train", "assin_train")
    out = []
    for _, r in df.iterrows():
        instr, crit = rnd.choice(Q_IMPLICA)
        chaves = list(crit)
        # mapeia a ordem canônica (nenhuma, implica, parafrase) para as chaves desta formulação
        mapa = {"implica": chaves[0], "parafrase": chaves[1], "nenhuma": chaves[2]}
        ouro = mapa[ORDEM_ASSIN[int(r.entailment_judgment)]]
        s_instr, s_crit = rnd.choice(Q_SIMIL)
        nivel = max(0, min(4, int(round(float(r.relatedness_score))) - 1))
        qs = {"relacao": {"type": "choice", "instructions": instr, "criteria": crit},
              "similaridade": {"type": "score", "instructions": s_instr, "criteria": s_crit}}
        alvo_s = [0.0] * 5
        for i in range(5):  # nota média de vários anotadores: espalha pela distância
            alvo_s[i] = max(0.0, 1 - abs(i - (float(r.relatedness_score) - 1)))
        s = sum(alvo_s)
        out.append({"id": f"assin-{r.sentence_pair_id}", "fonte": "assin",
                    "state": {"texto_A": r.premise, "texto_B": r.hypothesis}, "questions": qs,
                    "rotulos": {"relacao": {"ouro": ouro, "alvo": _suave(3, chaves.index(ouro))},
                                "similaridade": {"ouro": nivel, "alvo": [x / s for x in alvo_s]}}})
    return out


# ---------- FaQuAD-NLI ----------
Q_FAQ = [
    ("A frase responde à pergunta?", {"true": "Sim, a frase contém a resposta", "false": "Não, a frase não responde"}),
    ("O trecho traz a resposta para a pergunta feita?", {"true": "Traz a resposta", "false": "Não traz"}),
    ("Esta sentença é a resposta da pergunta?", {"true": "É a resposta", "false": "Não é"}),
]


def faquad(rnd):
    df = _pq("ruanchaves/faquad-nli", "default/train", "faquad_train")
    out = []
    for i, r in df.iterrows():
        instr, crit = rnd.choice(Q_FAQ)
        out.append({"id": f"faquad-{i}", "fonte": "faquad",
                    "state": {"pergunta": r.question, "frase": r.answer},
                    "questions": {"responde": {"type": "noul", "instructions": instr, "criteria": crit}},
                    "rotulos": {"responde": {"ouro": bool(r.label == 1), "alvo": _suave(2, 0 if r.label == 1 else 1)}}})
    return out


# ---------- HateBR ----------
Q_HATE = [
    ("O comentário é ofensivo?", {"true": "Ofensivo: insulta, xinga ou ataca alguém", "false": "Não ofensivo"}),
    ("Este comentário deveria ser moderado por linguagem ofensiva?", {"true": "Sim, é ofensivo", "false": "Não, é aceitável"}),
    ("Há linguagem ofensiva no texto?", {"true": "Há", "false": "Não há"}),
]


def hatebr(rnd):
    df = _pq("franciellevargas/HateBR", "default/train", "hatebr")
    out = []
    for _, r in df.iterrows():
        instr, crit = rnd.choice(Q_HATE)
        votos = [int(r.anotator1), int(r.anotator2), int(r.anotator3)]
        p = (sum(votos) + 0.25) / 3.5  # 3 anotadores humanos → probabilidade suave de verdade
        out.append({"id": f"hatebr-{r.id}", "fonte": "hatebr", "state": r.comentario,
                    "questions": {"ofensivo": {"type": "noul", "instructions": instr, "criteria": crit}},
                    "rotulos": {"ofensivo": {"ouro": bool(r.label_final == 1), "alvo": [p, 1 - p]}}})
    return out


# ---------- tweetSentBR (só avaliação) ----------
Q_TWEET = {"sentimento": {"type": "choice", "instructions": "Qual o sentimento do tweet?",
                          "criteria": {"positivo": "Elogio, alegria, aprovação", "neutro": "Informativo ou sem emoção clara",
                                       "negativo": "Crítica, raiva, tristeza, reclamação"}}}


def tweetsentbr(n=250, semente=0):
    df = _pq("eduagarcia/tweetsentbr_fewshot", "default/test", "tweetsentbr_test")
    mapa = {"Positive": "positivo", "Neutral": "neutro", "Negative": "negativo"}
    df = df[df.label.isin(mapa)]
    df = pd.concat([df[df.label == k].sample(n // 3 + 1, random_state=semente) for k in mapa])
    out = [{"id": f"tweet-{r.id}", "fonte": "tweetsentbr", "state": r.sentence, "questions": Q_TWEET,
            "rotulos": {"sentimento": {"ouro": mapa[r.label]}}} for _, r in df.iterrows()]
    random.Random(semente).shuffle(out)
    return out[:n]


if __name__ == "__main__":
    rnd = random.Random(7)
    treino = assin(rnd) + faquad(rnd) + hatebr(rnd)
    rnd.shuffle(treino)
    with open("dados/publicos_treino.jsonl", "w") as f:
        for e in treino:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    from collections import Counter
    print("treino:", len(treino), dict(Counter(e["fonte"] for e in treino)))
    tw = tweetsentbr()
    with open("dados/ext_tweetsentbr.jsonl", "w") as f:
        for e in tw:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print("tweetsentbr (avaliação):", len(tw), dict(Counter(e["rotulos"]["sentimento"]["ouro"] for e in tw)))
