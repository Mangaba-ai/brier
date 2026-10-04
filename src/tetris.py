"""Tetris como teste de decisão encadeada (só biblioteca padrão: roda igual no Mac e na VPS).

A cada peça, o modelo recebe um pedido no formato do Brier/Jev:
  state     = tabuleiro em texto (20×10), peça atual e próxima;
  questions = {"jogada": choice}, uma opção por posição final possível (rotação × coluna),
              cada uma descrita pelo efeito: linhas limpas, buracos novos, altura e irregularidade.
As peças vêm de um saco de 7 embaralhado com semente fixa, então os dois modelos recebem exatamente a
mesma sequência. O Brier aceita até 26 opções por pergunta; quando há mais (T, L, J), ficam as 26 de
menor altura final, para os dois modelos igualmente.
"""
import json
import random

LARG, ALT = 10, 20
PECAS = {
    "I": [(0, 0), (0, 1), (0, 2), (0, 3)],
    "O": [(0, 0), (0, 1), (1, 0), (1, 1)],
    "T": [(0, 0), (0, 1), (0, 2), (1, 1)],
    "S": [(0, 1), (0, 2), (1, 0), (1, 1)],
    "Z": [(0, 0), (0, 1), (1, 1), (1, 2)],
    "J": [(0, 0), (1, 0), (1, 1), (1, 2)],
    "L": [(0, 2), (1, 0), (1, 1), (1, 2)],
}
PONTOS = {0: 0, 1: 100, 2: 300, 3: 500, 4: 800}
LETRAS = [chr(ord("a") + i) for i in range(26)]


def _normaliza(cel):
    mr, mc = min(r for r, _ in cel), min(c for _, c in cel)
    return tuple(sorted((r - mr, c - mc) for r, c in cel))


def rotacoes(p):
    """Rotações distintas da peça (formato canônico, linha 0 = topo)."""
    vistas, out, cel = set(), [], PECAS[p]
    for _ in range(4):
        n = _normaliza(cel)
        if n not in vistas:
            vistas.add(n)
            out.append(n)
        cel = [(c, -r) for r, c in cel]
    return out


def sequencia(semente, n):
    rnd, seq = random.Random(semente), []
    while len(seq) < n:
        saco = list(PECAS)
        rnd.shuffle(saco)
        seq += saco
    return seq[:n]


def vazio():
    return [[0] * LARG for _ in range(ALT)]


def _cabe(tab, forma, r0, c0):
    for r, c in forma:
        rr, cc = r0 + r, c0 + c
        if cc < 0 or cc >= LARG or rr >= ALT or (rr >= 0 and tab[rr][cc]):
            return False
    return True


def solta(tab, forma, c0):
    """Linha onde a forma para ao cair na coluna c0, ou None se não cabe nem no topo."""
    if not _cabe(tab, forma, 0, c0):
        return None
    r = 0
    while _cabe(tab, forma, r + 1, c0):
        r += 1
    return r


def aplica(tab, forma, r0, c0, marca=1):
    novo = [linha[:] for linha in tab]
    for r, c in forma:
        novo[r0 + r][c0 + c] = marca
    cheias = [i for i, linha in enumerate(novo) if all(linha)]
    novo = [linha for i, linha in enumerate(novo) if i not in cheias]
    return [[0] * LARG for _ in cheias] + novo, len(cheias)


def alturas(tab):
    h = []
    for c in range(LARG):
        r = next((r for r in range(ALT) if tab[r][c]), ALT)
        h.append(ALT - r)
    return h


def buracos(tab):
    n = 0
    for c in range(LARG):
        visto = False
        for r in range(ALT):
            if tab[r][c]:
                visto = True
            elif visto:
                n += 1
    return n


def jogadas(tab, peca):
    """Todas as posições finais possíveis, com o efeito de cada uma."""
    b0, out = buracos(tab), []
    for ri, forma in enumerate(rotacoes(peca)):
        larg = max(c for _, c in forma) + 1
        for c0 in range(LARG - larg + 1):
            r0 = solta(tab, forma, c0)
            if r0 is None:
                continue
            novo, linhas = aplica(tab, forma, r0, c0)
            h = alturas(novo)
            out.append({"rot": ri, "col": c0, "larg": larg, "linhas": linhas, "tab": novo,
                        "buracos_novos": buracos(novo) - b0 + 0, "altura_max": max(h),
                        "irregularidade": sum(abs(h[i] - h[i + 1]) for i in range(LARG - 1))})
    out.sort(key=lambda j: (j["altura_max"], j["rot"], j["col"]))
    return out[:26]


def desenha(tab):
    return "\n".join("|" + "".join("#" if x else "." for x in linha) + "|" for linha in tab) + "\n+" + "-" * LARG + "+"


def pedido(tab, peca, proxima, js):
    """Pedido no formato do Brier/Jev; as opções seguem a ordem de `js`."""
    estado = (f"Tetris, tabuleiro 20x10 (# ocupado, . vazio; linha de baixo é o chão):\n{desenha(tab)}\n"
              f"Peça atual: {peca}. Próxima peça: {proxima}.\n"
              "Objetivo: limpar o máximo de linhas e sobreviver o máximo de peças. Linha completa some.")
    crit = {}
    for k, j in zip(LETRAS, js):
        crit[k] = (f"rotação {j['rot']}, colunas {j['col'] + 1} a {j['col'] + j['larg']}: "
                   f"limpa {j['linhas']} linha(s), cria {max(0, j['buracos_novos'])} buraco(s) novo(s), "
                   f"altura máxima fica {j['altura_max']}, irregularidade da superfície {j['irregularidade']}")
    q = {"jogada": {"type": "choice",
                    "instructions": "Qual jogada é a melhor para esta peça? Pense em limpar linhas, evitar buracos e manter o tabuleiro baixo e plano.",
                    "criteria": crit}}
    return estado, q


def joga(decide, semente=7, max_pecas=150, registro=None):
    """Joga uma partida. decide(state, questions) -> (probabilidades por chave, latência em s)."""
    seq = sequencia(semente, max_pecas + 1)
    tab, pontos, linhas, passos = vazio(), 0, 0, []
    for i in range(max_pecas):
        peca, prox = seq[i], seq[i + 1]
        js = jogadas(tab, peca)
        if not js:
            break
        estado, q = pedido(tab, peca, prox, js)
        probs, lat = decide(estado, q)
        chaves = LETRAS[:len(js)]
        k = max(chaves, key=lambda c: probs.get(c, 0.0))
        j = js[chaves.index(k)]
        forma = rotacoes(peca)[j["rot"]]
        r0 = solta(tab, forma, j["col"])
        tab, l = aplica(tab, forma, r0, j["col"])
        pontos += PONTOS[l]
        linhas += l
        passos.append({"peca": peca, "rot": j["rot"], "col": j["col"], "r0": r0, "linhas": l,
                       "conf": round(probs.get(k, 0.0), 4), "n_opcoes": len(js), "latencia": round(lat, 3),
                       "buracos": buracos(tab), "altura": max(alturas(tab))})
        if registro:
            registro(i, passos[-1])
    return {"semente": semente, "pecas": len(passos), "linhas": linhas, "pontos": pontos,
            "fim_de_jogo": len(passos) < max_pecas, "passos": passos, "sequencia": seq[:len(passos) + 1]}


if __name__ == "__main__":
    # teste rápido com um decisor guloso (mais linhas, menos buracos): o simulador tem que funcionar sozinho
    def guloso(estado, q):
        crit = q["jogada"]["criteria"]
        def nota(t):
            import re
            n = [int(x) for x in re.findall(r"-?\d+", t)]
            return n[-4] * 10 - n[-3] * 5 - n[-2] - n[-1] * 0.3
        melhor = max(crit, key=lambda k: nota(crit[k]))
        return {k: (1.0 if k == melhor else 0.0) for k in crit}, 0.0
    r = joga(guloso, semente=7, max_pecas=150)
    print(json.dumps({k: v for k, v in r.items() if k not in ("passos", "sequencia")}))
