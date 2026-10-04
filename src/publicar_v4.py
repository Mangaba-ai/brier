"""Publica o Brier v4 no Hugging Face com o cartão preenchido pelos resultados da avaliação.

Lê execucoes/laya/brier-v4.json (gerado por src/laya_avaliar.py --modelo execucoes/v4 --nome brier-v4),
preenche docs/cartao_hf.md e envia execucoes/v4 (sem o checkpoint intermediário) para o repositório.

Uso: python src/publicar_v4.py [--repo mangaba-ai/brier-v4] [--privado]
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

NOMES = {"sintetico": "Sintético (teste do Brier, maioria de votos)", "assin2": "ASSIN2 (implicação + similaridade)",
         "b2w": "B2W (avaliações de produto)", "tweetsentbr": "tweetSentBR (sentimento)"}


def tabela(rel: dict) -> str:
    linhas = ["Mesmos pedidos para o Brier v4 e para o Jev (TypeSafe AI, modelo comercial usado só como referência),",
              "250 por conjunto, intervalo de confiança de 95% por bootstrap pareado.", "",
              "| Conjunto | Acerto Brier v4 | Acerto Jev | Escore de Brier v4 ↓ | Escore de Brier Jev ↓ | Latência mediana |",
              "|---|---|---|---|---|---|"]
    for k, nome in NOMES.items():
        c = rel["conjuntos"].get(k)
        if not c:
            continue
        linhas.append(f"| {nome} | {c['modelo']['acc'] * 100:.1f}% | {c['jev']['acc'] * 100:.1f}% | "
                      f"{c['modelo']['brier']:.3f} | {c['jev']['brier']:.3f} | {c['latencia_p50_ms']:.0f} ms |")
    linhas += ["", f"Vitórias pelo critério fixado antes da medição (IC favorável em acerto e em escore de Brier, "
               f"em 2 de 3 conjuntos humanos): **{rel['vitorias_humanas']}**."]
    if "injecao" in rel:
        i = rel["injecao"]
        linhas.append(f"Injeção de instruções (frases nunca vistas no treino): a resposta virou o que o ataque pedia em "
                      f"{i['taxa_ataque'] * 100:.1f}% dos casos; acerto {i['acc_limpo']['acc'] * 100:.1f}% → "
                      f"{i['acc_alterado']['acc'] * 100:.1f}%.")
    if "longo" in rel:
        lg = rel["longo"]
        linhas.append(f"Documentos longos (~2.500 tokens, lidos por janelas): acerto {lg['acc_limpo']['acc'] * 100:.1f}% → "
                      f"{lg['acc_alterado']['acc'] * 100:.1f}%.")
    linhas.append("Medido num MacBook Air M5 (GPU integrada, MPS).")
    return "\n".join(linhas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="mangaba-ai/brier-v4")
    ap.add_argument("--privado", action="store_true")
    a = ap.parse_args()
    rel = json.loads(Path("execucoes/laya/brier-v4.json").read_text(encoding="utf-8"))
    cartao = Path("docs/cartao_hf.md").read_text(encoding="utf-8").replace("{{RESULTADOS}}", tabela(rel))
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / "brier-v4"
        shutil.copytree("execucoes/v4", dst, ignore=shutil.ignore_patterns("ultima_epoca", "*.log"))
        (dst / "README.md").write_text(cartao, encoding="utf-8")
        api = HfApi()
        api.create_repo(a.repo, private=a.privado, exist_ok=True)
        api.upload_folder(folder_path=str(dst), repo_id=a.repo,
                          commit_message="Brier v4: pesos ajustados sobre o Laya multilíngue e cartão com resultados")
    print(f"publicado em https://huggingface.co/{a.repo}")


if __name__ == "__main__":
    main()
