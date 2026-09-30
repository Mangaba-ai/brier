"""Testes portáveis do Brier: rodam em Windows, Linux e macOS, só com PyTorch e sem baixar modelos.

Usam um Qwen3 minúsculo com pesos aleatórios e um tokenizador de caracteres. Não medem qualidade;
garantem que a mecânica é a mesma em qualquer sistema:
  - a máscara em blocos é respeitada pelo transformers (uma pergunta não enxerga as outras);
  - a dequantização do formato MLX e a fusão do LoRA estão corretas;
  - a resposta sempre sai no tipo pedido, com probabilidades válidas;
  - o servidor valida o corpo e responde em UTF-8.
"""
import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nucleo import formata, mascara_blocos_np, valida_pergunta  # noqa: E402
from motor_torch import DecisorTorch, _dequantiza, _funde_lora  # noqa: E402


class TokenizadorDeCaracteres:
    """Um token por caractere (ids 3..), o bastante para montar pedidos e ler rótulos de 1 caractere."""

    def encode(self, texto, add_special_tokens=False):
        return [3 + (ord(c) % 250) for c in texto]

    def apply_chat_template(self, *a, **k):
        return "<|im_start|>assistant\n"


def qwen3_minusculo(semente=0):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(semente)
    cfg = Qwen3Config(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096,
                      tie_word_embeddings=True)
    return Qwen3ForCausalLM(cfg).to(torch.float32).eval()


@pytest.fixture(scope="module")
def decisor():
    return DecisorTorch.de_objetos(qwen3_minusculo(), TokenizadorDeCaracteres(), dispositivo="cpu")


Q_X = {"type": "choice", "instructions": "Qual?", "criteria": {"a": "um", "b": "dois", "c": "tres"}}
Q_Y = {"type": "noul", "instructions": "Sim?", "criteria": {"true": "sim", "false": "nao"}}
Q_Y2 = {"type": "noul", "instructions": "Nao?", "criteria": {"true": "nao", "false": "sim"}}  # mesmo tamanho
Q_S = {"type": "score", "instructions": "Nivel?", "criteria": ["baixo", "medio", "alto", "maximo"]}


def _probs(resp, k):
    a = resp["answers"][k]
    return [a["noul"], 1 - a["noul"]] if a["type"] == "noul" else list(a["probabilities"].values())


def test_mascara_isola_blocos():
    m = mascara_blocos_np(10, 4, [(0, 4), (4, 7), (7, 10)])
    assert m[5, :4].all() and m[8, :4].all()          # blocos veem o prefixo
    assert not m[8, 4:7].any()                         # 2º bloco não vê o 1º
    assert not m[5, 7:].any()                          # nem o 1º vê o 2º
    assert not m[2, 3]                                 # causal no prefixo


def test_pergunta_nao_enxerga_as_outras(decisor):
    st = "cliente reclamou da conta"
    sozinha = decisor.decide({"state": st, "questions": {"x": Q_X}})
    com_y = decisor.decide({"state": st, "questions": {"x": Q_X, "y": Q_Y}})
    com_y2 = decisor.decide({"state": st, "questions": {"x": Q_X, "y": Q_Y2}})
    assert np.allclose(_probs(sozinha, "x"), _probs(com_y, "x"), atol=1e-5)
    assert np.allclose(_probs(com_y, "x"), _probs(com_y2, "x"), atol=1e-5)
    # e o bloco de depois não depende do conteúdo do bloco de antes (mesmo comprimento)
    a = decisor.decide({"state": st, "questions": {"y0": Q_Y, "x": Q_X}})
    b = decisor.decide({"state": st, "questions": {"y0": Q_Y2, "x": Q_X}})
    assert np.allclose(_probs(a, "x"), _probs(b, "x"), atol=1e-5)


def test_saida_sempre_no_tipo(decisor):
    r = decisor.decide({"state": {"texto": "ção, acentuação e emoji ✓"},
                        "questions": {"x": Q_X, "y": Q_Y, "s": Q_S}}, amostras=3)["answers"]
    assert r["x"]["choice"] in Q_X["criteria"] and abs(sum(r["x"]["probabilities"].values()) - 1) < 1e-3
    assert 0 <= r["y"]["noul"] <= 1
    assert 0 <= r["s"]["score"] <= 3 and set(r["s"]["probabilities"]) == {"0", "1", "2", "3"}
    for a in r.values():
        assert a["incerteza"]["epistemica"] >= 0


def _quantiza_mlx(w: np.ndarray, bits=4, grupo=64):
    """Quantizador de referência no formato afim do MLX (para testar a dequantização)."""
    out_, in_ = w.shape
    g = w.reshape(out_, in_ // grupo, grupo)
    mn, mx = g.min(-1, keepdims=True), g.max(-1, keepdims=True)
    esc = (mx - mn) / (2 ** bits - 1)
    q = np.clip(np.round((g - mn) / esc), 0, 2 ** bits - 1).astype(np.uint32).reshape(out_, in_)
    por = 32 // bits
    q = q.reshape(out_, in_ // por, por)
    emp = np.zeros((out_, in_ // por), dtype=np.uint32)
    for j in range(por):
        emp |= q[..., j] << np.uint32(j * bits)
    return emp, esc[..., 0].astype(np.float32), mn[..., 0].astype(np.float32)


def test_dequantizacao_formato_mlx():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(8, 128)).astype(np.float32)
    emp, esc, vie = _quantiza_mlx(w)
    wq = _dequantiza(torch.from_numpy(emp.view(np.int32)), torch.from_numpy(esc), torch.from_numpy(vie), 4, 64)
    passo = float(esc.max())
    assert np.abs(wq.numpy() - w).max() <= passo / 2 + 1e-5


def test_fusao_lora_igual_ao_calculo_separado():
    torch.manual_seed(0)
    lin = torch.nn.Linear(6, 5, bias=False)
    modelo = torch.nn.Module()
    modelo.proj = lin
    a, b = torch.randn(6, 2), torch.randn(2, 5)
    x = torch.randn(3, 6)
    esperado = lin(x) + 20.0 * (x @ a) @ b
    _funde_lora(modelo, {"proj.lora_a": a, "proj.lora_b": b}, escala=20.0)
    assert torch.allclose(lin(x), esperado, atol=1e-4)


def test_valida_pergunta():
    assert valida_pergunta(Q_X) and valida_pergunta(Q_Y) and valida_pergunta(Q_S)
    assert not valida_pergunta({"type": "choice", "instructions": "x", "criteria": {"a": 1}})
    assert not valida_pergunta({"type": "outro", "instructions": "x", "criteria": {}})


def test_formata_conformal():
    r = formata(Q_X, np.array([[0.7, 0.2, 0.1]]), {"choice": 0.8})
    assert r["choice"] == "a" and r["conjunto"] == ["a", "b"]


def test_servidor_valida_e_responde_utf8(decisor):
    from servidor import cria_handler
    srv = HTTPServer(("127.0.0.1", 0), cria_handler(decisor))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"

    def post(caminho, corpo):
        req = urllib.request.Request(url + caminho, data=json.dumps(corpo).encode("utf-8"), method="POST")
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode("utf-8"))

    try:
        cod, r = post("/v1/decide", {"state": "ação", "questions": {"x": Q_X}})
        assert cod == 200 and r["answers"]["x"]["choice"] in Q_X["criteria"]
        cod, _ = post("/v1/systemone", {"state": "ação", "questions": {"x": Q_X}})
        assert cod == 200
        cod, r = post("/v1/decide", {"state": "x", "questions": {"x": {"type": "choice", "criteria": ["a"]}}})
        assert cod == 422 and "inválidas" in r["detail"][0]["msg"]
    finally:
        srv.shutdown()
