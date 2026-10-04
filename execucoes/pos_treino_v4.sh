#!/bin/bash
# Esteira do v4: espera o treino, avalia, publica no Hugging Face. Log em execucoes/pos_treino_v4.log
cd "$(dirname "$0")/.."
until grep -q "salvo em execucoes/v4" execucoes/v4.log; do
  grep -qE "Traceback|Error" execucoes/v4.log && { echo "treino falhou"; exit 1; }
  sleep 60
done
echo "treino terminou $(date)"
.venv/bin/python src/laya_avaliar.py --modelo execucoes/v4 --nome brier-v4 || { echo "avaliação falhou"; exit 1; }
echo "avaliação terminou $(date)"
.venv/bin/python src/publicar_v4.py || { echo "publicação falhou"; exit 1; }
echo "fim $(date)"
