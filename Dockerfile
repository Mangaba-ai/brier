# Brier (v3 por padrão; PyTorch) em CPU. Para GPU NVIDIA, troque a linha do torch pelo índice CUDA (ex.: /whl/cu124).
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY brier ./brier
RUN pip install --no-cache-dir .
ENV BRIER_MODELO=v3
EXPOSE 8790
CMD ["brier-serve", "--host", "0.0.0.0", "--porta", "8790"]
