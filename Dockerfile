FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /workspace

ENV HF_ENDPOINT=https://hf-mirror.com \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
COPY benchmarks ./benchmarks

RUN pip install --upgrade pip && \
    pip install -e ".[dev]"

RUN mkdir -p outputs/checkpoints outputs/logs outputs/wandb

CMD ["bash"]
