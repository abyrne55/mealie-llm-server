###############################################
# Stage 1: Builder
###############################################
FROM quay.io/hummingbird/python:3.14-builder AS builder

USER 0

ARG TARGETPLATFORM

RUN dnf install -y --setopt=install_weak_deps=False \
    gcc \
    gcc-c++ \
    make \
    cmake \
    python3-devel \
    uv \
    && dnf clean all

WORKDIR /build

COPY pyproject.toml uv.lock ./

ENV CMAKE_ARGS="-DGGML_NATIVE=off"
RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
        export CMAKE_ARGS="-DGGML_NATIVE=off -DGGML_AVX2=on -DGGML_FMA=on -DGGML_F16C=on"; \
    elif [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
        export CMAKE_ARGS="-DGGML_NATIVE=off -DGGML_SVE=off"; \
    fi && \
    CMAKE_ARGS="$CMAKE_ARGS" uv sync --no-dev --frozen

COPY mealie_llm_server ./mealie_llm_server

###############################################
# Stage 2: Runtime
###############################################
FROM quay.io/hummingbird/python:3.14

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY --from=builder --chown=65532:0 /build/.venv /app/.venv
COPY --from=builder --chown=65532:0 /build/mealie_llm_server /app/mealie_llm_server

ENV PATH="/app/.venv/bin:$PATH"

VOLUME ["/models"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=120s \
    CMD ["python3", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]

ENTRYPOINT ["uvicorn", "mealie_llm_server.app:app", "--host", "0.0.0.0", "--port", "8000"]
