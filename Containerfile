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
    python3-devel \
    && dnf clean all \
    && pip install --root-user-action=ignore uv cmake

WORKDIR /build

COPY pyproject.toml uv.lock ./

ENV CMAKE_ARGS="-DGGML_NATIVE=off"
RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
        export CMAKE_ARGS="-DGGML_NATIVE=off -DGGML_AVX2=on -DGGML_FMA=on -DGGML_F16C=on"; \
    elif [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
        export CMAKE_ARGS="-DGGML_NATIVE=off -DGGML_SVE=off"; \
    fi && \
    CMAKE_ARGS="$CMAKE_ARGS" uv sync --no-dev --frozen

COPY mealie_local_ai ./mealie_local_ai

###############################################
# Stage 2: Runtime
###############################################
FROM quay.io/hummingbird/python:3.14

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY --from=builder --chown=65532:0 /build/.venv /app/.venv
COPY --from=builder --chown=65532:0 /build/mealie_local_ai /app/mealie_local_ai
COPY --chmod=755 --chown=65532:0 scripts/healthcheck.py /app/healthcheck.py

# libstdc++: needed by llama-cpp-python's bundled libllama.so
COPY --from=builder /usr/lib64/libstdc++.so.6* /usr/lib64/
# libgomp: needed by llama-cpp-python for OpenMP parallel inference
COPY --from=builder /usr/lib64/libgomp.so* /usr/lib64/

ENV PATH="/app/.venv/bin:$PATH"

VOLUME ["/models"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=120s \
    CMD ["/app/healthcheck.py"]

ENTRYPOINT ["python3", "-m", "uvicorn", "mealie_local_ai.app:app", "--host", "0.0.0.0", "--port", "8000"]
