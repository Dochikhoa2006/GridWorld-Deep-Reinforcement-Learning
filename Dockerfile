FROM python:3.11.15-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    XDG_CACHE_HOME=/tmp/gridworld-cache \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/gridworld-matplotlib

WORKDIR /app

ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

RUN groupadd --gid 10001 gridworld \
    && useradd --uid 10001 --gid gridworld \
        --no-create-home --home-dir /nonexistent \
        --shell /usr/sbin/nologin gridworld \
    && chown gridworld:gridworld /app

COPY --chown=gridworld:gridworld requirements.txt pyproject.toml README.md LICENSE ./
COPY --chown=gridworld:gridworld src ./src

RUN python -m pip install \
        --index-url "${PYTORCH_INDEX_URL}" torch==2.11.0 \
    && python -m pip install -r requirements.txt \
    && python -m pip install --no-deps .

COPY --chown=gridworld:gridworld configs ./configs

USER gridworld

ENTRYPOINT ["gridworld-rl"]
CMD ["--help"]
