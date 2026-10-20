FROM python:3.12.14-bookworm

ARG INSTALL_CODEX=1
ARG INSTALL_CLAUDE_CODE=1
ARG CODEX_VERSION=0.148.0
ARG CLAUDE_CODE_VERSION=2.1.237

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AGENT_HOME_DIR=/root/.openclaw/ \
    DYNACOMSEC_WORKSPACE_ROOT=/workspace/agent \
    PATH="/workspace/.local/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git tini nodejs npm \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    packages=""; \
    if [ "$INSTALL_CODEX" = "1" ]; then packages="$packages @openai/codex@$CODEX_VERSION"; fi; \
    if [ "$INSTALL_CLAUDE_CODE" = "1" ]; then packages="$packages @anthropic-ai/claude-code@$CLAUDE_CODE_VERSION"; fi; \
    if [ -n "$packages" ]; then npm install -g $packages; fi

WORKDIR /workspace

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir openai==3.3.1 mcp==2.0.0

COPY . /workspace/dynacomsec

RUN mkdir -p /opt/dynacomsec/tasks /logs/agent /logs/verifier /logs/artifacts /workspace/agent

COPY docker/entrypoint.sh /usr/local/bin/dynacomsec-entrypoint.sh
RUN chmod +x /usr/local/bin/dynacomsec-entrypoint.sh

EXPOSE 8120

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/dynacomsec-entrypoint.sh"]
CMD ["sleep", "infinity"]
