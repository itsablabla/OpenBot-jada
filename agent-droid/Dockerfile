# Factory's Droid, as a Bot. Python for the translation layer only: Droid itself is Factory's
# published CLI, installed below, and `droid exec --output-format stream-json` is the seam the
# adapter reads. The protocol is the only thing a Bot has to share with the others.
FROM python:3.12-slim

# The Droid CLI, from Factory's own installer, plus the ripgrep it shells out to for search.
# curl and ca-certificates are kept: the CLI updates itself through them. The installer is
# downloaded to a file first — piping curl straight into sh exits 0 on a failed download, and a
# `droid` that is not there should fail the build, not the first run. `command -v` proves it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates ripgrep \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://app.factory.ai/cli -o /tmp/install-droid.sh \
    && sh /tmp/install-droid.sh \
    && rm /tmp/install-droid.sh
# The installer drops `droid` into /usr/local/bin, already on PATH; prove it before shipping.
RUN command -v droid

WORKDIR /app

# Dependencies before source, so editing the Bot does not re-resolve the framework.
COPY agent-droid/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# The provider spec every language in the box reads, one level above /app/src exactly as it is one
# level above the Bot in the repository.
COPY shared/model-providers.json shared/model_providers.py /shared/
COPY agent-droid/src ./src

ENV PORT=4214
EXPOSE 4214
# 0.0.0.0, not `::`. uvicorn binds `::` as IPv6-only, with no v4-mapped addresses, so a container
# started that way refuses 127.0.0.1: the Compose healthcheck never passes and the server cannot
# reach the Bot. Measured on agent-claude-sdk, inherited here.
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "4214"]
