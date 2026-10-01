"""Factory's Droid as a Bot, spoken to over AG-UI.

The default harness. Droid ships no AG-UI server of its own, so this one wraps
Droid's own headless modes behind the same endpoint shape every other harness
serves: `/health` for Compose, the run route at the server root, and nothing
without the deployment's own token.

Two transports, one seam. `droid exec --output-format stream-json` is the
default: one process per run, an output format this repository regression-tests.
`DROID_TRANSPORT=acp` keeps one Droid alive and speaks the Agent Client
Protocol instead — see `acp.py` for what that buys and costs. Everything above
the transport is identical: same events, same threads, same tools.

Credentials are decided once, at import, in `credentials.py`: a Factory key —
first-class on the model screen now — runs Droid as Factory ships it, and a
model-provider key is written into Droid's own BYOK config instead. Neither
being present is refused at startup with both remedies named.

The deployment's own tools are served to Droid natively, over MCP: when the
compose file hands this container the signed tool callback, the bridge in
`mcp_proxy.py` is registered in Droid's own `~/.factory/mcp.json` so `droid
exec` finds it, and per run the adapter says which tools that run actually
carries. Droid's workspace and sessions live in `/workspace` and `~/.factory`,
both mounted by the deployment so they outlive any one container.

The container can also stand as its own Droid Computer: `DROID_COMPUTER_NAME`
registers it with Factory (`droid computer register`) and keeps `droid daemon
--remote-access` running beside the server, so Factory's app, CLI and Slack
can reach the same persistent `~/.factory` and `/workspace` this harness uses.
Opt-in, and only on a Factory key — registration is a Factory account feature.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from ag_ui.core import RunAgentInput
from ag_ui.encoder import EventEncoder
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .acp import AcpDroid
from .adapter import DroidAdapter, DroidSettings, MCP_SERVER_NAME
from .credentials import resolve, write_config
from .sessions import SessionStore

TOKEN_HEADER = "x-openbot-agent-token"


def _workspace() -> str:
    """Where Droid works: the deployment's mount, or nowhere in particular.

    `/workspace` is where compose mounts the shared workspace volume; an
    `AGENTS.md` kept there is read by Droid itself, natively, with nothing for
    this harness to do. `DROID_WORKSPACE` overrides for deployments that mount
    elsewhere, and a path that does not exist means no mount was given, which
    must not become a spawn that dies on a missing directory.
    """
    workspace = (os.environ.get("DROID_WORKSPACE") or "/workspace").strip()
    return workspace if os.path.isdir(workspace) else ""


def _register_mcp_bridge() -> None:
    """Name the deployment-tools bridge in Droid's own MCP config.

    Written whole, like the BYOK config: this container's `~/.factory` is the
    deployment's own volume and Droid's config in it is this harness's to
    manage. Skipped when compose handed over no tool callback — a bridge with
    nothing to call would only put an empty listing in front of every run.
    """
    if not (os.environ.get("OPENBOT_TOOL_URL") or "").strip():
        return
    if not (os.environ.get("AGENT_TOOL_TOKEN") or "").strip():
        return
    path = Path.home() / ".factory" / "mcp.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    MCP_SERVER_NAME: {
                        "type": "stdio",
                        "command": sys.executable,
                        "args": [str(Path(__file__).with_name("mcp_proxy.py"))],
                    }
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )


_credentials = resolve()
if _credentials.config is not None:
    write_config(_credentials.config)
_register_mcp_bridge()


def _register_computer() -> subprocess.Popen | None:
    """Stand this container up as its own Droid Computer, when asked to.

    Registration (`droid computer register <name>`) is idempotent in effect
    here because `~/.factory` is the deployment's own volume: a name already
    registered on a previous start makes the command fail, and that failure is
    tolerated rather than fatal — the daemon beside it is what matters. The
    daemon (`droid daemon --remote-access`) is what connects the machine to
    Factory's relay; it runs for as long as this process does, next to the
    AG-UI server, sharing the same `~/.factory` and `/workspace`.

    Only on a Factory key: a Droid Computer is a Factory account feature, and
    without the account credential the register call could only fail after a
    network timeout, which is a worse refusal than this quiet one.
    """
    name = (os.environ.get("DROID_COMPUTER_NAME") or "").strip()
    if not name:
        return None
    if not (os.environ.get("FACTORY_API_KEY") or "").strip():
        print(
            "DROID_COMPUTER_NAME is set but FACTORY_API_KEY is not; a Droid "
            "Computer needs a Factory credential, so none was registered.",
            file=sys.stderr,
        )
        return None
    try:
        subprocess.run(["droid", "computer", "register", name], check=False, timeout=120)
        return subprocess.Popen(["droid", "daemon", "--remote-access"])
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"Droid Computer registration failed: {error}", file=sys.stderr)
        return None


# Held so the relay daemon lives and dies with this process, not the first GC.
_computer_daemon = _register_computer()

_settings = DroidSettings(
    model=_credentials.model,
    autonomy=(os.environ.get("DROID_AUTONOMY") or "low").strip() or "low",
    workspace=_workspace(),
)

# The transport is an implementation detail of this file: everything else in
# the deployment sees the same AG-UI endpoint whichever one is chosen.
if (os.environ.get("DROID_TRANSPORT") or "").strip() == "acp":
    adapter = AcpDroid(_settings)
else:
    adapter = DroidAdapter(
        _settings,
        sessions=SessionStore(Path.home() / ".factory" / "openbot-threads.json"),
    )

app = FastAPI()


@app.middleware("http")
async def refuse_without_the_server_token(request: Request, call_next):
    """Everything but `/health`, which Compose polls before any token exists."""
    if request.url.path != "/health":
        expected = (os.environ.get("MANAGED_AGENT_TOKEN") or "").strip()
        offered = (request.headers.get(TOKEN_HEADER) or "").strip()
        if not expected or offered != expected:
            return JSONResponse({"error": "unauthorised"}, status_code=401)
    return await call_next(request)


@app.get("/health")
async def health():
    return {"ok": True, "harness": "droid"}


@app.post("/")
async def run(input_data: RunAgentInput, request: Request):
    encoder = EventEncoder(accept=request.headers.get("accept"))

    async def stream():
        async for event in adapter.run(input_data):
            yield encoder.encode(event)

    return StreamingResponse(stream(), media_type=encoder.get_content_type())
