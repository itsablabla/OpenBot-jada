"""Factory's Droid as a Bot, spoken to over AG-UI.

The default harness. Droid ships no AG-UI server of its own, so this one wraps
`droid exec` — Factory's headless mode — behind the same endpoint shape every
other harness serves: `/health` for Compose, the run route at the server root,
and nothing without the deployment's own token.

Credentials are decided once, at import, in `credentials.py`: a Factory key
runs Droid as Factory ships it, and a model-provider key from the model screen
is written into Droid's own BYOK config instead. Neither is a startup refusal
that names both remedies.
"""

import os

from ag_ui.core import RunAgentInput
from ag_ui.encoder import EventEncoder
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .adapter import DroidAdapter, DroidSettings
from .credentials import resolve, write_config

TOKEN_HEADER = "x-openbot-agent-token"

_credentials = resolve()
if _credentials.config is not None:
    write_config(_credentials.config)

adapter = DroidAdapter(DroidSettings(model=_credentials.model))

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
