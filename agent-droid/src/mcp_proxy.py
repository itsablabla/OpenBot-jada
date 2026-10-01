"""OpenBot's deployment tools, served to Droid as a stdio MCP server.

Droid runs its own local tools, but the tools this deployment governs — the
ones a run arrives with — are called back through the server's signed
`agent-tools/call` endpoint, the same door `agent-langgraph` uses. Droid speaks
MCP natively, so this script is the bridge: registered once in
`~/.factory/mcp.json` (see `main.py`), spawned by Droid itself, and speaking
newline-delimited JSON-RPC on stdio as the MCP stdio transport requires.

Which tools exist, and on whose behalf they run, changes per run. The adapter
writes that to a file before every spawn and names it in `OPENBOT_RUN_FILE`,
which this process inherits from the `droid` that spawned it. No file means no
run to act for, and the listing is honestly empty.

Standard library only, deliberately. This runs inside the Bot's container as a
child of a child; a dependency here is a dependency `droid` has to be able to
start, and stdlib JSON over pipes plus `urllib` for one POST needs nothing.
"""

import json
import os
import sys
import urllib.error
import urllib.request

# One tool call, one bounded wait. The server's own budget for a call is a
# minute; waiting much longer only holds Droid's run open for a dead call.
CALL_TIMEOUT_SECONDS = 90


def context() -> dict:
    """The run this process is allowed to act for, or nothing."""
    path = os.environ.get("OPENBOT_RUN_FILE", "").strip()
    if not path:
        return {}
    try:
        with open(path, encoding="utf-8") as file:
            found = json.load(file)
    except (OSError, ValueError):
        return {}
    return found if isinstance(found, dict) else {}


def tools() -> list[dict]:
    listed = []
    for tool in context().get("tools") or []:
        if not isinstance(tool, dict) or not tool.get("name"):
            continue
        listed.append(
            {
                "name": str(tool["name"]),
                "description": str(tool.get("description") or ""),
                "inputSchema": tool.get("parameters")
                or {"type": "object", "properties": {}},
            }
        )
    return listed


def call(name: str, arguments: dict) -> dict:
    """One tool, called back through the deployment that governs it."""
    found = context()
    url = str(found.get("url") or "")
    token = str(found.get("token") or "")
    run = str(found.get("run") or "")
    if not url or not token:
        return refusal(
            "Refused. This Bot has no credential for calling tools back "
            "through its deployment."
        )
    if not run:
        return refusal(
            "Refused. This run carried no signed statement of which Bot and "
            "person it is for."
        )
    request = urllib.request.Request(
        url,
        data=json.dumps({"name": name, "args": arguments, "run": run}).encode(),
        headers={
            "content-type": "application/json",
            "x-openbot-agent-token": token,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=CALL_TIMEOUT_SECONDS) as response:
            text = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:2000]
        return refusal(f"That tool could not be called: HTTP {error.code}. {detail}")
    except (urllib.error.URLError, OSError, TimeoutError) as error:
        return refusal(f"That tool could not be called: {error}")
    return {"content": [{"type": "text", "text": text}]}


def refusal(text: str) -> dict:
    # A result the model reads, not a protocol error: the run continues and
    # says what it could not do, exactly as the LangGraph Bot's loop does.
    return {"content": [{"type": "text", "text": text}], "isError": True}


def answer(message: dict) -> dict | None:
    method = message.get("method")
    if method == "initialize":
        requested = (message.get("params") or {}).get("protocolVersion")
        return {
            "protocolVersion": requested or "2025-06-18",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "openbot", "version": "1.0.0"},
        }
    if method == "ping":
        return {}
    if method == "tools/list":
        return {"tools": tools()}
    if method == "tools/call":
        params = message.get("params") or {}
        arguments = params.get("arguments")
        return call(
            str(params.get("name") or ""),
            arguments if isinstance(arguments, dict) else {},
        )
    return None


def serve(stdin, stdout) -> None:
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except ValueError:
            continue
        if not isinstance(message, dict) or "id" not in message:
            # Notifications get no reply; that is what makes them notifications.
            continue
        result = answer(message)
        reply: dict = {"jsonrpc": "2.0", "id": message["id"]}
        if result is None:
            reply["error"] = {"code": -32601, "message": "Method not found"}
        else:
            reply["result"] = result
        stdout.write(json.dumps(reply) + "\n")
        stdout.flush()


if __name__ == "__main__":
    serve(sys.stdin, sys.stdout)
