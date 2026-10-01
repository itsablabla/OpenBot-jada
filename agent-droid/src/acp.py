"""Droid over the Agent Client Protocol, as the same AG-UI stream.

`droid exec --output-format acp` speaks ACP: JSON-RPC 2.0, one message per
line, over stdio, with the editor — here, this adapter — as the client. Where
the stream-json transport spawns one process per run and parses an output
format, this keeps one Droid alive and speaks a versioned protocol: prompts go
in as `session/prompt`, progress comes back as `session/update` notifications,
and cancellation and permission requests are protocol messages rather than
process signals.

Opt-in through `DROID_TRANSPORT=acp`, stream-json remaining the default: the
exec seam is the documented, regression-tested one, and this exists for
deployments that want the persistent process and the protocol contract.

The translation targets the same AG-UI events `adapter.py` emits, because the
transport is an implementation detail the rest of the deployment must not see:
same endpoint, same events, same session-per-thread continuity.

Permission requests are answered from the same `DROID_AUTONOMY` dial the exec
transport passes as `--auto`: "low" declines anything Droid did not already
have leave to do, anything else allows it once. A headless harness has nobody
to ask, so the dial the operator set is the answer.
"""

import asyncio
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path

from ag_ui.core import (
    RunAgentInput,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
)

from .adapter import DroidSettings, MCP_SERVER_NAME, _prompt_from, run_context

# ACP protocol major version this client speaks.
PROTOCOL_VERSION = 1


class AcpDroid:
    """One persistent Droid, one ACP conversation per AG-UI thread."""

    def __init__(self, settings: DroidSettings | None = None):
        self._settings = settings or DroidSettings()
        self._process: asyncio.subprocess.Process | None = None
        self._initialized = False
        self._next_id = 0
        # ACP sessions live and die with the agent process, so this map is
        # deliberately in memory: a restart means new sessions either way.
        self._sessions: dict[str, str] = {}
        self._run_files: dict[str, Path] = {}
        # One prompt at a time. ACP itself can multiplex, but interleaving two
        # runs' updates would need a demultiplexer this deliberately is not.
        self._lock = asyncio.Lock()

    def _argv(self) -> list[str]:
        return [
            self._settings.command,
            "exec",
            "--output-format",
            "acp",
            "--auto",
            self._settings.autonomy,
        ]

    async def _ensure_process(self) -> asyncio.subprocess.Process:
        if self._process is not None and self._process.returncode is None:
            return self._process
        self._process = await asyncio.create_subprocess_exec(
            *self._argv(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            cwd=self._settings.workspace or None,
        )
        # A dead process took its sessions with it; forget the ids so each
        # thread opens a fresh conversation instead of naming a ghost.
        self._sessions.clear()
        self._initialized = False
        return self._process

    async def aclose(self) -> None:
        """Stop the agent process, if one is running."""
        process, self._process = self._process, None
        self._sessions.clear()
        self._initialized = False
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()

    async def _send(self, process, message: dict) -> None:
        assert process.stdin is not None
        process.stdin.write((json.dumps(message) + "\n").encode())
        await process.stdin.drain()

    async def _request(self, process, method: str, params: dict) -> int:
        self._next_id += 1
        await self._send(
            process,
            {"jsonrpc": "2.0", "id": self._next_id, "method": method, "params": params},
        )
        return self._next_id

    async def _read(self, process) -> dict | None:
        assert process.stdout is not None
        line = await process.stdout.readline()
        if not line:
            return None
        try:
            message = json.loads(line.decode("utf-8", errors="replace"))
        except ValueError:
            return {}
        return message if isinstance(message, dict) else {}

    async def _await_response(self, process, request_id: int) -> dict:
        """Read until the response to `request_id`, discarding the rest.

        Used only for `initialize` and `session/new`, which produce no updates
        worth translating; the prompt loop below reads for itself.
        """
        while True:
            message = await self._read(process)
            if message is None:
                raise ConnectionError("Droid's ACP process ended mid-request.")
            if message.get("id") == request_id and "method" not in message:
                if "error" in message:
                    raise ConnectionError(
                        str((message["error"] or {}).get("message") or "ACP error")
                    )
                result = message.get("result")
                return result if isinstance(result, dict) else {}
            await self._answer_if_asked(process, message)

    async def _answer_if_asked(self, process, message: dict) -> None:
        """Agent-to-client requests, answered from what a headless client can say."""
        if "id" not in message or "method" not in message:
            return
        method = message.get("method")
        reply: dict = {"jsonrpc": "2.0", "id": message["id"]}
        if method == "session/request_permission":
            options = (message.get("params") or {}).get("options") or []
            chosen = self._permission(options)
            reply["result"] = {
                "outcome": (
                    {"outcome": "selected", "optionId": chosen}
                    if chosen
                    else {"outcome": "cancelled"}
                )
            }
        else:
            # `initialize` declared no fs or terminal capability, so anything
            # else the agent asks for is honestly unsupported.
            reply["error"] = {"code": -32601, "message": "Method not found"}
        await self._send(process, reply)

    def _permission(self, options: list) -> str | None:
        """The operator's autonomy dial, applied to a question nobody is here to read."""
        wanted = "allow" if self._settings.autonomy != "low" else "reject"
        for prefer in (f"{wanted}_once", wanted):
            for option in options:
                if (
                    isinstance(option, dict)
                    and str(option.get("kind", "")).startswith(prefer)
                    and option.get("optionId")
                ):
                    return str(option["optionId"])
        for option in options:
            if isinstance(option, dict) and option.get("optionId"):
                return str(option["optionId"])
        return None

    def _mcp_servers(self, thread_id: str) -> list[dict]:
        """The deployment-tools bridge, named at session birth as ACP wants."""
        url = (os.environ.get("OPENBOT_TOOL_URL") or "").strip()
        token = (os.environ.get("AGENT_TOOL_TOKEN") or "").strip()
        if not url or not token:
            return []
        run_file = self._run_files.get(thread_id)
        if run_file is None:
            handle, name = tempfile.mkstemp(prefix="openbot-acp-run-", suffix=".json")
            os.close(handle)
            run_file = Path(name)
            self._run_files[thread_id] = run_file
        return [
            {
                "name": MCP_SERVER_NAME,
                "command": sys.executable,
                "args": [str(Path(__file__).with_name("mcp_proxy.py"))],
                "env": [{"name": "OPENBOT_RUN_FILE", "value": str(run_file)}],
            }
        ]

    async def _session_for(self, process, thread_id: str) -> str:
        session = self._sessions.get(thread_id, "")
        if session:
            return session
        request_id = await self._request(
            process,
            "session/new",
            {
                "cwd": self._settings.workspace or os.getcwd(),
                "mcpServers": self._mcp_servers(thread_id),
            },
        )
        result = await self._await_response(process, request_id)
        session = str(result.get("sessionId") or "")
        if not session:
            raise ConnectionError("Droid's ACP process opened no session.")
        self._sessions[thread_id] = session
        return session

    async def run(self, input_data: RunAgentInput):
        thread_id, run_id = input_data.thread_id, input_data.run_id
        yield RunStartedEvent(thread_id=thread_id, run_id=run_id)

        prompt = _prompt_from(input_data)
        if not prompt:
            yield RunErrorEvent(
                message="The run carried no user message to send to Droid."
            )
            return

        async with self._lock:
            try:
                process = await self._ensure_process()
                if not self._initialized:
                    request_id = await self._request(
                        process,
                        "initialize",
                        {
                            "protocolVersion": PROTOCOL_VERSION,
                            "clientCapabilities": {
                                "fs": {"readTextFile": False, "writeTextFile": False}
                            },
                        },
                    )
                    await self._await_response(process, request_id)
                    self._initialized = True
                session = await self._session_for(process, thread_id)
                context = run_context(input_data)
                run_file = self._run_files.get(thread_id)
                if run_file is not None:
                    run_file.write_text(
                        json.dumps(context or {}), encoding="utf-8"
                    )
                prompt_id = await self._request(
                    process,
                    "session/prompt",
                    {
                        "sessionId": session,
                        "prompt": [{"type": "text", "text": prompt}],
                    },
                )
            except (OSError, ConnectionError) as error:
                yield RunErrorEvent(message=f"Droid could not be reached: {error}")
                return

            message_id = ""
            while True:
                message = await self._read(process)
                if message is None:
                    if message_id:
                        yield TextMessageEndEvent(message_id=message_id)
                    yield RunErrorEvent(message="Droid's ACP process ended mid-run.")
                    return
                if message.get("id") == prompt_id and "method" not in message:
                    if message_id:
                        yield TextMessageEndEvent(message_id=message_id)
                        message_id = ""
                    if "error" in message:
                        yield RunErrorEvent(
                            message=str(
                                (message["error"] or {}).get("message")
                                or "Droid reported an error."
                            )
                        )
                        return
                    stop = str((message.get("result") or {}).get("stopReason") or "")
                    if stop in ("refusal", "cancelled"):
                        yield RunErrorEvent(message=f"Droid stopped: {stop}.")
                        return
                    yield RunFinishedEvent(thread_id=thread_id, run_id=run_id)
                    return
                if message.get("method") == "session/update":
                    params = message.get("params") or {}
                    if params.get("sessionId") not in (None, session):
                        continue
                    update = params.get("update") or {}
                    kind = update.get("sessionUpdate")
                    if kind == "agent_message_chunk":
                        content = update.get("content") or {}
                        text = content.get("text") if isinstance(content, dict) else None
                        if text:
                            if not message_id:
                                message_id = str(uuid.uuid4())
                                yield TextMessageStartEvent(
                                    message_id=message_id, role="assistant"
                                )
                            yield TextMessageContentEvent(
                                message_id=message_id, delta=text
                            )
                    elif kind == "tool_call":
                        if message_id:
                            yield TextMessageEndEvent(message_id=message_id)
                            message_id = ""
                        tool_call_id = str(update.get("toolCallId") or uuid.uuid4())
                        yield ToolCallStartEvent(
                            tool_call_id=tool_call_id,
                            tool_call_name=str(
                                update.get("title") or update.get("kind") or "tool"
                            ),
                        )
                        yield ToolCallArgsEvent(
                            tool_call_id=tool_call_id,
                            delta=json.dumps(update.get("rawInput") or {}),
                        )
                        yield ToolCallEndEvent(tool_call_id=tool_call_id)
                    elif kind == "tool_call_update":
                        if update.get("status") in ("completed", "failed"):
                            texts = "".join(
                                str(((item.get("content") or {}).get("text")) or "")
                                for item in update.get("content") or []
                                if isinstance(item, dict)
                            )
                            yield ToolCallResultEvent(
                                message_id=str(uuid.uuid4()),
                                tool_call_id=str(update.get("toolCallId") or ""),
                                content=texts,
                            )
                    continue
                await self._answer_if_asked(process, message)
