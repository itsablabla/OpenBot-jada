"""Factory's Droid CLI as a Bot, translated by hand.

Droid has no AG-UI integration to lean on: `droid exec` is a headless CLI whose
`--output-format stream-json` emits newline-delimited events in the Claude
stream-json shape. This adapter spawns one `droid exec` per AG-UI run and
translates that stream into AG-UI events, which is the only thing a Bot has to
share with the others.

Threads outlive runs. Droid's own continuity is its session: the `system/init`
event names the session id, and passing it back through `--session-id` on the
next run of the same thread is what makes a second question land in the same
conversation. The map from AG-UI thread to Droid session lives here and nowhere
else.
"""

import asyncio
import json
import os
import tempfile
import uuid
from dataclasses import dataclass
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

from .sessions import SessionStore

# How much of stderr is kept for the error message when the CLI dies without a
# `result` event. Enough to say why, bounded so a looping process cannot grow it.
_STDERR_TAIL_BYTES = 4096

# The MCP server `main.py` registers for the deployment's own tools. Droid
# namespaces MCP tools by server, so its stream names them through this prefix;
# the AG-UI events carry the deployment's own names, with the plumbing removed.
MCP_SERVER_NAME = "openbot"
_MCP_PREFIX = f"mcp__{MCP_SERVER_NAME}__"


@dataclass(frozen=True)
class DroidSettings:
    """What one spawn needs to know beyond the prompt."""

    command: str = "droid"
    # The BYOK custom model's display name, when main.py wrote one, or a model
    # the environment pinned. Empty means Droid's own default.
    model: str = ""
    # Droid runs its own local tools. `DROID_AUTONOMY` in the deployment's
    # environment chooses how far they may go; "low" keeps it to reads, because
    # nothing about a fresh install says its Bot may edit what it can see.
    autonomy: str = "low"
    # Where Droid works. The deployment mounts its shared workspace here, so
    # Droid's file tools and an `AGENTS.md` the person keeps there are read
    # natively. Empty means the process's own directory, which is what a
    # deployment without the mount had before.
    workspace: str = ""


def _prompt_from(input_data: RunAgentInput) -> str:
    """The newest user message, which is what this run is being asked.

    History is not replayed: Droid's session carries it, keyed by thread below.
    """
    for message in reversed(input_data.messages):
        if message.role == "user" and (message.content or "").strip():
            return message.content
    return ""


def _texts(content) -> str:
    """Claude-shaped content is either a string or a list of typed blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def _tool_name(name: str) -> str:
    """The deployment's own name for a tool, with Droid's MCP namespacing removed."""
    return name[len(_MCP_PREFIX) :] if name.startswith(_MCP_PREFIX) else name


def run_context(input_data: RunAgentInput) -> dict | None:
    """What the MCP bridge needs to call this run's deployment tools, or None.

    The deployment marks which of the run's tools it runs itself
    (`openbotDeploymentTools`) and signs whose run this is (`openbotRun`), both
    in `forwardedProps` — the same contract the LangGraph Bot reads. Only those
    tools are offered to Droid: the rest are drawn by a surface this headless
    process cannot see, and offering them would produce calls nothing answers.
    """
    url = (os.environ.get("OPENBOT_TOOL_URL") or "").strip()
    token = (os.environ.get("AGENT_TOOL_TOKEN") or "").strip()
    if not url or not token:
        return None
    props = input_data.forwarded_props
    if not isinstance(props, dict):
        return None
    names = props.get("openbotDeploymentTools")
    ours = {name for name in names if isinstance(name, str)} if isinstance(names, list) else set()
    run = props.get("openbotRun")
    tools = [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
        for tool in input_data.tools or []
        if tool.name in ours
    ]
    if not tools:
        return None
    return {
        "url": url,
        "token": token,
        "run": run if isinstance(run, str) else "",
        "tools": tools,
    }


class DroidAdapter:
    def __init__(
        self,
        settings: DroidSettings | None = None,
        sessions: SessionStore | None = None,
    ):
        self._settings = settings or DroidSettings()
        # Threads outlive runs and, through the store's file, outlive this
        # process: see `sessions.py`. The in-memory dict is for tests only.
        self._sessions = sessions if sessions is not None else _EphemeralSessions()

    def _argv(self, prompt: str, thread_id: str) -> list[str]:
        settings = self._settings
        argv = [
            settings.command,
            "exec",
            "--output-format",
            "stream-json",
            "--auto",
            settings.autonomy,
        ]
        if settings.model:
            argv += ["--model", settings.model]
        session = self._sessions.get(thread_id)
        if session:
            argv += ["--session-id", session]
        argv.append(prompt)
        return argv

    async def run(self, input_data: RunAgentInput):
        thread_id, run_id = input_data.thread_id, input_data.run_id
        yield RunStartedEvent(thread_id=thread_id, run_id=run_id)

        prompt = _prompt_from(input_data)
        if not prompt:
            yield RunErrorEvent(
                message="The run carried no user message to send to Droid."
            )
            return

        try:
            context = run_context(input_data)
            run_file: Path | None = None
            environment = dict(os.environ)
            if context is not None:
                # Written before the spawn and named in the spawn's own
                # environment, so the MCP bridge Droid starts (see
                # `mcp_proxy.py`) serves exactly this run's tools on exactly
                # this run's signed statement. One file per run: two runs
                # answered at once must not read each other's grant.
                handle, name = tempfile.mkstemp(prefix="openbot-run-", suffix=".json")
                run_file = Path(name)
                with os.fdopen(handle, "w", encoding="utf-8") as file:
                    json.dump(context, file)
                environment["OPENBOT_RUN_FILE"] = str(run_file)
            process = await asyncio.create_subprocess_exec(
                *self._argv(prompt, thread_id),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._settings.workspace or None,
                env=environment,
            )
        except OSError as error:
            if run_file is not None:
                run_file.unlink(missing_ok=True)
            yield RunErrorEvent(message=f"Droid could not be started: {error}")
            return

        outcome: dict | None = None
        try:
            assert process.stdout is not None
            async for line in process.stdout:
                text = line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                try:
                    event = json.loads(text)
                except json.JSONDecodeError:
                    # The CLI prints the odd human line among the events; a line
                    # that is not an event is not a failure.
                    continue
                if not isinstance(event, dict):
                    continue
                kind = event.get("type")
                if kind == "system" and event.get("subtype") == "init":
                    session = str(event.get("session_id") or "").strip()
                    if session:
                        self._sessions.put(thread_id, session)
                elif kind == "assistant":
                    for block in (event.get("message") or {}).get("content") or []:
                        if not isinstance(block, dict):
                            continue
                        if block.get("type") == "text" and block.get("text"):
                            message_id = str(uuid.uuid4())
                            yield TextMessageStartEvent(
                                message_id=message_id, role="assistant"
                            )
                            yield TextMessageContentEvent(
                                message_id=message_id, delta=block["text"]
                            )
                            yield TextMessageEndEvent(message_id=message_id)
                        elif block.get("type") == "tool_use":
                            tool_call_id = str(block.get("id") or uuid.uuid4())
                            yield ToolCallStartEvent(
                                tool_call_id=tool_call_id,
                                tool_call_name=_tool_name(
                                    str(block.get("name") or "tool")
                                ),
                            )
                            yield ToolCallArgsEvent(
                                tool_call_id=tool_call_id,
                                delta=json.dumps(block.get("input") or {}),
                            )
                            yield ToolCallEndEvent(tool_call_id=tool_call_id)
                elif kind == "user":
                    for block in (event.get("message") or {}).get("content") or []:
                        if (
                            isinstance(block, dict)
                            and block.get("type") == "tool_result"
                            and block.get("tool_use_id")
                        ):
                            yield ToolCallResultEvent(
                                message_id=str(uuid.uuid4()),
                                tool_call_id=str(block["tool_use_id"]),
                                content=_texts(block.get("content")),
                            )
                elif kind == "result":
                    outcome = event

            stderr_tail = b""
            if process.stderr is not None:
                stderr_tail = (await process.stderr.read())[-_STDERR_TAIL_BYTES:]
            returncode = await process.wait()

            if outcome is not None and outcome.get("is_error"):
                yield RunErrorEvent(
                    message=_texts(outcome.get("result")) or "Droid reported an error."
                )
                return
            if outcome is None and returncode != 0:
                detail = stderr_tail.decode("utf-8", errors="replace").strip()
                yield RunErrorEvent(
                    message=detail or f"Droid exited with status {returncode}."
                )
                return
            yield RunFinishedEvent(thread_id=thread_id, run_id=run_id)
        finally:
            # A consumer that goes away must not leave a droid process behind,
            # and a finished run must not leave its grant readable on disk.
            if process.returncode is None:
                process.kill()
                await process.wait()
            if run_file is not None:
                run_file.unlink(missing_ok=True)


class _EphemeralSessions:
    """The store's shape without its file, for an adapter built bare in tests."""

    def __init__(self):
        self._sessions: dict[str, str] = {}

    def get(self, thread_id: str) -> str:
        return self._sessions.get(thread_id, "")

    def put(self, thread_id: str, session_id: str) -> None:
        self._sessions[thread_id] = session_id
