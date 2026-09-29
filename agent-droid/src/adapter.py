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
import uuid
from dataclasses import dataclass

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

# How much of stderr is kept for the error message when the CLI dies without a
# `result` event. Enough to say why, bounded so a looping process cannot grow it.
_STDERR_TAIL_BYTES = 4096


@dataclass(frozen=True)
class DroidSettings:
    """What one spawn needs to know beyond the prompt."""

    command: str = "droid"
    # The BYOK custom model's display name, when main.py wrote one, or a model
    # the environment pinned. Empty means Droid's own default.
    model: str = ""
    # Droid runs its own local tools; "low" keeps it to reads unless the
    # deployment says otherwise, because this Bot answers chat rather than
    # editing a repository it owns.
    autonomy: str = "low"


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


class DroidAdapter:
    def __init__(self, settings: DroidSettings | None = None):
        self._settings = settings or DroidSettings()
        self._sessions: dict[str, str] = {}

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
        session = self._sessions.get(thread_id, "")
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
            process = await asyncio.create_subprocess_exec(
                *self._argv(prompt, thread_id),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as error:
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
                        self._sessions[thread_id] = session
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
                                tool_call_name=str(block.get("name") or "tool"),
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
            # A consumer that goes away must not leave a droid process behind.
            if process.returncode is None:
                process.kill()
                await process.wait()
