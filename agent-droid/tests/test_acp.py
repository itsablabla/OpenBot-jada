"""The ACP transport: one process, protocol messages in, AG-UI events out.

The stand-in agent is a real subprocess speaking newline JSON-RPC, because the
transport owns process lifetime, request ids and interleaved notifications, and
those only fail honestly over real pipes.
"""

import asyncio
import json
import os
import stat
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ag_ui.core import EventType, RunAgentInput

from src.acp import AcpDroid
from src.adapter import DroidSettings

RUN = {
    "thread_id": "thread-1",
    "run_id": "run-1",
    "state": {},
    "messages": [{"id": "m1", "role": "user", "content": "Say hello"}],
    "tools": [],
    "context": [],
    "forwarded_props": {},
}

AGENT = '''#!/usr/bin/env python3
import json, os, sys

script = json.load(open(os.environ["ACP_SCRIPT"]))
log = open(os.environ["ACP_LOG"], "a")

def say(message):
    sys.stdout.write(json.dumps(message) + "\\n")
    sys.stdout.flush()

sessions = 0
for line in sys.stdin:
    message = json.loads(line)
    log.write(json.dumps(message) + "\\n")
    log.flush()
    method = message.get("method")
    if method == "initialize":
        say({"jsonrpc": "2.0", "id": message["id"],
             "result": {"protocolVersion": 1, "agentCapabilities": {}}})
    elif method == "session/new":
        sessions += 1
        say({"jsonrpc": "2.0", "id": message["id"],
             "result": {"sessionId": "acp-session-%d" % sessions}})
    elif method == "session/prompt":
        session = message["params"]["sessionId"]
        if script.get("ask_permission"):
            say({"jsonrpc": "2.0", "id": 900, "method": "session/request_permission",
                 "params": {"sessionId": session, "toolCall": {"toolCallId": "t1"},
                            "options": [
                                {"optionId": "yes", "name": "Allow", "kind": "allow_once"},
                                {"optionId": "no", "name": "Reject", "kind": "reject_once"},
                            ]}})
            answer = json.loads(sys.stdin.readline())
            log.write(json.dumps({"permission_answer": answer}) + "\\n")
            log.flush()
        for update in script.get("updates", []):
            say({"jsonrpc": "2.0", "method": "session/update",
                 "params": {"sessionId": session, "update": update}})
        say({"jsonrpc": "2.0", "id": message["id"],
             "result": {"stopReason": script.get("stop_reason", "end_turn")}})
'''


def fake_acp_agent(tmp_path: Path, script: dict) -> DroidSettings:
    agent = tmp_path / "droid"
    agent.write_text(AGENT, encoding="utf-8")
    agent.chmod(agent.stat().st_mode | stat.S_IEXEC)
    (tmp_path / "script.json").write_text(json.dumps(script), encoding="utf-8")
    os.environ["ACP_SCRIPT"] = str(tmp_path / "script.json")
    os.environ["ACP_LOG"] = str(tmp_path / "log.jsonl")
    return DroidSettings(command=str(agent))


def logged(tmp_path: Path):
    return [
        json.loads(line)
        for line in (tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def collect(adapter: AcpDroid, runs=1):
    async def gather():
        rounds = []
        try:
            for index in range(runs):
                events = []
                run = dict(RUN, run_id=f"run-{index + 1}")
                async for event in adapter.run(RunAgentInput(**run)):
                    events.append(event)
                rounds.append(events)
        finally:
            await adapter.aclose()
        return rounds

    return asyncio.run(gather())


def kinds(events):
    return [event.type for event in events]


def test_a_prompt_becomes_the_same_ag_ui_stream_the_exec_transport_speaks(tmp_path):
    settings = fake_acp_agent(
        tmp_path,
        {
            "updates": [
                {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "Hello"},
                },
                {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": " there"},
                },
                {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "call-1",
                    "title": "list_channels",
                    "kind": "fetch",
                    "status": "pending",
                    "rawInput": {"kind": "text"},
                },
                {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": "call-1",
                    "status": "completed",
                    "content": [
                        {"type": "content", "content": {"type": "text", "text": "three"}}
                    ],
                },
            ]
        },
    )
    [events] = collect(AcpDroid(settings))
    assert kinds(events) == [
        EventType.RUN_STARTED,
        EventType.TEXT_MESSAGE_START,
        EventType.TEXT_MESSAGE_CONTENT,
        EventType.TEXT_MESSAGE_CONTENT,
        EventType.TEXT_MESSAGE_END,
        EventType.TOOL_CALL_START,
        EventType.TOOL_CALL_ARGS,
        EventType.TOOL_CALL_END,
        EventType.TOOL_CALL_RESULT,
        EventType.RUN_FINISHED,
    ]
    deltas = [e.delta for e in events if e.type == EventType.TEXT_MESSAGE_CONTENT]
    assert deltas == ["Hello", " there"]
    start = next(e for e in events if e.type == EventType.TOOL_CALL_START)
    assert start.tool_call_name == "list_channels"
    result = next(e for e in events if e.type == EventType.TOOL_CALL_RESULT)
    assert result.content == "three"


def test_one_thread_is_one_acp_session_across_runs(tmp_path):
    settings = fake_acp_agent(tmp_path, {"updates": []})
    collect(AcpDroid(settings), runs=2)
    sessions = [m for m in logged(tmp_path) if m.get("method") == "session/new"]
    prompts = [m for m in logged(tmp_path) if m.get("method") == "session/prompt"]
    assert len(sessions) == 1
    assert [p["params"]["sessionId"] for p in prompts] == [
        "acp-session-1",
        "acp-session-1",
    ]
    # And initialize happened exactly once, for the one process.
    assert sum(1 for m in logged(tmp_path) if m.get("method") == "initialize") == 1


def test_low_autonomy_declines_a_permission_request(tmp_path):
    settings = fake_acp_agent(tmp_path, {"ask_permission": True, "updates": []})
    [events] = collect(AcpDroid(settings))
    assert kinds(events)[-1] == EventType.RUN_FINISHED
    answer = next(m for m in logged(tmp_path) if "permission_answer" in m)
    outcome = answer["permission_answer"]["result"]["outcome"]
    assert outcome == {"outcome": "selected", "optionId": "no"}


def test_higher_autonomy_allows_a_permission_request_once(tmp_path):
    settings = fake_acp_agent(tmp_path, {"ask_permission": True, "updates": []})
    settings = DroidSettings(command=settings.command, autonomy="medium")
    collect(AcpDroid(settings))
    answer = next(m for m in logged(tmp_path) if "permission_answer" in m)
    assert answer["permission_answer"]["result"]["outcome"]["optionId"] == "yes"


def test_a_refusal_is_a_run_error_not_a_quiet_finish(tmp_path):
    settings = fake_acp_agent(tmp_path, {"updates": [], "stop_reason": "refusal"})
    [events] = collect(AcpDroid(settings))
    assert kinds(events)[-1] == EventType.RUN_ERROR
    assert "refusal" in events[-1].message


def test_a_missing_binary_is_a_run_error_not_a_crash(tmp_path):
    settings = DroidSettings(command=str(tmp_path / "no-such-droid"))
    [events] = collect(AcpDroid(settings))
    assert kinds(events) == [EventType.RUN_STARTED, EventType.RUN_ERROR]
    assert "could not be reached" in events[-1].message
