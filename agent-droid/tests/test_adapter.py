"""The stream-json → AG-UI translation, exercised against a stand-in CLI.

The stand-in is a real subprocess: what the adapter owns is spawning, parsing
and killing a process, so the tests drive exactly that seam rather than a
mocked stream.
"""

import asyncio
import json
import tempfile
import time
import os
import stat
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ag_ui.core import EventType, RunAgentInput

from src.adapter import DroidAdapter, DroidSettings

RUN = {
    "thread_id": "thread-1",
    "run_id": "run-1",
    "state": {},
    "messages": [{"id": "m1", "role": "user", "content": "Say hello"}],
    "tools": [],
    "context": [],
    "forwarded_props": {},
}


def fake_droid(tmp_path: Path, body: str) -> str:
    """A `droid` that records its argv and prints what the test scripted."""
    path = tmp_path / "droid"
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        f"open({str(tmp_path / 'argv.json')!r}, 'a').write(json.dumps(sys.argv[1:]) + '\\n')\n"
        + body,
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def collect(adapter: DroidAdapter, run=None):
    async def gather():
        events = []
        async for event in adapter.run(RunAgentInput(**(run or RUN))):
            events.append(event)
        return events

    return asyncio.run(gather())


def argv_lines(tmp_path: Path):
    return [
        json.loads(line)
        for line in (tmp_path / "argv.json").read_text(encoding="utf-8").splitlines()
    ]


def test_a_full_stream_becomes_text_tools_and_a_finish(tmp_path):
    stream = [
        {"type": "system", "subtype": "init", "session_id": "sess-9"},
        "this line is not JSON and must be ignored",
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "Let me check."},
                    {"type": "tool_use", "id": "tu-1", "name": "Read", "input": {"path": "a.txt"}},
                ]
            },
        },
        {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu-1",
                        "content": [{"type": "text", "text": "contents"}],
                    }
                ]
            },
        },
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "Hello!"}]}},
        {"type": "result", "is_error": False, "result": "Hello!"},
    ]
    body = "".join(
        f"print({json.dumps(line if isinstance(line, str) else json.dumps(line))})\n"
        for line in stream
    )
    adapter = DroidAdapter(DroidSettings(command=fake_droid(tmp_path, body)))
    events = collect(adapter)

    assert [event.type for event in events] == [
        EventType.RUN_STARTED,
        EventType.TEXT_MESSAGE_START,
        EventType.TEXT_MESSAGE_CONTENT,
        EventType.TEXT_MESSAGE_END,
        EventType.TOOL_CALL_START,
        EventType.TOOL_CALL_ARGS,
        EventType.TOOL_CALL_END,
        EventType.TOOL_CALL_RESULT,
        EventType.TEXT_MESSAGE_START,
        EventType.TEXT_MESSAGE_CONTENT,
        EventType.TEXT_MESSAGE_END,
        EventType.RUN_FINISHED,
    ]
    assert events[2].delta == "Let me check."
    assert events[4].tool_call_name == "Read"
    assert json.loads(events[5].delta) == {"path": "a.txt"}
    assert events[7].tool_call_id == "tu-1"
    assert events[7].content == "contents"


def test_the_second_run_of_a_thread_reuses_droids_session(tmp_path):
    body = (
        'print(json.dumps({"type": "system", "subtype": "init", "session_id": "sess-9"}))\n'
        'print(json.dumps({"type": "result", "is_error": False, "result": "ok"}))\n'
    )
    adapter = DroidAdapter(DroidSettings(command=fake_droid(tmp_path, body)))
    collect(adapter)
    collect(adapter)

    first, second = argv_lines(tmp_path)
    assert "--session-id" not in first
    index = second.index("--session-id")
    assert second[index + 1] == "sess-9"
    # A different thread must not inherit the session.
    collect(adapter, {**RUN, "thread_id": "thread-2"})
    assert "--session-id" not in argv_lines(tmp_path)[2]


def test_the_model_flag_is_carried_when_settings_name_one(tmp_path):
    body = 'print(json.dumps({"type": "result", "is_error": False, "result": "ok"}))\n'
    adapter = DroidAdapter(
        DroidSettings(command=fake_droid(tmp_path, body), model="openbot")
    )
    collect(adapter)
    argv = argv_lines(tmp_path)[0]
    assert argv[argv.index("--model") + 1] == "openbot"
    assert argv[-1] == "Say hello"


def test_an_error_result_becomes_a_run_error(tmp_path):
    body = 'print(json.dumps({"type": "result", "is_error": True, "result": "Not signed in."}))\n'
    adapter = DroidAdapter(DroidSettings(command=fake_droid(tmp_path, body)))
    events = collect(adapter)
    assert events[-1].type == EventType.RUN_ERROR
    assert events[-1].message == "Not signed in."


def test_a_crash_without_a_result_reports_stderr(tmp_path):
    body = "sys.stderr.write('droid: exploded\\n')\nsys.exit(3)\n"
    adapter = DroidAdapter(DroidSettings(command=fake_droid(tmp_path, body)))
    events = collect(adapter)
    assert events[-1].type == EventType.RUN_ERROR
    assert "exploded" in events[-1].message


def test_a_missing_binary_is_an_error_event_rather_than_an_exception(tmp_path):
    adapter = DroidAdapter(DroidSettings(command=str(tmp_path / "absent")))
    events = collect(adapter)
    assert [event.type for event in events] == [
        EventType.RUN_STARTED,
        EventType.RUN_ERROR,
    ]
    assert "could not be started" in events[-1].message


def test_a_run_without_a_user_message_is_refused(tmp_path):
    adapter = DroidAdapter(DroidSettings(command=str(tmp_path / "never-spawned")))
    events = collect(adapter, {**RUN, "messages": []})
    assert [event.type for event in events] == [
        EventType.RUN_STARTED,
        EventType.RUN_ERROR,
    ]
    assert not (tmp_path / "argv.json").exists()


TOOLED_RUN = {
    **RUN,
    "tools": [
        {
            "name": "list_channels",
            "description": "The channels this deployment has.",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "ask_the_user",
            "description": "A frontend tool the deployment does not run.",
            "parameters": {"type": "object", "properties": {}},
        },
    ],
    "forwarded_props": {
        "openbotDeploymentTools": ["list_channels"],
        "openbotRun": "signed-run-statement",
    },
}


def snooping_droid(tmp_path: Path) -> str:
    """A `droid` that records its cwd and the run file it inherited."""
    return fake_droid(
        tmp_path,
        "import os\n"
        "run_file = os.environ.get('OPENBOT_RUN_FILE', '')\n"
        "seen = {'cwd': os.getcwd(),\n"
        "        'run': json.load(open(run_file)) if run_file else None}\n"
        f"open({str(tmp_path / 'seen.json')!r}, 'w').write(json.dumps(seen))\n"
        'print(json.dumps({"type": "result", "is_error": False, "result": "ok"}))\n',
    )


def seen(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "seen.json").read_text(encoding="utf-8"))


def test_droid_works_inside_the_settings_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = DroidAdapter(
        DroidSettings(command=snooping_droid(tmp_path), workspace=str(workspace))
    )
    collect(adapter)
    assert seen(tmp_path)["cwd"] == str(workspace)


def test_deployment_tools_reach_droid_through_a_run_file(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENBOT_TOOL_URL", "http://server.test/api/agent-tools/call")
    monkeypatch.setenv("AGENT_TOOL_TOKEN", "tool-token")
    adapter = DroidAdapter(DroidSettings(command=snooping_droid(tmp_path)))
    collect(adapter, TOOLED_RUN)
    context = seen(tmp_path)["run"]
    assert context["url"] == "http://server.test/api/agent-tools/call"
    assert context["token"] == "tool-token"
    assert context["run"] == "signed-run-statement"
    # Only the deployment's own tools are offered; frontend tools have no
    # server to answer them from a headless process.
    assert [tool["name"] for tool in context["tools"]] == ["list_channels"]


def test_the_run_file_does_not_outlive_the_run(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENBOT_TOOL_URL", "http://server.test/api/agent-tools/call")
    monkeypatch.setenv("AGENT_TOOL_TOKEN", "tool-token")
    adapter = DroidAdapter(DroidSettings(command=snooping_droid(tmp_path)))
    collect(adapter, TOOLED_RUN)
    run_file = seen(tmp_path)["run"]
    assert run_file is not None  # it existed during the run…
    leftovers = [
        p
        for p in Path(tempfile.gettempdir()).glob("openbot-run-*.json")
        if p.stat().st_mtime > time.time() - 60
    ]
    assert leftovers == []  # …and is gone after it.


def test_without_a_callback_no_run_file_is_offered(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENBOT_TOOL_URL", raising=False)
    monkeypatch.delenv("AGENT_TOOL_TOKEN", raising=False)
    adapter = DroidAdapter(DroidSettings(command=snooping_droid(tmp_path)))
    collect(adapter, TOOLED_RUN)
    assert seen(tmp_path)["run"] is None


def test_a_bridged_tool_call_loses_its_mcp_prefix(tmp_path):
    body = (
        "print(json.dumps({'type': 'assistant', 'message': {'content': ["
        "{'type': 'tool_use', 'id': 'tu-1', 'name': 'mcp__openbot__list_channels',"
        " 'input': {}}]}}))\n"
        'print(json.dumps({"type": "result", "is_error": False, "result": "ok"}))\n'
    )
    adapter = DroidAdapter(DroidSettings(command=fake_droid(tmp_path, body)))
    events = collect(adapter)
    start = next(e for e in events if e.type == EventType.TOOL_CALL_START)
    assert start.tool_call_name == "list_channels"


def test_a_session_store_carries_the_thread_across_adapter_restarts(tmp_path):
    from src.sessions import SessionStore

    body = (
        'print(json.dumps({"type": "system", "subtype": "init", "session_id": "sess-9"}))\n'
        'print(json.dumps({"type": "result", "is_error": False, "result": "ok"}))\n'
    )
    command = fake_droid(tmp_path, body)
    store_path = tmp_path / "threads.json"
    collect(DroidAdapter(DroidSettings(command=command), SessionStore(store_path)))
    # A new adapter — a restarted container — resumes the same conversation.
    collect(DroidAdapter(DroidSettings(command=command), SessionStore(store_path)))
    second = argv_lines(tmp_path)[1]
    assert second[second.index("--session-id") + 1] == "sess-9"
