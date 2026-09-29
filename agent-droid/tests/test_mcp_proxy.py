"""The MCP bridge: listing, calling, and refusing, all against the run file."""

import io
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import mcp_proxy


def ask(monkeypatch, run_file, *messages):
    """Drive the stdio server with a scripted conversation, returning replies."""
    if run_file is None:
        monkeypatch.delenv("OPENBOT_RUN_FILE", raising=False)
    else:
        monkeypatch.setenv("OPENBOT_RUN_FILE", str(run_file))
    stdin = io.StringIO("".join(json.dumps(m) + "\n" for m in messages))
    stdout = io.StringIO()
    mcp_proxy.serve(stdin, stdout)
    return [json.loads(line) for line in stdout.getvalue().splitlines()]


def run_file_with(tmp_path, **overrides):
    context = {
        "url": "http://deployment.test/api/agent-tools/call",
        "token": "tool-token",
        "run": "signed-run-statement",
        "tools": [
            {
                "name": "list_channels",
                "description": "The channels this deployment has.",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
    }
    context.update(overrides)
    path = tmp_path / "run.json"
    path.write_text(json.dumps(context), encoding="utf-8")
    return path


def test_initialize_answers_with_the_tools_capability(monkeypatch, tmp_path):
    replies = ask(
        monkeypatch,
        None,
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
    )
    assert replies[0]["result"]["protocolVersion"] == "2025-06-18"
    assert "tools" in replies[0]["result"]["capabilities"]
    assert replies[0]["result"]["serverInfo"]["name"] == "openbot"


def test_the_listing_is_the_run_files_tools(monkeypatch, tmp_path):
    replies = ask(
        monkeypatch,
        run_file_with(tmp_path),
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
    )
    tools = replies[0]["result"]["tools"]
    assert [tool["name"] for tool in tools] == ["list_channels"]
    assert tools[0]["inputSchema"] == {"type": "object", "properties": {}}


def test_no_run_file_means_an_empty_listing_not_a_crash(monkeypatch):
    replies = ask(monkeypatch, None, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert replies[0]["result"]["tools"] == []


def test_a_call_posts_the_signed_statement_back_to_the_deployment(
    monkeypatch, tmp_path
):
    seen = {}

    class Answer:
        def read(self):
            return b"three channels"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["token"] = request.get_header("X-openbot-agent-token")
        seen["body"] = json.loads(request.data)
        return Answer()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    replies = ask(
        monkeypatch,
        run_file_with(tmp_path),
        {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {"name": "list_channels", "arguments": {"kind": "text"}},
        },
    )
    assert seen["url"] == "http://deployment.test/api/agent-tools/call"
    assert seen["token"] == "tool-token"
    assert seen["body"] == {
        "name": "list_channels",
        "args": {"kind": "text"},
        "run": "signed-run-statement",
    }
    assert replies[0]["result"]["content"] == [
        {"type": "text", "text": "three channels"}
    ]
    assert "isError" not in replies[0]["result"]


def test_a_run_with_no_signed_statement_is_refused_in_words(monkeypatch, tmp_path):
    replies = ask(
        monkeypatch,
        run_file_with(tmp_path, run=""),
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "list_channels", "arguments": {}},
        },
    )
    result = replies[0]["result"]
    assert result["isError"] is True
    assert "signed statement" in result["content"][0]["text"]


def test_notifications_get_no_reply_and_unknown_methods_get_an_error(
    monkeypatch, tmp_path
):
    replies = ask(
        monkeypatch,
        None,
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "resources/list"},
    )
    assert len(replies) == 1
    assert replies[0]["error"]["code"] == -32601
