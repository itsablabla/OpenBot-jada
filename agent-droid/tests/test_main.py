"""The endpoint contract: token, health, credential refusal, and the SSE shape."""

import importlib
import json
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

TOKEN = "test-token"
RUN = {
    "threadId": "thread-1",
    "runId": "run-1",
    "state": {},
    "messages": [{"id": "m1", "role": "user", "content": "Say hello"}],
    "tools": [],
    "context": [],
    "forwardedProps": {},
}


def load_main(monkeypatch, **env):
    """Import `src.main` fresh under the environment a test decides."""
    for variable in (
        "FACTORY_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "BOT_PROVIDER",
        "BOT_MODEL",
        "OPENBOT_TOOL_URL",
        "AGENT_TOOL_TOKEN",
        "DROID_TRANSPORT",
        "DROID_AUTONOMY",
        "DROID_WORKSPACE",
    ):
        monkeypatch.delenv(variable, raising=False)
    for variable, value in env.items():
        monkeypatch.setenv(variable, value)
    monkeypatch.setenv("MANAGED_AGENT_TOKEN", TOKEN)
    sys.modules.pop("src.main", None)
    return importlib.import_module("src.main")


def test_health_answers_without_a_token_and_names_the_harness(monkeypatch, tmp_path):
    main = load_main(monkeypatch, FACTORY_API_KEY="fk-1", HOME=str(tmp_path))
    client = TestClient(main.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"ok": True, "harness": "droid"}


def test_everything_else_refuses_without_the_server_token(monkeypatch, tmp_path):
    main = load_main(monkeypatch, FACTORY_API_KEY="fk-1", HOME=str(tmp_path))
    client = TestClient(main.app)
    assert client.post("/", json=RUN).status_code == 401
    assert (
        client.post("/", json=RUN, headers={"x-openbot-agent-token": "wrong"}).status_code
        == 401
    )


def test_an_authorised_run_streams_ag_ui_events(monkeypatch, tmp_path):
    fake = tmp_path / "droid"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        'print(json.dumps({"type": "assistant", "message": {"content": '
        '[{"type": "text", "text": "Hello!"}]}}))\n'
        'print(json.dumps({"type": "result", "is_error": False, "result": "Hello!"}))\n',
        encoding="utf-8",
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)

    main = load_main(monkeypatch, FACTORY_API_KEY="fk-1", HOME=str(tmp_path))
    from src.adapter import DroidAdapter, DroidSettings

    main.adapter = DroidAdapter(DroidSettings(command=str(fake)))
    client = TestClient(main.app)
    response = client.post("/", json=RUN, headers={"x-openbot-agent-token": TOKEN})
    assert response.status_code == 200

    types = [
        json.loads(line.removeprefix("data: "))["type"]
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert types == [
        "RUN_STARTED",
        "TEXT_MESSAGE_START",
        "TEXT_MESSAGE_CONTENT",
        "TEXT_MESSAGE_END",
        "RUN_FINISHED",
    ]


def test_a_factory_key_needs_no_byok_config(monkeypatch, tmp_path):
    load_main(monkeypatch, FACTORY_API_KEY="fk-1", HOME=str(tmp_path))
    assert not (tmp_path / ".factory" / "config.json").exists()


def test_a_provider_key_becomes_droids_own_byok_config(monkeypatch, tmp_path):
    main = load_main(monkeypatch, OPENAI_API_KEY="sk-test", HOME=str(tmp_path))
    written = json.loads(
        (tmp_path / ".factory" / "config.json").read_text(encoding="utf-8")
    )
    (model,) = written["custom_models"]
    assert model["model_display_name"] == "openbot"
    assert model["api_key"] == "sk-test"
    assert model["provider"] == "openai"
    assert model["base_url"] == "https://api.openai.com/v1"
    assert main.adapter._settings.model == "openbot"


def test_no_credential_at_all_is_refused_at_startup(monkeypatch, tmp_path):
    with pytest.raises(SystemExit) as refusal:
        load_main(monkeypatch, HOME=str(tmp_path))
    assert "FACTORY_API_KEY" in str(refusal.value)
    assert "OPENAI_API_KEY" in str(refusal.value)


def test_the_deployment_tool_bridge_is_registered_with_droid(monkeypatch, tmp_path):
    load_main(
        monkeypatch,
        FACTORY_API_KEY="fk-1",
        HOME=str(tmp_path),
        OPENBOT_TOOL_URL="http://server.test/api/agent-tools/call",
        AGENT_TOOL_TOKEN="tool-token",
    )
    config = json.loads(
        (tmp_path / ".factory" / "mcp.json").read_text(encoding="utf-8")
    )
    server = config["mcpServers"]["openbot"]
    assert server["type"] == "stdio"
    assert server["args"][-1].endswith("mcp_proxy.py")


def test_no_callback_means_no_bridge_in_droids_config(monkeypatch, tmp_path):
    load_main(monkeypatch, FACTORY_API_KEY="fk-1", HOME=str(tmp_path))
    assert not (tmp_path / ".factory" / "mcp.json").exists()


def test_the_default_transport_persists_sessions_under_the_factory_home(
    monkeypatch, tmp_path
):
    from src.adapter import DroidAdapter
    from src.sessions import SessionStore

    main = load_main(monkeypatch, FACTORY_API_KEY="fk-1", HOME=str(tmp_path))
    assert isinstance(main.adapter, DroidAdapter)
    assert isinstance(main.adapter._sessions, SessionStore)


def test_the_acp_transport_is_chosen_by_environment(monkeypatch, tmp_path):
    from src.acp import AcpDroid

    main = load_main(
        monkeypatch, FACTORY_API_KEY="fk-1", HOME=str(tmp_path), DROID_TRANSPORT="acp"
    )
    assert isinstance(main.adapter, AcpDroid)


def test_autonomy_and_workspace_come_from_the_environment(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    main = load_main(
        monkeypatch,
        FACTORY_API_KEY="fk-1",
        HOME=str(tmp_path),
        DROID_AUTONOMY="medium",
        DROID_WORKSPACE=str(workspace),
    )
    assert main._settings.autonomy == "medium"
    assert main._settings.workspace == str(workspace)


def test_a_missing_workspace_directory_is_simply_not_used(monkeypatch, tmp_path):
    main = load_main(
        monkeypatch,
        FACTORY_API_KEY="fk-1",
        HOME=str(tmp_path),
        DROID_WORKSPACE=str(tmp_path / "nowhere"),
    )
    assert main._settings.workspace == ""
