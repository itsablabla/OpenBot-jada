"""The credential decision: which path wins, and what gets written for it."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.credentials import BYOK_MODEL_NAME, resolve


def test_a_custom_endpoint_wins_over_a_factory_key(monkeypatch):
    monkeypatch.delenv("BOT_PROVIDER", raising=False)
    monkeypatch.delenv("BOT_MODEL", raising=False)
    credentials = resolve(
        {
            "FACTORY_API_KEY": "fk-1",
            "DROID_BASE_URL": "http://ollama:11434/v1",
            "DROID_MODEL": "llama3",
        }
    )
    assert credentials.model == BYOK_MODEL_NAME
    [row] = credentials.config["custom_models"]
    assert row["base_url"] == "http://ollama:11434/v1"
    assert row["model"] == "llama3"
    assert row["provider"] == "generic-chat-completion-api"


def test_an_endpoint_that_wants_no_key_gets_the_placeholder(monkeypatch):
    credentials = resolve(
        {"DROID_BASE_URL": "http://ollama:11434/v1", "DROID_MODEL": "llama3"}
    )
    [row] = credentials.config["custom_models"]
    assert row["api_key"] == "not-needed"


def test_an_endpoint_key_is_carried_when_given(monkeypatch):
    credentials = resolve(
        {
            "DROID_BASE_URL": "https://models.example/v1",
            "DROID_MODEL": "their-model",
            "DROID_MODEL_API_KEY": "ek-1",
        }
    )
    [row] = credentials.config["custom_models"]
    assert row["api_key"] == "ek-1"


def test_the_model_falls_back_to_bot_model(monkeypatch):
    credentials = resolve(
        {"DROID_BASE_URL": "http://ollama:11434/v1", "BOT_MODEL": "llama3"}
    )
    [row] = credentials.config["custom_models"]
    assert row["model"] == "llama3"


def test_an_endpoint_without_a_model_is_refused_in_words(monkeypatch):
    with pytest.raises(SystemExit, match="DROID_MODEL"):
        resolve({"DROID_BASE_URL": "http://ollama:11434/v1"})


def test_a_factory_key_still_runs_factory_models_without_an_endpoint(monkeypatch):
    credentials = resolve({"FACTORY_API_KEY": "fk-1"})
    assert credentials.config is None
    assert credentials.model == ""
