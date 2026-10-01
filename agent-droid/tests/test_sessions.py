"""The thread→session map's file: what survives, and what failure costs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sessions import SessionStore


def test_a_session_survives_a_new_store_over_the_same_file(tmp_path):
    path = tmp_path / "threads.json"
    SessionStore(path).put("thread-1", "session-1")
    assert SessionStore(path).get("thread-1") == "session-1"


def test_threads_do_not_share_sessions(tmp_path):
    store = SessionStore(tmp_path / "threads.json")
    store.put("thread-1", "session-1")
    store.put("thread-2", "session-2")
    assert store.get("thread-1") == "session-1"
    assert store.get("thread-2") == "session-2"


def test_a_missing_or_broken_file_means_a_fresh_start_not_a_crash(tmp_path):
    path = tmp_path / "threads.json"
    assert SessionStore(path).get("thread-1") == ""
    path.write_text("not json at all", encoding="utf-8")
    store = SessionStore(path)
    assert store.get("thread-1") == ""
    # And it heals: the next write replaces the broken file whole.
    store.put("thread-1", "session-1")
    assert SessionStore(path).get("thread-1") == "session-1"


def test_the_parent_directory_is_made_when_absent(tmp_path):
    path = tmp_path / "deep" / "down" / "threads.json"
    SessionStore(path).put("thread-1", "session-1")
    assert SessionStore(path).get("thread-1") == "session-1"
