"""Where the thread-to-session map lives, so continuity survives a restart.

Droid's own continuity is its session, and OpenBot's is its thread. The map
between them used to live in adapter memory, which meant every container
restart silently started every conversation over. It now lives in a JSON file
beside Droid's own state in `~/.factory`, which Compose mounts as a named
volume for exactly this reason: the two kinds of state age together.

The file is tiny and rewritten whole on every new session, atomically, by
writing a sibling and replacing. A reader never sees half a file, and a file
that cannot be read is treated as empty rather than fatal: the worst outcome
is a conversation that starts over, which is where it would have been anyway.
"""

import json
import os
from pathlib import Path


class SessionStore:
    def __init__(self, path: Path):
        self._path = path

    def _load(self) -> dict[str, str]:
        try:
            found = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(found, dict):
            return {}
        return {
            str(thread): str(session)
            for thread, session in found.items()
            if isinstance(thread, str) and isinstance(session, str)
        }

    def get(self, thread_id: str) -> str:
        return self._load().get(thread_id, "")

    def put(self, thread_id: str, session_id: str) -> None:
        sessions = self._load()
        if sessions.get(thread_id) == session_id:
            return
        sessions[thread_id] = session_id
        self._path.parent.mkdir(parents=True, exist_ok=True)
        sibling = self._path.with_name(self._path.name + ".tmp")
        sibling.write_text(json.dumps(sessions, indent=2), encoding="utf-8")
        os.replace(sibling, self._path)
