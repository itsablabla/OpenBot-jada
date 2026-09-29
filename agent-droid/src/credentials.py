"""What Droid signs its model calls with, decided once at startup.

Two credential paths, never mixed. `FACTORY_API_KEY` is Factory's own: Droid
uses it as it is and picks its models from Factory's catalogue. Without one,
the key the model screen chose is turned into a Droid BYOK ("bring your own
key") custom model, written to `~/.factory/config.json` before the CLI first
runs — which provider that is arrives through `BOT_PROVIDER` and the key
variable that `shared/model-providers.json` names for it, the same contract
every other Bot reads.

Neither being present is refused at startup with both remedies named, because
the alternative is a correct-looking Bot whose first answer is a credential
error from a process the person cannot see.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from model_providers import SPEC, bot_settings  # noqa: E402

BOT_ID = "agent-droid"

# The name the BYOK model is registered and addressed by. One name, because the
# config file is rewritten whole at startup and this Bot registers exactly one.
BYOK_MODEL_NAME = "openbot"

# Where a provider's API lives when its base URL variable is unset. Droid's
# BYOK config takes the full base URL; these are the providers' published ones.
_DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai/",
}

# Droid's BYOK rows name the wire protocol, not the vendor. Google is spoken to
# through its OpenAI-compatible endpoint, which is the generic row.
_BYOK_PROVIDERS = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google": "generic-chat-completion-api",
}


@dataclass(frozen=True)
class Credentials:
    """The one decision: which model flag the spawn carries, and what to write."""

    # The `--model` value, or empty for Droid's own default on a Factory key.
    model: str
    # The BYOK config to write, or None when the Factory key is in charge.
    config: dict | None


def resolve(environ=None) -> Credentials:
    env = os.environ if environ is None else environ
    factory_key = (env.get("FACTORY_API_KEY") or "").strip()
    if factory_key:
        # A model the environment pins is passed through; otherwise Droid's own
        # default model answers, which is the point of the Factory path.
        return Credentials(model=(env.get("BOT_MODEL") or "").strip(), config=None)

    settings = bot_settings(BOT_ID, env=env)
    provider_row = SPEC["providers"].get(settings.provider)
    if provider_row is None:
        raise SystemExit(
            f"BOT_PROVIDER is {settings.provider!r}, which shared/model-providers.json has no "
            "row for. Set FACTORY_API_KEY, or a provider this deployment knows."
        )
    key = (env.get(provider_row["key_variable"]) or "").strip()
    if not key:
        raise SystemExit(
            f"Set FACTORY_API_KEY for Factory's own models, or {provider_row['key_variable']} "
            f"so Droid can drive {provider_row['label']} with your key."
        )
    base_url = (env.get(provider_row["base_url_variable"]) or "").strip() or (
        _DEFAULT_BASE_URLS[settings.provider]
    )
    return Credentials(
        model=BYOK_MODEL_NAME,
        config={
            "custom_models": [
                {
                    "model_display_name": BYOK_MODEL_NAME,
                    "model": settings.model,
                    "base_url": base_url,
                    "api_key": key,
                    "provider": _BYOK_PROVIDERS[settings.provider],
                    "max_tokens": 16384,
                }
            ]
        },
    )


def write_config(config: dict, home: Path | None = None) -> Path:
    """Lay the BYOK config where the CLI reads it, replacing what was there.

    Replaced whole rather than merged: the file's only writer in this container
    is this function, and a stale custom model from a previous key is exactly
    the kind of leftover that answers with the wrong bill.
    """
    directory = (home or Path.home()) / ".factory"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path
