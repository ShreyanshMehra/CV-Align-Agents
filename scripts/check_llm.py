"""Quick connectivity check for the configured LLM provider.

Usage (from the project root, with the virtualenv active):

    python scripts/check_llm.py

Reads configuration from your .env file, builds the chat model via the
provider-agnostic factory, sends one tiny prompt, and prints the reply. It never
prints your API key. Exits non-zero on any failure so it can be used in CI.
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv

from cv_align_agents.llm.client import LLMConfigError, get_chat_model
from cv_align_agents.settings import get_settings


def main() -> int:
    # Ensure .env is loaded even if the process didn't auto-load it.
    load_dotenv()

    settings = get_settings()
    print(f"Provider : {settings.llm_provider}")
    print(f"Model    : {settings.llm_model or '(provider default)'}")

    try:
        model = get_chat_model()
    except LLMConfigError as exc:
        print(f"\n[CONFIG ERROR] {exc}")
        return 2

    prompt = "Reply with exactly the word: pong"
    print(f"\nSending test prompt: {prompt!r}")

    try:
        response = model.invoke(prompt)
    except Exception as exc:  # noqa: BLE001 - surface any provider/network error
        print(f"\n[REQUEST FAILED] {type(exc).__name__}: {exc}")
        print("Check that your API key is valid and you have network access.")
        return 1

    text = getattr(response, "content", str(response))
    print(f"\nLLM replied: {text!r}")
    print("\nOK - LLM client is working end to end.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
