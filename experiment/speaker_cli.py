"""Container-side OpenAI-compatible participant speaker calls for DynaComSec trials."""

from __future__ import annotations

import json
import os
import sys

from dynacomsec.experiment.adaptive import generate_attacker_decision, generate_speaker_message


def main() -> None:
    payload = json.load(sys.stdin)
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise SystemExit("speaker request requires a messages list")

    model = (
        os.environ.get("DYNACOMSEC_SPEAKER_MODEL")
        or os.environ.get("DYNACOMSEC_NORMAL_MODEL")
        or os.environ.get("DYNACOMSEC_ATTACKER_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or "gpt-4o"
    )
    api_key = (
        os.environ.get("DYNACOMSEC_SPEAKER_API_KEY")
        or os.environ.get("DYNACOMSEC_NORMAL_API_KEY")
        or os.environ.get("DYNACOMSEC_ATTACKER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    )
    base_url = (
        os.environ.get("DYNACOMSEC_SPEAKER_BASE_URL")
        or os.environ.get("DYNACOMSEC_NORMAL_BASE_URL")
        or os.environ.get("DYNACOMSEC_ATTACKER_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
    )
    if not api_key:
        raise SystemExit("DYNACOMSEC_SPEAKER_API_KEY or OPENAI_API_KEY is required")

    mode = str(payload.get("mode") or "decision")
    if mode == "message":
        message = generate_speaker_message(
            messages,
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=float(payload.get("temperature", 0.8)),
        )
        if not message:
            raise SystemExit("speaker model returned an empty message")
        print(json.dumps({"message": message}, ensure_ascii=False))
        return
    if mode != "decision":
        raise SystemExit(f"unsupported speaker request mode: {mode}")

    decision = generate_attacker_decision(
        messages,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
    print(json.dumps(decision, ensure_ascii=False))


if __name__ == "__main__":
    main()
