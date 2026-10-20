"""Small JSON helpers used by DynaComSec.

This module keeps the benchmark independent from the surrounding Prism project.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def coerce_json(text: str) -> dict[str, Any]:
    """Extract a JSON object from an LLM response.

    Handles common model wrappers such as markdown fences, surrounding prose, and
    trailing commas before closing braces/brackets.
    """

    if not text:
        raise ValueError("empty JSON text")

    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text, flags=re.MULTILINE)

    start = text.find("{")
    if start == -1:
        raise json.JSONDecodeError("No JSON object found", text, 0)

    depth = 0
    in_string = False
    escape_next = False
    end = -1
    for index in range(start, len(text)):
        char = text[index]
        if escape_next:
            escape_next = False
            continue
        if char == "\\" and in_string:
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = index
                break

    if end == -1:
        raise json.JSONDecodeError("No complete JSON object found", text, 0)

    text = re.sub(r",\s*(\}|\])", r"\1", text[start : end + 1])
    return json.loads(text)


def write_json(path: str | Path, obj: Any, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=indent, ensure_ascii=False), encoding="utf-8")
