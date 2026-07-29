"""Parsing of ```tool_call``` blocks emitted by the model."""

import json
import re

TOOL_CALL_RE = re.compile(r"```tool_call\s*(\{.*?\})\s*```", re.DOTALL)


def extract_tool_call(text: str):
    match = TOOL_CALL_RE.search(text)
    if not match:
        return None
    try:
        data = json.loads(match.group(1))
        return data.get("tool"), data.get("arguments", {})
    except json.JSONDecodeError:
        return None
