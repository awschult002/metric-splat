"""Minimal YAML subset loader (mappings, lists, scalars, comments). No anchors."""
from __future__ import annotations

import re
from typing import Any


def load(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except ImportError:
        return _parse(text)


def _parse(text: str) -> Any:
    lines = []
    for raw in text.splitlines():
        # strip full-line and inline comments (not inside quotes)
        if "#" in raw:
            in_q = False
            out = []
            for ch in raw:
                if ch in "\"'":
                    in_q = not in_q
                if ch == "#" and not in_q:
                    break
                out.append(ch)
            raw = "".join(out)
        if raw.strip() == "":
            continue
        lines.append(raw.rstrip())
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    i = 0
    while i < len(lines):
        line = lines[i]
        indent = len(line) - len(line.lstrip(" "))
        content = line.lstrip(" ")
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if content.startswith("- "):
            item_raw = content[2:].strip()
            if not isinstance(parent, list):
                raise ValueError(f"list item without list parent: {line}")
            if ":" in item_raw and not item_raw.startswith(("'", '"', "[")):
                # inline mapping as list item — rare; treat as scalar string
                parent.append(_scalar(item_raw))
            else:
                parent.append(_scalar(item_raw))
            i += 1
            continue

        if ":" not in content:
            raise ValueError(f"expected key: {line}")
        key, _, rest = content.partition(":")
        key = key.strip()
        rest = rest.strip()
        if not isinstance(parent, dict):
            raise ValueError(f"mapping key under non-dict: {line}")

        if rest == "":
            # peek next line to decide list vs dict
            nxt = lines[i + 1] if i + 1 < len(lines) else None
            if nxt is not None:
                nindent = len(nxt) - len(nxt.lstrip(" "))
                if nindent > indent and nxt.lstrip(" ").startswith("- "):
                    child: Any = []
                else:
                    child = {}
            else:
                child = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _scalar(rest)
        i += 1
    return root


def _scalar(s: str) -> Any:
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        return [_scalar(p.strip()) for p in inner.split(",")]
    low = s.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("null", "~"):
        return None
    # int
    if re.fullmatch(r"-?\d+", s):
        return int(s)
    # float
    if re.fullmatch(r"-?\d+\.\d+", s):
        return float(s)
    return s
