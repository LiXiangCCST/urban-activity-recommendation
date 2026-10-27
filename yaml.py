"""Small YAML fallback used when PyYAML is unavailable.

This intentionally supports the simple mappings/lists/scalars used by this project.
It is not a general YAML implementation.
"""

from __future__ import annotations

import ast
from typing import Any


def _strip_comment(line: str) -> str:
    in_quote = False
    quote = ""
    for i, char in enumerate(line):
        if char in {'"', "'"} and (i == 0 or line[i - 1] != "\\"):
            if not in_quote:
                in_quote = True
                quote = char
            elif quote == char:
                in_quote = False
        if char == "#" and not in_quote:
            return line[:i]
    return line


def _value(text: str) -> Any:
    text = text.strip()
    if text == "":
        return None
    if text in {"null", "Null", "NULL", "~"}:
        return None
    if text in {"true", "True", "TRUE"}:
        return True
    if text in {"false", "False", "FALSE"}:
        return False
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return ast.literal_eval(text)
    if text.startswith("[") and text.endswith("]"):
        return [_value(item) for item in _split_inline_list(text[1:-1])]
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text



def _split_inline_list(text: str) -> list[str]:
    items = []
    current = []
    in_quote = False
    quote = ""
    bracket_depth = 0
    for i, char in enumerate(text):
        if char in {"\"", "'"} and (i == 0 or text[i - 1] != "\\"):
            if not in_quote:
                in_quote = True
                quote = char
            elif quote == char:
                in_quote = False
        elif char == "[" and not in_quote:
            bracket_depth += 1
        elif char == "]" and not in_quote and bracket_depth > 0:
            bracket_depth -= 1
        if char == "," and not in_quote and bracket_depth == 0:
            item = "".join(current).strip()
            if item:
                items.append(item)
            current = []
        else:
            current.append(char)
    item = "".join(current).strip()
    if item:
        items.append(item)
    return items

def _split_key_value(text: str) -> tuple[str, str | None]:
    if ":" not in text:
        return text.strip(), None
    key, rest = text.split(":", 1)
    return key.strip(), rest.strip()


def _prepare(stream) -> list[tuple[int, str]]:
    content = stream.read() if hasattr(stream, "read") else str(stream)
    rows = []
    for raw in content.splitlines():
        line = _strip_comment(raw).rstrip()
        if not line.strip():
            continue
        rows.append((len(line) - len(line.lstrip(" ")), line.strip()))
    return rows


def _parse_block(rows: list[tuple[int, str]], i: int, indent: int) -> tuple[Any, int]:
    if i >= len(rows):
        return {}, i
    is_list = rows[i][1].startswith("- ")
    if is_list:
        items = []
        while i < len(rows) and rows[i][0] == indent and rows[i][1].startswith("- "):
            text = rows[i][1][2:].strip()
            i += 1
            if not text:
                child, i = _parse_block(rows, i, rows[i][0]) if i < len(rows) and rows[i][0] > indent else ({}, i)
                items.append(child)
                continue
            key, rest = _split_key_value(text)
            if rest is None:
                items.append(_value(text))
                continue
            item = {key: _value(rest) if rest else None}
            if not rest and i < len(rows) and rows[i][0] > indent:
                child, i = _parse_block(rows, i, rows[i][0])
                item[key] = child
            while i < len(rows) and rows[i][0] > indent:
                child, i = _parse_block(rows, i, rows[i][0])
                if isinstance(child, dict):
                    item.update(child)
                else:
                    break
            items.append(item)
        return items, i

    data = {}
    while i < len(rows) and rows[i][0] == indent and not rows[i][1].startswith("- "):
        key, rest = _split_key_value(rows[i][1])
        i += 1
        if rest:
            data[key] = _value(rest)
        else:
            if i < len(rows) and rows[i][0] > indent:
                child, i = _parse_block(rows, i, rows[i][0])
                data[key] = child
            else:
                data[key] = None
    return data, i


def safe_load(stream) -> Any:
    rows = _prepare(stream)
    if not rows:
        return None
    data, _ = _parse_block(rows, 0, rows[0][0])
    return data


def _dump_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if any(ch in text for ch in [":", "#", "\n"]) or text == "":
        return repr(text)
    return text


def _dump(obj: Any, indent: int = 0) -> list[str]:
    pad = " " * indent
    lines = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{pad}{key}:")
                lines.extend(_dump(value, indent + 2))
            else:
                lines.append(f"{pad}{key}: {_dump_value(value)}")
    elif isinstance(obj, list):
        for value in obj:
            if isinstance(value, dict):
                lines.append(f"{pad}-")
                lines.extend(_dump(value, indent + 2))
            else:
                lines.append(f"{pad}- {_dump_value(value)}")
    else:
        lines.append(f"{pad}{_dump_value(obj)}")
    return lines


def safe_dump(obj: Any, stream=None, sort_keys: bool = False, **_: Any) -> str | None:
    if sort_keys and isinstance(obj, dict):
        obj = dict(sorted(obj.items()))
    text = "\n".join(_dump(obj)) + "\n"
    if stream is not None:
        stream.write(text)
        return None
    return text
