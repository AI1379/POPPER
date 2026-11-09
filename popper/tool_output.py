"""Lossless local output archives and concise, diagnostic-preserving observations."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import re
from typing import Any


WARNING_HEADER = re.compile(r"^.+?:\d+: (?P<category>\w*Warning): .*$", re.MULTILINE)
IMPORTANT = re.compile(
    r"(?i)(\bp[ _-]?(?:value|val)\b|p\s*[<>=]|P>\||statistic|confidence|"
    r"\bCI\b|coefficient|estimate|standard.error|n_train|n_test|sample.size|"
    r"convergen|iteration|warning|traceback|error|exception|\bnan\b|\binf\b|"
    r"\bB\s*=|T_obs|T_null|observed T|log.loss|rank|degrees.of.freedom)"
)


@dataclass(frozen=True)
class WarningSpan:
    start: int
    end: int
    category: str
    message: str
    location: str


def warning_spans_from_text(text: str) -> list[WarningSpan]:
    """Fallback for replaying historical, default-formatted warning output."""
    spans = []
    for match in WARNING_HEADER.finditer(text):
        end = match.end()
        if end < len(text) and text[end] == "\n":
            end += 1
        # Default warnings formatting adds the indented source line.
        if text[end:].startswith("  "):
            newline = text.find("\n", end)
            end = len(text) if newline == -1 else newline + 1
        spans.append(WarningSpan(match.start(), end, match.group("category"),
                                 match.group(0), match.group(0).split(": " + match.group("category"), 1)[0]))
    return spans


def deduplicate_warnings(text: str, spans: list[WarningSpan] | None = None) -> tuple[str, list[dict[str, Any]]]:
    spans = warning_spans_from_text(text) if spans is None else spans
    counts = Counter((span.category, span.message, span.location) for span in spans)
    seen = set()
    chunks = []
    diagnostics = []
    cursor = 0
    for span in sorted(spans, key=lambda item: item.start):
        if not 0 <= cursor <= span.start <= span.end <= len(text):
            raise ValueError("Invalid warning span")
        chunks.append(text[cursor:span.start])
        key = (span.category, span.message, span.location)
        if key not in seen:
            count = counts[key]
            chunks.append(text[span.start:span.end])
            if count > 1:
                chunks.append(f"[Same warning occurred {count} times in this tool call; duplicates omitted.]\n")
            diagnostics.append({"category": span.category, "message": span.message,
                                "location": span.location, "count": count})
            seen.add(key)
        cursor = span.end
    chunks.append(text[cursor:])
    return "".join(chunks), diagnostics


def bound_observation(text: str, max_chars: int) -> tuple[str, bool]:
    """Head/tail view plus verbatim diagnostic/result lines; never silent clipping.

    The budget is soft when important lines exceed it. Full data stays available
    through the local archive reader; no LLM-generated summarization is used.
    """
    if len(text) <= max_chars:
        return text, False
    lines = text.splitlines(keepends=True)
    # Bound oversized single-line dumps too; line-based head/tail selection alone
    # would otherwise retain the entire first line.
    for index, line in enumerate(lines):
        if len(line) > max_chars and not IMPORTANT.search(line):
            head, tail = max_chars // 3, max_chars // 3
            lines[index] = (line[:head] + f"\n[... {len(line) - head - tail} characters omitted from this line; full archive available ...]\n"
                            + line[-tail:])
    offsets = []
    position = 0
    for line in lines:
        offsets.append(position)
        position += len(line)
    head_budget, tail_budget = max_chars // 3, max_chars // 3
    keep = {index for index, (offset, line) in enumerate(zip(offsets, lines))
            if offset < head_budget or offset + len(line) > position - tail_budget
            or IMPORTANT.search(line)}
    # Preserve the surrounding context of important diagnostics/result lines.
    for index in list(keep):
        if IMPORTANT.search(lines[index]):
            keep.update(range(max(0, index - 2), min(len(lines), index + 3)))
    chunks = []
    omitted = 0
    for index, line in enumerate(lines):
        if index not in keep:
            omitted += len(line)
            continue
        if omitted:
            chunks.append(f"\n[... {omitted} characters omitted from this view; available in the full-output archive ...]\n")
            omitted = 0
        chunks.append(line)
    if omitted:
        chunks.append(f"\n[... {omitted} characters omitted from this view; available in the full-output archive ...]\n")
    return "".join(chunks), True


class ToolOutputStore:
    def __init__(self, root: Path, max_chars: int = 12000, compact: bool = True):
        self.root = root.resolve()
        self.max_chars = max_chars
        self.compact = compact
        self.audit: list[dict[str, Any]] = []

    def publish(self, raw: str, *, code: str, spans: list[WarningSpan]) -> str:
        handle = f"tool_{len(self.audit) + 1:04d}"
        self.root.mkdir(parents=True, exist_ok=True)
        raw_path = self.root / f"{handle}.txt.gz"
        with gzip.open(raw_path, "wt", encoding="utf-8") as stream:
            stream.write(raw)
        deduplicated, diagnostics = deduplicate_warnings(raw, spans)
        view, truncated = bound_observation(deduplicated, self.max_chars) if self.compact else (raw, False)
        (self.root / f"{handle}.view.txt").write_text(deduplicated, encoding="utf-8")
        changed = self.compact and (deduplicated != raw or truncated)
        if changed:
            view += (
                f"\n[Full output: {handle}. It is losslessly archived. For omitted details, run "
                f"print(read_tool_output('{handle}', start=0, length=4000, raw=False)). "
                "Use raw=True for the original output including every warning. "
                "Do not assume omitted data or numerical warnings are harmless.]\n"
            )
        entry = {"handle": handle, "original_chars": len(raw), "deduplicated_chars": len(deduplicated),
                 "delivered_chars": len(view), "truncated": truncated, "warnings": diagnostics,
                 "raw_path": str(raw_path), "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                 "code": code}
        self.audit.append(entry)
        (self.root / f"{handle}.json").write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
        return view

    def read(self, handle: str, start: int = 0, length: int = 4000, raw: bool = False) -> str:
        if not re.fullmatch(r"tool_\d{4,}", handle) or not any(e["handle"] == handle for e in self.audit):
            raise ValueError("Unknown tool output handle")
        if start < 0 or not 1 <= length <= 12000:
            raise ValueError("Use start >= 0 and length between 1 and 12000")
        if raw:
            with gzip.open(self.root / f"{handle}.txt.gz", "rt", encoding="utf-8") as stream:
                text = stream.read()
        else:
            text = (self.root / f"{handle}.view.txt").read_text(encoding="utf-8")
        return f"[{handle}: characters {start}:{min(start + length, len(text))} of {len(text)}]\n" + text[start:start + length]
