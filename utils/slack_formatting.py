"""Normalize model-generated Markdown into Slack-compatible mrkdwn."""

from __future__ import annotations

import re
import unicodedata


SLACK_MESSAGE_MAX_CHARS = 3_500
_CHUNK_LABEL_RESERVE = 32
# When a split lands inside a code block the next part reopens the fence. The
# reopened header is capped to "```" plus a short language token, so the space
# reserved for repair is derived from that cap (reopen header + newline, and
# the closing "\n```") rather than from whatever the model wrote after the
# backticks — a long fence line such as "```sql -- some comment ..." can never
# push a chunk over the limit.
_MAX_FENCE_LANGUAGE_CHARS = 20
_FENCE_MARKER = "```"
_FENCE_REPAIR_RESERVE = (
    len(_FENCE_MARKER) + _MAX_FENCE_LANGUAGE_CHARS + len("\n")  # reopen header
    + len("\n") + len(_FENCE_MARKER)  # closing fence
)
_FENCE_LANGUAGE_TOKEN = re.compile(r"^[A-Za-z0-9_+#.-]+")
_PIPE_SEPARATOR_CELL = re.compile(r"^:?-{3,}:?$")
_INLINE_CODE = re.compile(r"(`[^`\n]*`)")
_MARKDOWN_LINK = re.compile(
    r"!?\[([^\]]+)\]\(\s*<?(https?://[^)\s>|]+)(?:\|[^)>]*)?>?\s*\)"
)


def _parse_pipe_row(line: str) -> list[str] | None:
    stripped = line.strip()
    if "|" not in stripped:
        return None
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|") and not stripped.endswith(r"\|"):
        stripped = stripped[:-1]
    cells = [cell.strip().replace(r"\|", "|") for cell in re.split(r"(?<!\\)\|", stripped)]
    return cells if len(cells) >= 2 else None


def _is_pipe_separator(cells: list[str] | None, column_count: int) -> bool:
    return bool(
        cells
        and len(cells) == column_count
        and all(_PIPE_SEPARATOR_CELL.fullmatch(cell) for cell in cells)
    )


def _plain_table_cell(cell: str) -> str:
    cell = _MARKDOWN_LINK.sub(lambda match: f"{match.group(1)} ({match.group(2)})", cell)
    cell = re.sub(r"`([^`]+)`", r"\1", cell)
    cell = re.sub(r"\*\*(.+?)\*\*", r"\1", cell)
    cell = re.sub(r"__(.+?)__", r"\1", cell)
    return cell


def _display_width(text: str) -> int:
    """Approximate monospace display width for Unicode Slack code blocks."""
    width = 0
    for character in text:
        if character in {"\u200d", "\ufe0f"} or unicodedata.combining(character):
            continue
        width += 2 if unicodedata.east_asian_width(character) in {"F", "W"} else 1
    return width


def _render_pipe_table(rows: list[list[str]]) -> list[str]:
    plain_rows = [[_plain_table_cell(cell) for cell in row] for row in rows]
    widths = [
        max(_display_width(row[column]) for row in plain_rows)
        for column in range(len(plain_rows[0]))
    ]

    def render_row(row: list[str]) -> str:
        return "  ".join(
            cell + " " * (widths[column] - _display_width(cell))
            for column, cell in enumerate(row)
        ).rstrip()

    separator = "  ".join("-" * width for width in widths)
    return ["```", render_row(plain_rows[0]), separator, *map(render_row, plain_rows[1:]), "```"]


def _format_inline_markdown(text: str) -> str:
    parts = _INLINE_CODE.split(text)
    for index in range(0, len(parts), 2):
        part = parts[index]
        part = _MARKDOWN_LINK.sub(
            lambda match: f"<{match.group(2)}|{match.group(1)}>",
            part,
        )
        part = re.sub(r"\*\*(.+?)\*\*", r"*\1*", part)
        part = re.sub(r"__(.+?)__", r"*\1*", part)
        parts[index] = part
    return "".join(parts)


def _format_non_code_line(line: str) -> str:
    if re.fullmatch(r"\s*(?:-{3,}|\*{3,}|_{3,})\s*", line):
        return ""

    heading = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
    if heading:
        content = _format_inline_markdown(heading.group(1))
        if content.startswith("*") and content.endswith("*") and len(content) > 2:
            content = content[1:-1]
        return f"*{content}*"

    bullet = re.match(r"^(\s*)\*\s+(.+)$", line)
    if bullet:
        line = f"{bullet.group(1)}- {bullet.group(2)}"

    return _format_inline_markdown(line)


def normalize_slack_mrkdwn(text: str) -> str:
    """Convert common standard-Markdown constructs that Slack renders literally.

    Fenced code blocks are preserved byte-for-byte. Markdown pipe tables outside
    code fences become aligned monospace blocks so their columns remain readable.
    """
    lines = text.splitlines()
    rendered: list[str] = []
    index = 0
    in_code_fence = False

    while index < len(lines):
        line = lines[index]
        if line.lstrip().startswith("```"):
            in_code_fence = not in_code_fence
            rendered.append(line)
            index += 1
            continue

        if in_code_fence:
            rendered.append(line)
            index += 1
            continue

        header = _parse_pipe_row(line)
        separator = _parse_pipe_row(lines[index + 1]) if index + 1 < len(lines) else None
        if header and _is_pipe_separator(separator, len(header)):
            rows = [header]
            index += 2
            while index < len(lines):
                row = _parse_pipe_row(lines[index])
                if not row or len(row) != len(header):
                    break
                rows.append(row)
                index += 1
            rendered.extend(_render_pipe_table(rows))
            continue

        rendered.append(_format_non_code_line(line))
        index += 1

    if in_code_fence:
        rendered.append("```")

    return "\n".join(rendered)


def _split_at_readable_boundary(text: str, limit: int) -> tuple[str, str]:
    """Split text within ``limit``, preferring paragraph, line, then word breaks."""
    if len(text) <= limit:
        return text, ""

    window = text[: limit + 1]
    candidates = (
        window.rfind("\n\n"),
        window.rfind("\n"),
        window.rfind(" "),
    )
    minimum_readable_chunk = limit // 2
    boundary = next(
        (value for value in candidates if value >= minimum_readable_chunk),
        max((*candidates, limit)),
    )
    if window[boundary : boundary + 2] == "\n\n":
        consumed = boundary + 2
    elif boundary < len(window) and window[boundary] in {"\n", " "}:
        consumed = boundary + 1
    else:
        consumed = boundary

    head = text[:boundary].rstrip()
    tail = text[consumed:].lstrip("\n")
    return head, tail


def _raw_chunks(text: str, limit: int) -> list[str]:
    chunks: list[str] = []
    remaining = text.strip()
    while remaining:
        chunk, remaining = _split_at_readable_boundary(remaining, limit)
        if not chunk:
            chunk, remaining = remaining[:limit], remaining[limit:]
        chunks.append(chunk)
    return chunks


def _fence_reopen_header(fence_line: str) -> str:
    """Return the fence header used to reopen a split code block.

    Only the language token survives (``"```sql -- comment"`` reopens as
    ``"```sql"``), and an implausibly long token is dropped so the header never
    exceeds the reserve accounted for in ``_FENCE_REPAIR_RESERVE``.
    """
    remainder = fence_line.lstrip()[len(_FENCE_MARKER):].lstrip()
    token = _FENCE_LANGUAGE_TOKEN.match(remainder)
    if token and len(token.group(0)) <= _MAX_FENCE_LANGUAGE_CHARS:
        return f"{_FENCE_MARKER}{token.group(0)}"
    return _FENCE_MARKER


def _repair_split_fences(chunks: list[str]) -> list[str]:
    """Balance code fences in each independently rendered Slack message."""
    repaired: list[str] = []
    open_fence: str | None = None

    for chunk in chunks:
        prefix = f"{_fence_reopen_header(open_fence)}\n" if open_fence else ""
        state = open_fence
        for line in chunk.splitlines():
            stripped = line.lstrip()
            if not stripped.startswith("```"):
                continue
            if state is None:
                state = stripped
            else:
                state = None

        suffix = "\n```" if state is not None else ""
        repaired.append(f"{prefix}{chunk}{suffix}")
        open_fence = state

    return repaired


def split_slack_message(
    text: str,
    *,
    max_chars: int = SLACK_MESSAGE_MAX_CHARS,
) -> list[str]:
    """Split a response into labeled, independently valid Slack messages.

    Paragraph and line boundaries are preferred. When a split lands inside a
    fenced code block, the current part is closed and the next part reopens the
    same fence so every Slack message renders correctly on its own.
    """
    if not text:
        return []
    if max_chars <= _CHUNK_LABEL_RESERVE + _FENCE_REPAIR_RESERVE:
        raise ValueError("max_chars is too small for Slack chunk metadata")
    if len(text) <= max_chars:
        return [text]

    payload_limit = max_chars - _CHUNK_LABEL_RESERVE - _FENCE_REPAIR_RESERVE
    chunks = _repair_split_fences(_raw_chunks(text, payload_limit))
    total = len(chunks)
    labeled = [
        f"_Part {index} of {total}_\n\n{chunk}"
        for index, chunk in enumerate(chunks, 1)
    ]
    if any(len(chunk) > max_chars for chunk in labeled):
        raise ValueError("Slack chunk exceeded configured maximum")
    return labeled
