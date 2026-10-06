"""Load the system prompt by concatenating every .md file under prompts/.

Layout:
    prompts/
      core/                    - generic, ships with the bot
        00_behavior.md         - tool/SQL rules, communication style
        workflows/*.md         - one file per workflow procedure
      business/                - domain-specific; replace these to repurpose
        00_repo_context.md     - catalog map, history tables, warehouse IDs
        01_glossary.md         - team terminology snapshot
        02_*.md                - additional business knowledge files
      providers/<name>.md      - provider-specific style notes, appended LAST

Files load in this order: core/*.md, core/workflows/*.md, business/*.md, each
group sorted alphabetically, then the provider file. Order matters because
Anthropic's prompt cache is prefix-based — core comes first because it's the
most stable; the business folder churns more as the team adds new knowledge
files; the provider file is last so core and business stay provider-neutral.

A provider file that is empty, or contains only ``<!-- ... -->`` comment
lines, appends nothing — ``prompts/providers/anthropic.md`` ships that way so
the deployed Anthropic prompt is byte-identical to the pre-provider layout.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
DEFAULT_PROVIDER = "anthropic"
SECTION_SEPARATOR = "\n\n---\n\n"

_PROVIDER_NAME_RE = re.compile(r"^[a-z0-9_-]+$")
_COMMENT_LINE_RE = re.compile(r"^\s*<!--.*?-->\s*$")


def _strip_comment_lines(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not _COMMENT_LINE_RE.match(line))


@lru_cache(maxsize=None)
def load_base_prompt() -> str:
    """Provider-neutral prompt: core + business, in load order."""
    parts: list[str] = []

    core_dir = PROMPTS_DIR / "core"
    if core_dir.is_dir():
        for path in sorted(core_dir.glob("*.md")):
            parts.append(path.read_text())
        workflows_dir = core_dir / "workflows"
        if workflows_dir.is_dir():
            for path in sorted(workflows_dir.glob("*.md")):
                parts.append(path.read_text())

    business_dir = PROMPTS_DIR / "business"
    if business_dir.is_dir():
        for path in sorted(business_dir.glob("*.md")):
            parts.append(path.read_text())

    if not parts:
        raise RuntimeError(f"No prompt files found under {PROMPTS_DIR}")

    return SECTION_SEPARATOR.join(parts)


@lru_cache(maxsize=None)
def load_provider_prompt(provider: str) -> str:
    """Contents of ``prompts/providers/<provider>.md`` minus comment lines ("" if none)."""
    if not _PROVIDER_NAME_RE.match(provider):
        raise ValueError(f"Invalid provider name {provider!r}")
    path = PROMPTS_DIR / "providers" / f"{provider}.md"
    if not path.is_file():
        return ""
    return _strip_comment_lines(path.read_text()).strip()


@lru_cache(maxsize=None)
def load_system_prompt(provider: str = DEFAULT_PROVIDER) -> str:
    base = load_base_prompt()
    addition = load_provider_prompt(provider)
    if not addition:
        return base
    return base + SECTION_SEPARATOR + addition
