"""Prompt layout: core + business stay provider-neutral; provider files append last."""

from pathlib import Path

import pytest

from utils import system_prompt
from utils.system_prompt import (
    PROMPTS_DIR,
    SECTION_SEPARATOR,
    load_base_prompt,
    load_provider_prompt,
    load_system_prompt,
)


def _legacy_concatenation() -> str:
    """The exact algorithm the deployed baseline used before provider files existed."""
    parts = []
    core_dir = PROMPTS_DIR / "core"
    parts.extend(path.read_text() for path in sorted(core_dir.glob("*.md")))
    parts.extend(path.read_text() for path in sorted((core_dir / "workflows").glob("*.md")))
    parts.extend(path.read_text() for path in sorted((PROMPTS_DIR / "business").glob("*.md")))
    return "\n\n---\n\n".join(parts)


def test_anthropic_prompt_is_byte_identical_to_core_plus_business():
    expected = _legacy_concatenation()
    assert load_system_prompt("anthropic") == expected
    assert load_system_prompt() == expected  # default provider
    assert load_base_prompt() == expected


def test_anthropic_provider_file_is_comment_only():
    text = (PROMPTS_DIR / "providers" / "anthropic.md").read_text()
    assert text.strip().startswith("<!--") and text.strip().endswith("-->")
    assert load_provider_prompt("anthropic") == ""


def test_openai_prompt_appends_provider_notes_last():
    base = load_base_prompt()
    prompt = load_system_prompt("openai")
    addition = load_provider_prompt("openai")

    assert addition
    assert "<!--" not in addition
    assert prompt == base + SECTION_SEPARATOR + addition
    assert prompt.startswith(base)


def test_core_and_business_files_do_not_reference_a_provider():
    provider_words = ("cache_control", "Responses API", "function_call_output")
    for folder in ("core", "business"):
        for path in sorted((PROMPTS_DIR / folder).rglob("*.md")):
            text = path.read_text()
            for word in provider_words:
                assert word not in text, f"{path} mentions provider-specific term {word!r}"


def test_unknown_provider_without_file_returns_base_prompt():
    assert load_system_prompt("nonexistent-provider") == load_base_prompt()


@pytest.mark.parametrize("bad", ["../core", "Anthropic", "open ai", ""])
def test_provider_name_is_validated(bad):
    with pytest.raises(ValueError):
        load_provider_prompt(bad)


def test_comment_only_and_whitespace_provider_files_append_nothing(tmp_path, monkeypatch):
    prompts = tmp_path / "prompts"
    (prompts / "core").mkdir(parents=True)
    (prompts / "providers").mkdir()
    (prompts / "core" / "00.md").write_text("CORE")
    (prompts / "providers" / "blank.md").write_text("\n  \n")
    (prompts / "providers" / "commented.md").write_text(
        "<!-- one -->\n\n<!-- two -->\n"
    )
    (prompts / "providers" / "real.md").write_text("<!-- note -->\n# Real\nline\n")
    monkeypatch.setattr(system_prompt, "PROMPTS_DIR", prompts)
    for fn in (load_base_prompt, load_provider_prompt, load_system_prompt):
        fn.cache_clear()
    try:
        assert load_system_prompt("blank") == "CORE"
        assert load_system_prompt("commented") == "CORE"
        assert load_system_prompt("real") == "CORE" + SECTION_SEPARATOR + "# Real\nline"
    finally:
        for fn in (load_base_prompt, load_provider_prompt, load_system_prompt):
            fn.cache_clear()
