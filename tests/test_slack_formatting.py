from utils.slack_formatting import (
    _display_width,
    normalize_slack_mrkdwn,
    split_slack_message,
)


def test_markdown_pipe_table_becomes_aligned_code_block():
    source = """Now I have exact durations:

| Job | Duration (ms) | Duration |
|---|---:|---|
| Ops Email | 3,991,371 | 66m 31s |
| PartnerB Email | 2,752,500 | 45m 53s |

*Answer*
"""

    result = normalize_slack_mrkdwn(source)

    assert "|---" not in result
    assert """```
Job             Duration (ms)  Duration
--------------  -------------  --------
Ops Email       3,991,371      66m 31s
PartnerB Email  2,752,500      45m 53s
```""" in result
    assert result.endswith("*Answer*")


def test_standard_markdown_constructs_become_slack_mrkdwn():
    source = """## Results

**All runs succeeded.**
* First item
[Open the run](https://example.test/run/123)
---
"""

    assert normalize_slack_mrkdwn(source) == """*Results*

*All runs succeeded.*
- First item
<https://example.test/run/123|Open the run>
"""


def test_slack_wrapped_markdown_link_becomes_one_clean_slack_link():
    source = "[Run link](<https://example.com|example.com>)"

    assert normalize_slack_mrkdwn(source) == "<https://example.com|Run link>"


def test_code_fences_and_inline_code_are_preserved():
    source = """**Outside**
`**inline stays unchanged**`
```sql
SELECT '**inside**' AS value;
| Job | Duration |
|---|---|
```
"""

    assert normalize_slack_mrkdwn(source) == """*Outside*
`**inline stays unchanged**`
```sql
SELECT '**inside**' AS value;
| Job | Duration |
|---|---|
```"""


def test_table_cell_markdown_is_rendered_as_plain_monospace_text():
    source = """| Job | Reference |
|---|---|
| **Coverage** | [run](https://example.test/run/123) |
| `Access` | complete |
"""

    result = normalize_slack_mrkdwn(source)

    assert "Coverage  run (https://example.test/run/123)" in result
    assert "Access    complete" in result
    assert "**Coverage**" not in result


def test_long_response_splits_at_readable_boundaries_with_part_labels():
    source = "\n\n".join(f"Paragraph {index}: " + "x" * 80 for index in range(12))

    chunks = split_slack_message(source, max_chars=350)

    assert len(chunks) > 1
    assert all(len(chunk) <= 350 for chunk in chunks)
    assert chunks[0].startswith(f"_Part 1 of {len(chunks)}_\n\n")
    assert chunks[-1].startswith(f"_Part {len(chunks)} of {len(chunks)}_\n\n")
    assert "Paragraph 0:" in chunks[0]


def test_long_code_block_is_closed_and_reopened_between_parts():
    source = "```sql\n" + "\n".join(f"SELECT {index};" for index in range(80)) + "\n```"

    chunks = split_slack_message(source, max_chars=300)

    assert len(chunks) > 1
    assert all(chunk.count("```") % 2 == 0 for chunk in chunks)
    assert all(len(chunk) <= 300 for chunk in chunks)
    assert chunks[1].splitlines()[2] == "```sql"


def test_long_fence_line_never_overflows_chunk_limit():
    fence_line = "```sql -- " + "x" * 110
    assert len(fence_line) == 120
    source = fence_line + "\n" + "\n".join(f"SELECT {index};" for index in range(120)) + "\n```"
    assert len(source) > 300

    chunks = split_slack_message(source, max_chars=300)

    assert len(chunks) > 1
    assert all(len(chunk) <= 300 for chunk in chunks)
    assert all(chunk.count("```") % 2 == 0 for chunk in chunks)
    # The reopened fence carries only the language token, not the long comment.
    assert all(chunk.splitlines()[2] == "```sql" for chunk in chunks[1:])
    # The original long fence line is preserved verbatim in the first part.
    assert fence_line in chunks[0]


def test_fence_reopen_header_keeps_only_language_token():
    from utils.slack_formatting import _FENCE_REPAIR_RESERVE, _fence_reopen_header

    assert _fence_reopen_header("```sql") == "```sql"
    assert _fence_reopen_header("```sql -- a long trailing comment") == "```sql"
    assert _fence_reopen_header("```") == "```"
    assert _fence_reopen_header("   ```python  # indented") == "```python"
    assert _fence_reopen_header("```" + "z" * 200) == "```"
    for header in ("```sql", "```" + "z" * 200, "```sql -- " + "x" * 300):
        assert len(_fence_reopen_header(header)) + len("\n") + len("\n```") <= _FENCE_REPAIR_RESERVE


def test_unclosed_code_fence_is_closed_without_discarding_content():
    source = "```sql\nSELECT COUNT(*) FROM users;\n\nResult was returned."

    result = normalize_slack_mrkdwn(source)

    assert result == source + "\n```"
    assert result.count("```") == 2


def test_unicode_table_columns_align_by_display_width():
    source = """| Market | Status |
|---|---|
| 日本 | ✅ Active |
| Boston | Ready |
"""

    result = normalize_slack_mrkdwn(source)
    rows = result.splitlines()
    japan_prefix = rows[3].split("✅", 1)[0]
    boston_prefix = rows[4].split("Ready", 1)[0]

    assert _display_width("日本") == 4
    assert _display_width("✅") == 2
    assert _display_width(japan_prefix) == _display_width(boston_prefix)
