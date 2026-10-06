"""Regression checks for focused table-explanation evidence gathering."""

from utils.system_prompt import load_system_prompt


def test_explicit_lineage_is_prioritized_before_optional_table_profiling():
    prompt = load_system_prompt()

    assert "When the user explicitly asks for lineage" in prompt
    assert "Describe `system.access.table_lineage` first" in prompt
    assert "before optional profiling" in prompt


def test_table_explanations_do_not_require_unrequested_job_discovery():
    prompt = load_system_prompt()

    assert "Do not spend tool calls locating job configuration" in prompt
    assert "Only inspect a refresh job, schedule, or runtime" in prompt
