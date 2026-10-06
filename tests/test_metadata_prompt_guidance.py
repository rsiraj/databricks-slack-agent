"""Regression checks for Unity Catalog metadata guidance in the system prompt."""

from utils.system_prompt import load_system_prompt


def test_system_prompt_requires_live_table_and_column_comments():
    prompt = load_system_prompt()

    assert "Unity Catalog table and column comments" in prompt
    assert "<catalog>.information_schema.tables" in prompt
    assert "<catalog>.information_schema.columns" in prompt
    assert "once per conversation" in prompt


def test_system_prompt_handles_metadata_conflicts_and_wide_tables():
    prompt = load_system_prompt()

    assert "comments as documentation, not as proof of implementation" in prompt
    assert "very wide event tables" in prompt
    assert "tool output is sampled" in prompt


def test_example_business_context_is_loaded_and_marked_as_a_placeholder():
    prompt = load_system_prompt()

    assert "Example business context" in prompt
    assert "Replace these files with your own organization" in prompt
    assert "Example glossary" in prompt


def test_example_business_context_treats_retired_variants_as_historical():
    prompt = load_system_prompt()

    assert "variants for the retired mobile-app experiment" in prompt
    assert "Do not use their logic for current questions" in prompt


def test_example_business_context_keeps_the_history_table_contract():
    prompt = load_system_prompt()

    assert "acme_prod.acme_history.sql_view_backup" in prompt
    assert "acme_prod.acme_history.databricks_workspace_jobs" in prompt
    assert "**NOT** the fully-qualified `catalog.schema.name` path" in prompt
    assert "a row is inserted only when a definition has actually changed" in prompt


def test_example_business_context_distinguishes_cohort_definitions():
    prompt = load_system_prompt()

    assert "not necessarily one real person" in prompt
    assert "Prove joins before scaling them" in prompt
