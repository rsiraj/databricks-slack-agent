from pathlib import Path


PROMPTS = Path(__file__).resolve().parents[1] / "prompts" / "core"


def test_shared_contract_routes_workflows_without_duplicate_body_templates():
    contract = (PROMPTS / "99_response_templates.md").read_text()
    normalized = " ".join(contract.split())

    assert "sole authority for the answer body's headings and order" in normalized
    assert "append one short *Checked* section" in normalized
    assert "Choose one primary workflow" in normalized
    assert "*Most likely cause*" not in contract
    assert "*Query A summary*" not in contract
    assert "*Metric definition*" not in contract


def test_every_workflow_is_named_by_the_shared_contract():
    contract = (PROMPTS / "99_response_templates.md").read_text()
    workflow_names = sorted(path.name for path in (PROMPTS / "workflows").glob("*.md"))

    assert workflow_names
    for workflow_name in workflow_names:
        assert f"core/workflows/{workflow_name}" in contract


def test_job_workflow_defines_one_shape_for_each_interaction_stage():
    job_prompt = (PROMPTS / "workflows" / "job-trigger.md").read_text()

    for stage in (
        "Before confirmation:",
        "After triggering:",
        "For a status follow-up:",
        "For cancellation:",
    ):
        assert stage in job_prompt
