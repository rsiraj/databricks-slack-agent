"""Path-safety guarantees for CLI-derived values in eval output and baseline paths."""

import builtins
import io
import json

import pytest

from evals import regression
from evals import run as eval_run
from evals.regression import BaselineError

MINIMAL_BASELINE = {
    "schema_version": 2,
    "baselines": {
        "anthropic/claude-sonnet-4-6/high": {
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
            "effort": "high",
            "recorded_at": "2026-09-05T00:00:00+00:00",
            "source_run_urls": ["https://example.test/run/1"],
            "scenarios": {
                "s": {"repeats": 3, "passed": 3, "pass_rate": 1.0, "mean_score": 1.0, "hard_violation_codes": {}}
            },
        }
    },
}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("claude-sonnet-4-6", "claude-sonnet-4-6"),
        ("signups_table_explanation", "signups_table_explanation"),
        ("high", "high"),
        ("../x", "_x"),
        ("..", None),
        ("...", None),
        ("a/b", "a_b"),
        ("a\\b", "a_b"),
        ("a/../../etc/passwd", "a_____etc_passwd"),
        (".hidden", "hidden"),
        ("us.anthropic.claude-3", "us.anthropic.claude-3"),
        ("a..b", "a_b"),
        ("model name/with spaces", "model_name_with_spaces"),
        ("", None),
        ("/", "_"),
        ("./", "_"),
    ],
)
def test_safe_component(raw, expected):
    if expected is None:
        with pytest.raises(ValueError, match="Path component"):
            eval_run._safe_component(raw)
        return
    result = eval_run._safe_component(raw)
    assert result == expected
    assert ".." not in result
    assert "/" not in result and "\\" not in result
    assert not result.startswith(".")


def test_safe_component_rejects_non_strings():
    with pytest.raises(ValueError, match="must be a string"):
        eval_run._safe_component(None)  # type: ignore[arg-type]


def _run(model, effort, scenario, rep=1):
    return {
        "model": model,
        "effort": effort,
        "scenario": scenario,
        "rep": rep,
        "status": "errored",
        "attempts": 1,
        "error_type": "RuntimeError",
        "trace": {"stop_reason": "infrastructure_error"},
        "result": None,
    }


def test_run_output_path_stays_under_output_root(tmp_path):
    root = tmp_path.resolve()
    hostile = _run("../../escape", "../high", "../../../../../../etc/passwd", rep=3)

    path = eval_run.run_output_path(root, hostile)

    assert path.resolve().is_relative_to(root)
    assert path == root / "___escape" / "_high" / "___________etc_passwd" / "rep3.json"


def test_run_output_path_uses_sanitized_dirs_but_keeps_original_names_in_json(tmp_path):
    hostile = _run("../../escape", None, "weird name/../x")

    eval_run.write_outputs(tmp_path, [hostile])

    written = list(tmp_path.rglob("rep1.json"))
    assert len(written) == 1
    assert written[0].resolve().is_relative_to(tmp_path.resolve())
    assert written[0].parent.name == "weird_name___x"
    assert written[0].parent.parent.name == "none"
    assert written[0].parent.parent.parent.name == "___escape"
    payload = json.loads(written[0].read_text())
    assert payload["model"] == "../../escape"
    assert payload["scenario"] == "weird name/../x"
    assert payload["effort"] is None
    assert (tmp_path / "trace.json").is_file()
    assert (tmp_path / "result.json").is_file()
    # Nothing escaped the output directory.
    assert not (tmp_path.parent / "escape").exists()
    assert not (tmp_path.parent / "x").exists()


def test_path_under_refuses_escape(tmp_path):
    root = tmp_path.resolve()
    with pytest.raises(ValueError, match="outside output directory"):
        eval_run._path_under(root, "..", "sibling.json")
    with pytest.raises(ValueError, match="outside output directory"):
        eval_run._path_under(root, "/etc/passwd")
    assert eval_run._path_under(root) == root
    assert eval_run._path_under(root, "a", "b.json") == root / "a" / "b.json"


# --- baseline files (--baseline / --write-baseline / --from-results) -----------


def test_baseline_read_path_requires_existing_json_without_parent_components(tmp_path):
    good = tmp_path / "baseline.json"
    good.write_text(json.dumps(MINIMAL_BASELINE))
    (tmp_path / "sub").mkdir()
    (tmp_path / "dir.json").mkdir()
    (tmp_path / "notes.txt").write_text("{}")

    assert regression.resolve_baseline_read_path(good) == good.resolve()
    assert regression.load_baseline(good) == MINIMAL_BASELINE
    with pytest.raises(BaselineError, match="must not contain '..'"):
        regression.load_baseline(tmp_path / "sub" / ".." / "baseline.json")
    with pytest.raises(BaselineError, match="must name a .json file"):
        regression.load_baseline(tmp_path / "notes.txt")
    with pytest.raises(BaselineError, match="does not exist"):
        regression.load_baseline(tmp_path / "missing.json")
    with pytest.raises(BaselineError, match="does not exist"):
        regression.load_baseline(tmp_path / "dir.json")  # a directory, not a file


def test_baseline_write_path_must_be_json_under_an_allowed_root(tmp_path):
    out = tmp_path / "out"
    out.mkdir()

    assert regression.resolve_baseline_write_path(out / "b.json", allowed_roots=(out,)) == (out / "b.json").resolve()
    # The repository root is always allowed (the committed evals/baseline.json).
    committed = regression.REPO_ROOT / "evals" / "baseline.json"
    assert regression.resolve_baseline_write_path(committed) == committed
    with pytest.raises(ValueError, match="outside the repository or the output directory"):
        regression.resolve_baseline_write_path(tmp_path / "b.json", allowed_roots=(out,))
    with pytest.raises(ValueError, match="must not contain '..'"):
        regression.resolve_baseline_write_path(out / ".." / "b.json", allowed_roots=(out,))
    with pytest.raises(ValueError, match="must not contain '..'"):
        # Would resolve back under ``out`` but still names a parent component.
        regression.resolve_baseline_write_path(out / "x" / ".." / "b.json", allowed_roots=(out,))
    with pytest.raises(ValueError, match="must name a .json file"):
        regression.resolve_baseline_write_path(out / "b.txt", allowed_roots=(out,))
    with pytest.raises(ValueError, match="names a directory"):
        (out / "dir.json").mkdir()
        regression.resolve_baseline_write_path(out / "dir.json", allowed_roots=(out,))
    with pytest.raises(ValueError, match="outside"):
        regression.write_baseline(tmp_path / "b.json", MINIMAL_BASELINE, allowed_roots=(out,))
    assert not (tmp_path / "b.json").exists()


def test_results_dir_must_exist(tmp_path):
    with pytest.raises(ValueError, match="not a directory"):
        regression.load_runs_from_dir(tmp_path / "missing")
    (tmp_path / "file").write_text("x")
    with pytest.raises(ValueError, match="not a directory"):
        regression.load_runs_from_dir(tmp_path / "file")


def test_cli_rejects_unsafe_baseline_paths_before_doing_anything(tmp_path):
    def forbidden_client():  # pragma: no cover - only hit on regression
        raise AssertionError("client built despite an unsafe baseline path")

    with pytest.raises(SystemExit, match="--baseline: --baseline file does not exist"):
        eval_run.main(["--baseline", str(tmp_path / "nope.json"), "--output-dir", str(tmp_path)], client_factory=forbidden_client)
    with pytest.raises(SystemExit, match=r"--baseline: --baseline path must not contain '\.\.'"):
        eval_run.main(["--baseline", str(tmp_path / "x" / ".." / "b.json"), "--output-dir", str(tmp_path)], client_factory=forbidden_client)
    (tmp_path / "b.txt").write_text(json.dumps(MINIMAL_BASELINE))
    with pytest.raises(SystemExit, match=r"--baseline: --baseline must name a \.json file"):
        eval_run.main(["--baseline", str(tmp_path / "b.txt"), "--output-dir", str(tmp_path)], client_factory=forbidden_client)

    # tmp_path is neither the repository nor the --output-dir given here.
    with pytest.raises(SystemExit, match="--write-baseline: Refusing to write baseline outside"):
        eval_run.main(["--write-baseline", str(tmp_path / "b.json"), "--output-dir", str(tmp_path / "out")], client_factory=forbidden_client)
    with pytest.raises(SystemExit, match=r"--write-baseline: --write-baseline path must not contain '\.\.'"):
        eval_run.main(["--write-baseline", str(tmp_path / "out" / ".." / "b.json"), "--output-dir", str(tmp_path / "out")], client_factory=forbidden_client)
    with pytest.raises(SystemExit, match="--from-results is not a directory"):
        eval_run.main(["--from-results", str(tmp_path / "missing"), "--write-baseline", str(tmp_path / "b.json"), "--output-dir", str(tmp_path)])
    assert not (tmp_path / "b.json").exists()
    assert not (tmp_path / "out").exists()


def test_validate_scenarios_accepts_only_existing_fixture_files():
    assert eval_run.validate_scenarios(["signups_table_explanation"]) == [
        "signups_table_explanation"
    ]
    with pytest.raises(ValueError, match="Unknown scenario"):
        eval_run.validate_scenarios(["signups_table_explanation", "../../etc/passwd"])
    with pytest.raises(ValueError, match=r"Unknown scenario\(s\): 'nope'"):
        eval_run.validate_scenarios(["nope"])


def test_unknown_scenario_rejected_before_any_file_is_opened(tmp_path, monkeypatch):
    def forbidden_open(*args, **kwargs):  # pragma: no cover - only hit on regression
        raise AssertionError(f"open() called before scenario validation: {args[:1]}")

    def forbidden_client():  # pragma: no cover - only hit on regression
        raise AssertionError("client built before scenario validation")

    monkeypatch.setattr(builtins, "open", forbidden_open)
    monkeypatch.setattr(io, "open", forbidden_open)

    with pytest.raises(SystemExit) as excinfo:
        eval_run.main(
            ["--output-dir", str(tmp_path), "--scenarios", "../../etc/passwd"],
            client_factory=forbidden_client,
        )

    assert "Unknown scenario(s): '../../etc/passwd'" in str(excinfo.value)
    assert "Available: " in str(excinfo.value)
    assert "signups_table_explanation" in str(excinfo.value)
    assert not (tmp_path / "summary.md").exists()
