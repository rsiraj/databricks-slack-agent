import worker


def test_run_status_uses_one_message_and_hides_older_steps(monkeypatch):
    posts = []
    updates = []
    monkeypatch.setattr(
        worker,
        "post_message",
        lambda channel, text, thread_ts=None, blocks=None: (
            posts.append((channel, text, thread_ts)) or "status-ts"
        ),
    )
    monkeypatch.setattr(
        worker,
        "update_message",
        lambda channel, ts, text: updates.append((channel, ts, text)),
    )

    status = worker.RunStatus("C123", "thread-1")
    for index in range(7):
        status.start_tool(f"Check {index}")
        status.finish_tool(error=index == 5)
    status.complete()

    assert len(posts) == 1
    assert posts[0] == (
        "C123",
        "_Working on it..._\n_- In progress: Check 0_",
        "thread-1",
    )
    assert updates
    assert all(channel == "C123" and ts == "status-ts" for channel, ts, _ in updates)
    final = updates[-1][2]
    assert final.startswith("_Checked_\n_2 earlier check(s) hidden_")
    assert "Check 0" not in final
    assert "Check 1" not in final
    assert "_- Had trouble: Check 5_" in final
    assert "_- Check 6_" in final


def test_delivery_failure_is_added_to_existing_status(monkeypatch):
    updates = []
    monkeypatch.setattr(worker, "post_message", lambda *args, **kwargs: "status-ts")
    monkeypatch.setattr(
        worker,
        "update_message",
        lambda channel, ts, text: updates.append(text),
    )

    status = worker.RunStatus("C123", "thread-1")
    status.start_tool("Run Databricks query")
    status.finish_tool()
    status.complete()
    status.delivery_failed(1, 3)

    assert updates[-1].startswith("_Checked_")
    assert "_- Had trouble: Answer delivery stopped after 1 of 3 part(s)_" in updates[-1]
