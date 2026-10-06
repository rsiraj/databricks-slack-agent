# Model providers

The worker talks to the model through `utils/llm/`, a small provider layer.
The default deployment runs Anthropic only. The OpenAI Responses translation is fully
implemented and unit-tested, the `openai` SDK is a locked runtime dependency,
and the **evaluation harness** can call OpenAI with its own evaluation-only key
(`MODEL_EVAL_OPENAI_API_KEY`, a repository Actions secret). A default deployment has
**no** `openai_api_key` in the app secret, so `LLM_PROVIDER=openai` on the
worker would raise `ProviderNotConfigured` at the first model call. This page
is the reference for what the two providers send and receive, what changes if
a thread switches providers, and what remains before OpenAI could run in
production.

## Layout

| Module | Role |
|---|---|
| `utils/llm/types.py` | `ToolSpec`, `ModelRequest`, `ModelResponse`, `Usage`, the `provider_reasoning` block |
| `utils/llm/base.py` | `Provider` protocol, `ProviderNotConfigured` |
| `utils/llm/anthropic_provider.py` | Messages API call, byte-identical to the pre-provider worker request |
| `utils/llm/openai_provider.py` | Responses API translation (pure functions) + gated client |
| `utils/llm/__init__.py` | `get_provider()`, `resolve_model()`, env handling |
| `tools/registry.py` | `all_tool_specs()` derived from the Anthropic-shaped tool lists |
| `prompts/providers/<name>.md` | Provider-specific notes appended after `core/` + `business/` |

### Environment

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic` or `openai`. Unknown values fail at import so a typo cannot silently pick a provider. |
| `LLM_MODEL` | unset | Provider-neutral model override; wins over the two below when set. |
| `ANTHROPIC_MODEL` | `claude-sonnet-5` | Anthropic model id. |
| `OPENAI_MODEL` | `gpt-5.6-terra` | OpenAI model id. |
| `ANTHROPIC_EFFORT` | `high` | Reasoning effort, applied to whichever provider is selected (`low` / `medium` / `high` / `xhigh` / `max`). |
| `ANTHROPIC_MAX_TOKENS` | `24000` (Terraform) | Output cap. OpenAI raises it to at least 25,000 (see below). When unset, the worker uses `recommended_max_tokens(model, effort)` (`utils/llm/anthropic_provider.py`) — 24000 for `claude-sonnet-5` at `high` — except `claude-sonnet-4-6`, which stays at the 16000 floor. |
| `ANTHROPIC_STREAM` | `auto` | Anthropic transport: `auto` streams only when `max_tokens > 21000`; `always` / `never` force it. Same kwargs, same `Message`, same audit fields either way. |
| `ANTHROPIC_TIMEOUT_SECONDS` | `180` | Per-request timeout, bounded by the Lambda deadline, for either provider. |

Terraform exposes `llm_provider` and `llm_model` (module and root variables)
and maps them to `LLM_PROVIDER` / `LLM_MODEL` on the worker.

## Canonical message format

DynamoDB stores the **Anthropic block format** for every thread (365-day TTL),
whichever provider produced it. Providers translate at the edge. One block
type was added so a non-Anthropic provider can carry state across turns:

```json
{"type": "provider_reasoning", "provider": "openai", "payload": {"type": "reasoning", "id": "rs_...", "encrypted_content": "..."}}
```

Rules every consumer follows:

- `AnthropicProvider` **strips** `provider_reasoning` before sending (Anthropic rejects unknown block types). A turn that would become empty gets a one-line placeholder text block.
- `OpenAIProvider` **replays** its own `payload` verbatim as an input item and **strips** Anthropic `thinking` / `redacted_thinking`. Blocks written by a different provider are skipped.
- `utils/compaction.py` drops `provider_reasoning` from assistant turns of *previous* agent runs (same boundary as tool-result compaction). It is kept for every turn of the current run because the Responses API pairs each `function_call` with the `reasoning` item that produced it.
- `worker._extract_text`, `worker._replace_assistant_text`, `NumericEvidenceGuard.record_history`, `JobRunEvidenceGuard`, `utils/sessions.py` and `scripts/export_sessions.py` all skip it (the numeric guard never grounds numbers from a payload; the export renders it as an opaque note).

## Translation table

### Request

| Concept | Anthropic Messages API | OpenAI Responses API |
|---|---|---|
| System prompt | `system=[{"type":"text","text":…,"cache_control":{"type":"ephemeral"}}]` | `instructions=…` (no cache marker; automatic prefix caching) |
| Conversation cache | second `cache_control: ephemeral` on the last content block of the last message (moves each iteration; string user text is sent as one `text` block to carry it; written to a copy, never persisted) | none; translation drops any `cache_control` key, automatic caching covers the prefix |
| Model | `model` | `model` |
| Output cap | `max_tokens` (default 24,000 for `claude-sonnet-5` at `high`) | `max_output_tokens = max(max_tokens, 25000)`; caps reasoning + output, so lower values can starve the answer (logged when raised) |
| Reasoning | `thinking={"type":"adaptive"}` + `output_config={"effort": e}` | `reasoning={"effort": e}`, `include=["reasoning.encrypted_content"]`, `store=False` |
| Effort levels | `low` `medium` `high` `max` (Sonnet 4.6); `xhigh` also on `claude-sonnet-5` / `claude-opus-5`. `xhigh` → `max` where unsupported | `none` `low` `medium` `high` `xhigh` `max`, passed through |
| Tools | `{name, description, input_schema}` | `{"type":"function", name, description, parameters}` |
| Timeout | `anthropic.Timeout(connect=10, read/write/pool=bounded)` | `timeout=<bounded seconds>` |
| User text | `{"type":"text","text"}` in a user message | `{"role":"user","content":[{"type":"input_text","text"}]}` |
| User image | `{"type":"image","source":{"type":"base64","media_type","data"}}` | `{"type":"input_image","image_url":"data:<mime>;base64,…"}` (URL sources pass through) |
| Assistant text | `{"type":"text","text"}` | `{"role":"assistant","content":[{"type":"output_text","text"}]}` |
| Tool call | `{"type":"tool_use","id","name","input"}` | `{"type":"function_call","call_id":id,"name","arguments":json}` |
| Tool result | `{"type":"tool_result","tool_use_id","content","is_error"}` | `{"type":"function_call_output","call_id","output"}`; `is_error` folded in as `{"error": …}` unless the content is already a JSON object with `error` |
| Anthropic thinking | sent as-is (current run) / stripped (prior runs) | stripped |
| `provider_reasoning` (openai) | stripped | payload replayed verbatim |

### Response

| Concept | Anthropic | OpenAI | Stored as |
|---|---|---|---|
| Text | `text` block | `message` → `output_text` part | `{"type":"text"}` |
| Tool call | `tool_use` | `function_call` (`call_id`, parsed `arguments`; unparseable → `{}` + warning) | `{"type":"tool_use","id":call_id,…}` |
| Reasoning | `thinking` / `redacted_thinking` | `reasoning` item | as-is / `provider_reasoning` |
| Refusal | `stop_reason: refusal` | `refusal` content part | `stop_reason` `refusal` (refusal text kept as a text block) |
| Model | `resp.model` | `resp.model` | audited as `model`; requested id as `requested_model` |

### Stop reasons

`ModelResponse.stop_reason` is normalized; `raw_stop_reason` keeps the
provider's value for the `model_call` audit and the "Stopped unexpectedly"
message.

| Normalized | Anthropic `stop_reason` | OpenAI |
|---|---|---|
| `tool_use` | `tool_use` | any `function_call` item in `output` (wins over truncation) |
| `max_tokens` | `max_tokens` | `status: incomplete` + `incomplete_details.reason: max_output_tokens` |
| `refusal` | `refusal` | a `refusal` content part |
| `end_turn` | `end_turn` | `status: completed` |
| `pause_turn` | `pause_turn` | n/a |
| `other` | anything else (e.g. `stop_sequence`) | other `incomplete` reasons, `failed`, `cancelled` |

### Usage

Both are recorded in Anthropic's accounting so cost math and cache-hit rates
stay comparable:

| Field | Anthropic | OpenAI |
|---|---|---|
| `input_tokens` | `usage.input_tokens` (excludes cache reads) | `usage.input_tokens − input_tokens_details.cached_tokens` |
| `cache_creation_input_tokens` | `usage.cache_creation_input_tokens` | `0` (no write charge) |
| `cache_read_input_tokens` | `usage.cache_read_input_tokens` | `usage.input_tokens_details.cached_tokens` |
| `output_tokens` | `usage.output_tokens` (includes thinking) | `usage.output_tokens` (includes reasoning) |

`evals/pricing.py` prices `gpt-5.6-terra` at $2 / $0.20 cached / $12 per MTok
with no cache-write premium (`cache_write_multiplier: 1.0`); Claude rows keep
the 1.25× / 0.1× multipliers.

## Switching providers mid-thread

Sessions are never rewritten, so a thread started on one provider continues on
another. What is lost:

- **Reasoning continuity.** Anthropic `thinking` blocks are dropped when
  sending to OpenAI; OpenAI `provider_reasoning` is dropped when sending to
  Anthropic. The visible text, tool calls and tool results all survive, so the
  model still sees *what* happened, not *why* it decided it.
- **Tool-call pairing across providers.** Replayed `tool_use` / `tool_result`
  pairs translate cleanly in both directions (`id` ↔ `call_id`).
- **Prompt cache.** The system prompt differs by one trailing section
  (`prompts/providers/<name>.md`), so the first call after a switch is a cache
  miss on either side.
- **Effort semantics.** The same level name means different compute on each
  vendor; compare `model_call` usage and latency, not the label.

Reasoning from *previous* runs is dropped by compaction on both providers.
On OpenAI this also means prior-run `function_call` items are replayed
without their `reasoning` item; validate on a multi-turn thread at enablement
(see the checklist) and, if the API rejects that shape, keep
`provider_reasoning` for all turns (one condition in
`utils/compaction.py`).

## Enablement checklist (OpenAI)

Steps 1 and 4 are in place; the production path still raises
`ProviderNotConfigured("OpenAI provider is not enabled: install the openai
package and add openai_api_key to the app secret")` because step 2 is not
done. **Only the evaluation harness holds an OpenAI key today.**

1. **Dependency — done.** `openai>=3.8,<4` is in `[project] dependencies`
   and `uv.lock`; the Lambda image installs it from the lock (about 15 MB of
   site-packages, no new transitive dependencies — it shares `httpx2`,
   `pydantic`, `anyio` and `jiter` with `anthropic`). `OpenAIProvider._client()`
   now passes the import check and fails only on the missing secret.
2. **Secret — not done.** Add `"openai_api_key": "sk-…"` to the consolidated
   app secret (`APP_SECRET_ARN`). `utils/secrets.get_secret("openai_api_key")`
   reads it with the same 5-minute cache as the other keys. Until this exists,
   `LLM_PROVIDER=openai` in production raises `ProviderNotConfigured` on the
   first call; do not flip the Terraform variable before it.
3. **Baseline first.** Export the current `model_call` / `run_summary`
   measurements (`scripts/export_sessions.py`) so there is a comparable
   window before the switch.
4. **Offline comparison — available.** The `Model comparison` workflow
   (`.github/workflows/model-comparison.yml`) has `MODEL_EVAL_OPENAI_API_KEY`
   (an evaluation-only OpenAI project key, distinct from the app secret) and
   accepts provider-qualified model ids, so one run yields one table:
   `gh workflow run model-comparison.yml -f
   models='anthropic:claude-sonnet-4-6,openai:gpt-5.6-terra' -f efforts='high'
   -f repeats='3'`. Locally: `MODEL_EVAL_ANTHROPIC_API_KEY=…
   MODEL_EVAL_OPENAI_API_KEY=… python -m evals.run --models
   anthropic:claude-sonnet-4-6,openai:gpt-5.6-terra --repeats 3`. See
   `evals/README.md`.
5. **Flip.** Set `llm_provider = "openai"` (Terraform) — optionally
   `llm_model` — and apply. `ANTHROPIC_EFFORT` / `ANTHROPIC_MAX_TOKENS` /
   `ANTHROPIC_TIMEOUT_SECONDS` keep their meaning.
6. **Validate live.** Run one fresh thread and one multi-turn thread. Check
   `model_call` events show `provider: openai`, cache reads climb after the
   first call, and no 400 about a `function_call` missing its `reasoning`
   item (see the note above).
7. **Roll back** by setting `llm_provider = "anthropic"`; existing threads
   continue with the losses described above.
