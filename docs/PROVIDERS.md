# Generation providers

The pipeline, the app and the CLI talk to one interface, `GenerationProvider` (`providers/base.py`). Pick the
service with `GEN_BACKEND` in `.env` (default `enhancor`). In the app you can also choose per job from a list
built from the registry, with each provider's capabilities and an estimated cost before anything is sent.

## What the facade guarantees for every provider

`seedance_bridge.py` owns the behaviour that must be identical for every service:

* refuses a second submission for a job that already has a provider response (resume or prepare a new job);
* writes `seedance-request.json` before sending and `provider-task.json` (provider and task id) right after,
  before any polling, so a crash never loses a paid task; a resumed job keeps the provider that created it;
* sends a paid request **once**, never retries it;
* enforces `GEN_MAX_USD` before uploading anything;
* restores the entire original audio at the end (generated audio is always discarded).

## The interface

| Member | Purpose |
|---|---|
| `name` | registry key |
| `capabilities()` | `Capabilities`: reference video yes/no, max reference images, draft yes/no, draft and final resolutions, duration range, needs webhook, accepts real faces (`None` = not measured) |
| `preflight()` | cheap checks (keys, webhook, catalog) before media is prepared or uploaded |
| `doctor()` | `Check` lines for `doctor.py`, offline |
| `validate(job)` | raise `ValueError` listing every local problem, before any upload |
| `estimate_cost(job)`, `estimate_final_cost(seconds)` | USD or `None` when no price is known |
| `upload(path)` | return a URL the service accepts |
| `build_request(job, urls)` | the request body |
| `submit(payload, final=False)` | one paid request; returns `(task_id, raw_response)` or raises `SubmissionRejected` |
| `poll(task_id)` | `PollResult(state='pending'|'done'|'failed', result_url, error, raw)` |
| `download(url, dest)` | default is a plain GET |
| `is_draft_request(payload)`, `build_final(draft_task_id, original, resolution)` | second pass, if `draft` is true |

## Adding a provider

1. Create `providers/<name>.py` with a class that implements the abstract members above. Everything specific
   to the service (URLs, field names, limits, billing) stays in that file.
2. Register it in `providers/__init__.py` (`_REGISTRY`), or call `providers.register(name, module, class)`.
3. Describe what it needs in `doctor()` so `doctor.py` and the UI can show it.
4. Add tests with a mocked HTTP layer (see `tests/test_providers.py`: the common contract is parametrized over
   every registered provider). No test may make a paid call.

## Capabilities (as of 2026-11-04)

| | `enhancor` | `kie` | `openrouter` |
|---|---|---|---|
| Reference video (this workflow needs it) | yes | yes | **no** (see below) |
| Reference images | up to 2 | up to 30 | images only |
| Draft then final pass | yes (draft, then 1080p) | yes (480p draft, then 720p/1080p) | no |
| Duration | 4-30 s | 4-30 s, or `-1` = same as the input | n/a |
| Needs a webhook | yes (cloudflared tunnel or your own URL) | no, polling | no |
| Public media URLs | tmpfiles / catbox | no, uses Kie's file API (kept 24 h) | n/a |
| Real faces accepted | yes (`pass_faces`) | unknown, measured in the first real run | unknown |
| Price | per plan | `KIE_UNIT_CREDITS` x billable seconds x $0.005 | n/a |

### Kie.ai (`GEN_BACKEND=kie`)

Model `bytedance/seedance-2-5`, reference scenario with the composite (vocals embedded) as `@Video1` and the
character images as `@Image1...`, `draft: true`, 480p, `duration: -1`, `generate_audio: false`. The final pass
uses `bytedance/seedance-2-5-draft` with `draft_task_id` (the draft id is valid 7 days). Kie answers HTTP 200
even on failure, so every response is judged by its `code`; a task can wait 30+ minutes and fail without charge.

Local validation before uploading anything (all problems reported together):

* reference video: mp4/mov/mkv, 24-60 fps, 2-30 s, sides 300-6000 px, aspect 0.4-2.5, width x height between
  409 600 and 927 408 px (960x540 = 518 400 passes), at most 200 MB;
* images: jpeg/png/webp, sides 300-6000 px, aspect 0.4-2.5, under 30 MB;
* total reference video at most 30 s.

Billing with a reference video: unit price x (seconds of reference video + seconds of output). Published price
is "from $0.085/s (17 credits)"; the price per variant (resolution, draft) is not published, so
`estimate_final_cost` is a lower bound. Estimated cost for a 15 s clip: 17 x (15 + 15) x $0.005 = $2.55.
Settings: `KIE_API_KEY`, `KIE_MODEL`, `KIE_UNIT_CREDITS` (correct it after measuring), `KIE_HD_RESOLUTION`.
If the final pass rejects 720p, Kie's own docs list only 1080p for the draft completion: set
`KIE_HD_RESOLUTION=1080p`.

### OpenRouter (`GEN_BACKEND=openrouter`)

Capabilities are declared per configured model (`OPENROUTER_MODEL`). The public catalog
(`GET /api/v1/models`, free) does **not** list video models, so a model missing from it proves nothing. The
provider therefore uses the catalog when it does list the model, otherwise a small table of known models
(`KNOWN_MODELS` in `providers/openrouter.py`), otherwise treats the model as unsupported.

As of 2026-11-04, OpenRouter's Wan 3.0 (`alibaba/wan-3.0`) accepts **only image references**, not a reference
video, so it cannot run this workflow yet: the run stops before anything is prepared or uploaded, with an
explicit message. Submission is implemented only for what is confirmed to work, which today is nothing. Use
`kie` or `enhancor`.

### Enhancor (`GEN_BACKEND=enhancor`)

Upstream behaviour, unchanged. See [ENHANCOR-API.md](ENHANCOR-API.md) and [MEDIA-HOSTING.md](MEDIA-HOSTING.md).
