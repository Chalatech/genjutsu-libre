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
| Reference video (this workflow needs it) | yes | yes | yes, Seedance 2 and newer (`video_url` references) |
| Reference images | up to 2 | up to 30 | up to 30 (`OPENROUTER_MAX_REF_IMAGES`) |
| Draft then final pass | yes (draft, then 1080p) | yes (480p draft, then 720p/1080p) | no draft: the second pass is a new, separately billed generation at a higher resolution |
| Duration | 4-30 s | 4-30 s, or `-1` = same as the input | 4-30 s, integers; the source length rounded and clamped |
| Needs a webhook | yes (cloudflared tunnel or your own URL) | no, polling | no |
| Public media URLs | tmpfiles / catbox | no, uses Kie's file API (kept 24 h) | yes, OpenRouter has no storage: Kie's file API or tmpfiles (`OPENROUTER_MEDIA_HOST`) |
| Real faces accepted | yes (`pass_faces`) | unknown, measured in the first real run | unknown |
| Price | per plan | `KIE_UNIT_CREDITS` x billable seconds x $0.005 | per video token (Seedance 2.5: $0.0000107, or $0.0000064 with a video input); the real `usage.cost` is stored |

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

Model `OPENROUTER_MODEL` (default `bytedance/seedance-2.5`) through the asynchronous video API
(`POST /api/v1/videos`, then `GET /api/v1/videos/{id}` until `completed`). The composite goes first as a
`video_url` entry of `input_references`, then the character images as `image_url` entries, with the same
`@Video1` / `@Image1...` notes as Kie. OpenRouter documents that audio and video references are honored only by
providers that support them, including BytePlus Seedance generation 2 and newer, so this workflow runs only
with those models. Any other model (for example `alibaba/wan-3.0`, image references only) stops before anything
is prepared or uploaded.

* **Capabilities** come from the free `GET /api/v1/videos/models` (the general `/api/v1/models` catalog does
  not list video models), with a small table of known models as fallback (`KNOWN_MODELS`).
* **Request**: `duration` is the source length rounded and clamped to the supported durations (4-30 s),
  `resolution` is `OPENROUTER_RESOLUTION` (default `480p`), `aspect_ratio` is the supported one closest to the
  source, `generate_audio` is `false` (the original audio is restored afterwards).
* **Media host**: OpenRouter has no file storage and needs public HTTPS URLs. `OPENROUTER_MEDIA_HOST=kie`
  uploads through Kie's file API (needs `KIE_API_KEY`; the default when that key exists); `tmpfiles` uses
  tmpfiles with no fallback hosts. Catbox is never used for this provider.
* **Billing**: per video token, not per second. The pre-send estimate (checked against `GEN_MAX_USD`) is an
  estimate only: width x height x 24 fps x seconds / 1024 tokens for the output plus the input video, at the
  video-input rate. The real price is `usage.cost` from the completed job; it is saved as
  `provider-usage.json` next to the result.
* **No draft**: OpenRouter has no draft concept. "HD" (`submit_hd`) submits a **new generation** at the next
  higher resolution (720p), billed again in full. It does not reuse or upscale the first task.
* **Keys**: `OPENROUTER_API_KEY`, or `OPENROUTER_KEY` when that one is unset.
* Local input limits default to Kie's documented Seedance 2.5 limits (24-60 fps, 2-30 s, 409 600-927 408 px,
  sizes 300-6000 px); override with `OPENROUTER_MIN_PIXELS`, `OPENROUTER_MAX_PIXELS`,
  `OPENROUTER_MAX_REF_SECONDS` and `OPENROUTER_MAX_REF_IMAGES`.

More than two character images: `workflow.py submit --reference a.png --reference b.png --reference c.png`
(the first is primary; `--second-reference` still works). `seedance_bridge.submit(..., extra_references=[...])`
is the same thing for callers. Providers cap the count with `max_reference_images` (Enhancor: 2).

### Enhancor (`GEN_BACKEND=enhancor`)

Upstream behaviour, unchanged. See [ENHANCOR-API.md](ENHANCOR-API.md) and [MEDIA-HOSTING.md](MEDIA-HOSTING.md).
