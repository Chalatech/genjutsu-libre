# Preparing on your own GPU host

By default preparation (vocal/music separation, depth, subject masks) runs on hosted models through Replicate
or Hugging Face. With `PREP_BACKEND=gpu` the same three stages run on a GPU machine you control, and no
Replicate token is needed. Generation is a separate choice (`GEN_BACKEND`, see [PROVIDERS.md](PROVIDERS.md)).

| Stage | Hosted default | GPU host |
|---|---|---|
| Voice and music | Replicate `cjwbw/demucs` | Demucs `htdemucs` |
| Depth in colour | Replicate `lucataco/depth-anything-video` | Video-Depth-Anything **Small** (Apache-2.0), inferno colormap |
| Subject masks | Replicate `lucataco/sam3-video` | SAM 3 (`transformers`), one mask track per detected subject |

Video-Depth-Anything **Large** is CC-BY-NC and is not used.

## How it works

Everything that is not a model runs on this machine (24 fps normalization, compositing, audio, review).
Only the model calls go to a small inference API on the host:

```
this machine                                   GPU host
app / CLI -> gpu_backend -> gpu_api -- ssh tunnel --> 127.0.0.1:GPU_API_PORT (gpu_workers/server.py)
```

* The API listens on the host's loopback only. This machine reaches it through an `ssh -L` tunnel, so no
  port is opened on any network.
* One session per job: the app starts the server if it is not running, opens one tunnel, and for each stage
  uploads the input, waits for the prediction, downloads the result and deletes the host copies.
* Models load on first use and are unloaded after `GPU_API_UNLOAD_IDLE_MIN` minutes without work; the server
  exits by itself after `GPU_API_EXIT_IDLE_MIN` minutes without requests. Uploaded files expire after
  `GPU_API_TTL_MIN` minutes, so a dropped connection never leaves data behind for long.
* Only one model runs at a time. Before each run the server waits for `GPU_MIN_FREE_GB` of free memory,
  re-checking every `GPU_MEM_POLL_S` seconds for up to `GPU_MEM_WAIT_MIN` minutes, then fails with a
  clear "out of memory" error instead of starting.

## One-time setup

You need ssh access with key authentication, a Python 3.12 interpreter on the host and a CUDA-capable GPU.
Nothing needs sudo: everything is written inside one folder, `GPU_DIR`.

1. Add an alias for the host to `~/.ssh/config` (key auth, `BatchMode` friendly) and put these in `.env`:

   ```
   PREP_BACKEND=gpu
   GPU_HOST=<ssh alias>
   GPU_DIR=~/genjutsu-gpu            # the only folder written on the host
   GPU_PYTHON=~/genjutsu-gpu/.venv/bin/python
   ```

2. Copy the workers and run the idempotent setup on the host:

   ```
   rsync -az --exclude __pycache__ gpu_workers/ "$GPU_HOST:$GPU_DIR/workers/"
   ssh "$GPU_HOST" "GPU_DIR=$GPU_DIR bash $GPU_DIR/workers/setup_gpu.sh"
   ```

   `setup_gpu.sh` creates a virtualenv with a CUDA build of PyTorch, installs `requirements-gpu.txt`, clones
   Video-Depth-Anything and downloads its Small weights, and prepares SAM 3. Useful variables (also read from
   `$GPU_DIR/.env` on the host, keep it `chmod 600`): `SAM3_WEIGHTS` (folder with Sam3VideoModel weights in
   transformers format), `SAM3_SOURCE` (an existing local copy to copy once), `HF_TOKEN` (your own token,
   only to download `facebook/sam3`), `TORCH_INDEX` (PyTorch wheel index).

3. Check it: `.venv/bin/python doctor.py --require-keys` verifies ssh, starts the server if needed, opens the
   tunnel and reads `/health`. It prints free memory and the loaded models, never host paths.
   `gpu_workers/check.py` on the host prints a full environment report (it includes absolute paths: do not paste
   it into public places).

## Settings (`.env` on this machine)

| Variable | Default | Meaning |
|---|---|---|
| `PREP_BACKEND` | `replicate` | `gpu` makes the GPU host the default in the app and CLI; `--provider gpu` forces it per run |
| `GPU_HOST`, `GPU_DIR`, `GPU_PYTHON` | | ssh alias, working folder, venv python on the host |
| `GPU_API_PORT` | 18770 | loopback port of the server on the host and of the local tunnel end |
| `GPU_API_TOKEN` | empty | optional bearer token the server requires |
| `GPU_API_UNLOAD_IDLE_MIN` / `GPU_API_EXIT_IDLE_MIN` / `GPU_API_TTL_MIN` | 10 / 30 / 60 | idle unload, idle exit, file expiry |
| `GPU_MIN_FREE_GB` / `GPU_MEM_POLL_S` / `GPU_MEM_WAIT_MIN` | 20 / 120 / 30 | memory guard before each model |
| `SEGMENT_MODEL` | `sam3` | `sam2` is not implemented yet |
| `SAM3_WEIGHTS` | | folder on the host with SAM 3 weights |
| `DEPTH_SUBJECT_STRETCH` | off | see below |

**A running server keeps the limits it started with.** `GPU_MEM_*` and `GPU_API_*` are passed to the server only
when this app starts it. If you change them while a server is still alive (it exits by itself after the idle
time), the change takes effect on the next start. Also, `doctor.py` may start the server and leave it running
until that idle exit; that is expected and harmless.

## Choosing which subjects to replace

SAM 3 tracks every person-like subject and numbers them. The review step shows `instances-contact.jpg` with a
tick box per id. Pick any ids, including things that are not people (an object in the actor's hands, faces on a
poster). Applying the selection rebuilds the mask, composite and prepared input and clears the review approval,
so the new clip must be reviewed again. From the CLI:

```
.venv/bin/python workflow.py instances <job>
.venv/bin/python workflow.py select <job> --instances 1,2,4,9
```

## Optional: `DEPTH_SUBJECT_STRETCH=1`

Far subjects use a narrow slice of the colormap and look like flat single-colour silhouettes. With this on, the
depth colours inside each selected subject are remapped so its 2nd-98th percentile range spans the whole
colormap (bounds smoothed over time, a minimum span so noise is not amplified, edge pixels copy the nearest
interior pixel to avoid a dark outline). It assumes the inferno colormap the GPU backend produces. It is off by
default because it discards the depth relation between subjects; judge it on your own clip before relying on it.

## Measured times

Inputs normalized to 24 fps, 960x540, on a shared Grace-Blackwell class machine with roughly 30 GB of free memory:

| Clip | Frames | Separate | Depth | Segment | Whole job on this machine |
|---|---|---|---|---|---|
| 6 s | 144 | 2 s (0.6 GB GPU) | 12-14 s (3 GB) | 70-73 s (3.7 GB) | 1 min 49 s, server cold start included |
| 15 s | 359 | 2 s | 30 s (2.9 GB) | 299 s (12.5 GB, 14 subjects) | 6 min 30 s |

Compositing and the final audio mix are included in the whole-job time. Memory fragmentation can make the first
CPU allocations very slow on some hosts; the workers set `MIMALLOC_ALLOW_THP=0` and `NUMPY_MADVISE_HUGEPAGE=0`
before importing torch, which removed the stalls on the measured machine.
