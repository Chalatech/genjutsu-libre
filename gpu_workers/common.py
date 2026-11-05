"""Shared helpers for the GPU host workers (separate, depth, segment).

Each worker is a CLI ``--in <dir> --out <dir>`` that writes ``report-<worker>.json``
with wall time, per-stage times and peak memory. Exit codes: 0 ok, 1 error,
2 bad arguments, 3 not enough free memory after waiting (GPU_MIN_FREE_GB,
re-checked every GPU_MEM_POLL_S seconds for up to GPU_MEM_WAIT_MIN minutes).

Nothing here imports torch at module level, so the mask helpers run anywhere.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXIT_OK, EXIT_ERROR, EXIT_LOW_MEMORY = 0, 1, 3
MASK_RE = re.compile(r"mask_(\d+)\.png")

# On a busy host with fragmented memory, transparent huge pages make each fresh CPU
# allocation wait for compaction (measured: 336 MB took 9-95 s instead of 0.02-0.1 s).
# Both switches are read when numpy / torch (its mimalloc allocator) load, so this
# module must be imported first.
NO_HUGE_PAGES = {"MIMALLOC_ALLOW_THP": "0", "NUMPY_MADVISE_HUGEPAGE": "0"}
for _name, _value in NO_HUGE_PAGES.items():
    os.environ.setdefault(_name, _value)


def gpu_dir() -> Path:
    value = os.getenv("GPU_DIR")
    return Path(value).expanduser() if value else HERE.parent


def setup_env(root: Path | None = None) -> Path:
    """Keep model caches inside GPU_DIR (never the shared ~/.cache)."""
    root = root or gpu_dir()
    os.environ.setdefault("TORCH_HOME", str(root / "models" / "torch"))
    os.environ.setdefault("HF_HOME", str(root / "hf-cache"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return root


def free_memory_gb(meminfo: str = "/proc/meminfo") -> float | None:
    try:
        with open(meminfo) as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return round(int(line.split()[1]) / 1024 / 1024, 1)
    except (OSError, ValueError, IndexError):
        pass
    return None


def min_free_gb() -> float:
    return float(os.getenv("GPU_MIN_FREE_GB", "20"))


def wait_for_memory(min_gb: float, *, read=None, wait_s: float | None = None, poll_s: float | None = None,
                    sleep=time.sleep, clock=time.monotonic, log=None) -> tuple[bool, float | None, float]:
    """Wait until MemAvailable >= min_gb. Returns (ok, available_gb, waited_s).

    Unknown memory (no /proc/meminfo) counts as enough.
    """
    read = read or free_memory_gb
    wait_s = float(os.getenv("GPU_MEM_WAIT_MIN", "30")) * 60 if wait_s is None else wait_s
    poll_s = float(os.getenv("GPU_MEM_POLL_S", "120")) if poll_s is None else poll_s
    log = log or (lambda message: print(message, file=sys.stderr, flush=True))
    started = clock()
    while True:
        available = read()
        waited = round(clock() - started, 1)
        if available is None or available >= min_gb:
            return True, available, waited
        if waited + poll_s > wait_s:
            return False, available, waited
        log(f"only {available} GB available, need {min_gb} GB; re-checking in {poll_s:.0f} s")
        sleep(poll_s)


def lower_priority(target: int = 10) -> int:
    """Raise our own niceness to at least ``target`` (never touches other processes)."""
    try:
        current = os.nice(0)
        return os.nice(target - current) if current < target else current
    except OSError:
        return -1


@contextmanager
def skip_weight_init(torch):
    """Build modules without random initialization (the checkpoint overwrites every weight).

    On the GPU host's busy CPU, demucs' random init alone took 30-130 s.
    """
    names = ("uniform_", "normal_", "trunc_normal_", "kaiming_uniform_", "kaiming_normal_",
             "xavier_uniform_", "xavier_normal_", "orthogonal_")
    saved = {name: getattr(torch.nn.init, name) for name in names if hasattr(torch.nn.init, name)}
    try:
        for name in saved:
            setattr(torch.nn.init, name, lambda tensor, *args, **kwargs: tensor)
        yield
    finally:
        for name, function in saved.items():
            setattr(torch.nn.init, name, function)


def rss_peak_gb() -> float | None:
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(peak / (1024**3 if sys.platform == "darwin" else 1024**2), 2)
    except (ImportError, OSError):
        return None


# ---- video files --------------------------------------------------------------

def read_frames(path: Path) -> list:
    """Every frame of a video as an RGB uint8 array (the clips are <= 30 s, <= 960 px)."""
    import av

    with av.open(str(path)) as container:
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]


def count_frames(path: Path) -> int:
    import av

    with av.open(str(path)) as container:
        return sum(1 for _ in container.decode(video=0))


def write_video(path: Path, frames, fps: int = 24) -> int:
    """H.264 yuv420p at ``fps`` with pts = frame index, like pipeline.writer()/put()."""
    from fractions import Fraction

    import av

    count = 0
    container = stream = None
    try:
        for index, array in enumerate(frames):
            if container is None:
                container = av.open(str(path), "w")
                stream = container.add_stream("libx264", rate=fps)
                stream.height, stream.width = array.shape[:2]
                stream.pix_fmt = "yuv420p"
                stream.options = {"crf": "18", "preset": "fast"}
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, fps)
            for packet in stream.encode(frame):
                container.mux(packet)
            count += 1
        if container is None:
            raise ValueError("no frames to write")
        for packet in stream.encode():
            container.mux(packet)
    finally:
        if container is not None:
            container.close()
    return count


# ---- mask files ---------------------------------------------------------------

def mask_name(index: int) -> str:
    """Frame index -> file name used by composite() (``mask_00000.png``)."""
    if index < 0:
        raise ValueError("frame index must be >= 0")
    return f"mask_{index:05d}.png"


def mask_files(folder: Path) -> list[Path]:
    """Sorted mask files of a folder; they must be mask_0..N-1 with no gaps."""
    items = []
    for path in Path(folder).glob("mask_*.png"):
        match = MASK_RE.fullmatch(path.name)
        if match:
            items.append((int(match.group(1)), path))
    items.sort()
    if [index for index, _ in items] != list(range(len(items))):
        raise ValueError(f"missing or shifted frame indices in {folder}")
    return [path for _, path in items]


def read_mask(path: Path):
    import numpy as np
    from PIL import Image

    return np.asarray(Image.open(path).convert("L")) > 127


def write_mask(path: Path, mask) -> None:
    import numpy as np
    from PIL import Image

    Image.fromarray(np.asarray(mask, dtype=bool).astype(np.uint8) * 255).save(path, optimize=False, compress_level=1)


def parse_ids(text: str) -> list[int]:
    """'1,3' -> [1, 3] (sorted, unique, positive)."""
    ids = set()
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if not part.isdigit() or int(part) < 1:
            raise ValueError(f"instance ids are positive integers, got {part!r}")
        ids.add(int(part))
    if not ids:
        raise ValueError("no instance ids given")
    return sorted(ids)


def union_masks(instances_dir: Path, ids: list[int], out_dir: Path) -> dict:
    """Write the per-frame union of ``instances/<id>/`` into ``out_dir/mask_00000.png...``."""
    import numpy as np

    instances_dir, out_dir = Path(instances_dir), Path(out_dir)
    available = sorted(int(p.name) for p in instances_dir.iterdir() if p.is_dir() and p.name.isdigit()) if instances_dir.is_dir() else []
    missing = [i for i in ids if i not in available]
    if missing:
        raise ValueError(f"unknown instance ids {missing}; available: {available}")
    sequences = [mask_files(instances_dir / str(i)) for i in ids]
    counts = {len(files) for files in sequences}
    if len(counts) != 1 or not counts.pop():
        raise ValueError(f"instances {ids} have different or empty frame counts: {[len(f) for f in sequences]}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("mask_*.png"):
        stale.unlink()
    nonempty = 0
    for index, paths in enumerate(zip(*sequences)):
        merged = np.logical_or.reduce([read_mask(p) for p in paths])
        nonempty += bool(merged.any())
        write_mask(out_dir / mask_name(index), merged)
    return {"ids": ids, "frames": len(sequences[0]), "nonempty_frames": nonempty}


# ---- reports and the worker wrapper ------------------------------------------

class Report(dict):
    """report-<worker>.json: wall time, per-stage seconds and peak memory."""

    def __init__(self, worker: str, out_dir: Path):
        super().__init__(worker=worker, started=time.strftime("%Y-%m-%dT%H:%M:%S"), stages={})
        self.out_dir = Path(out_dir)
        self._t0 = time.perf_counter()

    @contextmanager
    def stage(self, name: str):
        """Time a stage; progress goes to stderr so a remote caller can follow it."""
        started = time.perf_counter()
        print(f"[{self['worker']}] {time.strftime('%H:%M:%S')} {name}...", file=sys.stderr, flush=True)
        try:
            yield
        finally:
            self["stages"][name] = round(time.perf_counter() - started, 2)
            print(f"[{self['worker']}] {name} {self['stages'][name]:.1f} s", file=sys.stderr, flush=True)

    def write(self, torch=None) -> Path:
        self["wall_s"] = round(time.perf_counter() - self._t0, 2)
        if torch is not None and torch.cuda.is_available():
            self["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
            self["gpu_reserved_peak_gb"] = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
        self["rss_peak_gb"] = rss_peak_gb()
        path = self.out_dir / f"report-{self['worker']}.json"
        path.write_text(json.dumps(self, indent=2))
        return path


def run_worker(name: str, args, work) -> int:
    """Memory guard, CUDA check, timing and report around ``work(args, report, torch)``."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = Report(name, out)
    report["nice"] = lower_priority()
    root = setup_env()
    need = min_free_gb()
    ok, available, waited = wait_for_memory(need)
    report.update(memory_available_gb=available, min_free_gb=need, waited_for_memory_s=waited)
    if not ok:
        report["error"] = f"only {available} GB of memory available after {waited:.0f} s, need {need} GB"
        print(json.dumps({"worker": name, "exit": EXIT_LOW_MEMORY, "error": report["error"]}), flush=True)
        report.write()
        return EXIT_LOW_MEMORY
    torch = None
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available to torch")
        with report.stage("cuda_init"):  # create the context before any slow CPU work
            torch.zeros(1, device="cuda")
            torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        report["gpu"] = torch.cuda.get_device_name(0)
        work(args, report, torch, root)
        code = EXIT_OK
    except Exception as exc:  # report every failure, then exit 1
        report["error"] = f"{type(exc).__name__}: {exc}"[:1000]
        code = EXIT_ERROR
    path = report.write(torch)
    summary = {k: report.get(k) for k in ("worker", "wall_s", "gpu_peak_gb", "rss_peak_gb", "error") if report.get(k) is not None}
    print(json.dumps({**summary, "exit": code, "report": path.name}), flush=True)
    return code


def worker_parser(description: str):
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--in", dest="inp", required=True, help="input folder")
    parser.add_argument("--out", required=True, help="output folder (created if missing)")
    return parser
