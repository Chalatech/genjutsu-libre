#!/usr/bin/env python3
"""Environment check for the GPU host workers.

Prints a JSON report: torch version, CUDA availability, GPU name and capability,
the time of a 4096x4096 matmul, the free system memory and which worker
components are installed. Exit codes: 0 ok, 1 no usable CUDA (torch missing, no
GPU, no kernels for the GPU's capability, failed matmul), 3 not enough free
memory to start (GPU_MIN_FREE_GB, default 30).

    python check.py                # environment + matmul
    python check.py --load-models  # also load SAM 3 and Video-Depth-Anything Small on the GPU
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
VDA_SMALL = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}


def gpu_dir() -> Path:
    value = os.getenv("GPU_DIR")
    return Path(value).expanduser() if value else HERE.parent


def free_memory_gb(meminfo: str = "/proc/meminfo") -> float | None:
    try:
        with open(meminfo) as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return round(int(line.split()[1]) / 1024 / 1024, 1)
    except (OSError, ValueError, IndexError):
        pass
    return None


def _arch_number(item: str) -> int | None:
    match = re.fullmatch(r"(?:sm|compute)_(\d+)[a-z]?", item)
    return int(match.group(1)) if match else None


def capability_supported(capability: tuple[int, int], arch_list: list[str]) -> bool:
    """True when the torch build can run on a GPU of this compute capability.

    Native code (sm_XY) runs on the same major with an equal or higher minor;
    PTX (compute_XY) is JIT-compiled for any equal or newer GPU.
    """
    major, minor = capability
    wanted = major * 10 + minor
    for item in arch_list:
        number = _arch_number(item)
        if number is None:
            continue
        if item.startswith("sm_") and number // 10 == major and number % 10 <= minor:
            return True
        if item.startswith("compute_") and number <= wanted:
            return True
    return False


def version_of(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def components(root: Path) -> dict:
    sam3 = Path(os.getenv("SAM3_WEIGHTS") or root / "models" / "sam3").expanduser()
    sam3_type = None
    try:
        sam3_type = json.loads((sam3 / "config.json").read_text()).get("model_type")
    except (OSError, ValueError):
        pass
    vda_weights = root / "models" / "video-depth-anything" / "video_depth_anything_vits.pth"
    return {
        "segment_model": os.getenv("SEGMENT_MODEL", "sam3"),
        "demucs": version_of("demucs"),
        "torchaudio": version_of("torchaudio"),
        "transformers": version_of("transformers"),
        "opencv": version_of("opencv-python-headless"),
        "av": version_of("av"),
        "sam3_weights": str(sam3) if (sam3 / "model.safetensors").is_file() else None,
        "sam3_model_type": sam3_type,
        "vda_source": (root / "src" / "video-depth-anything" / "video_depth_anything").is_dir(),
        "vda_small_weights": vda_weights.is_file(),
        "htdemucs_cached": any((root / "models" / "torch" / "hub" / "checkpoints").glob("*.th")),
    }


def matmul_ms(torch) -> tuple[float, bool]:
    a = torch.randn(4096, 4096, device="cuda")
    b = torch.randn(4096, 4096, device="cuda")
    (a @ b).sum().item()  # warm-up: cuBLAS init, kernel selection
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(10):
        c = a @ b
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - started) / 10 * 1000
    return round(elapsed, 2), bool(torch.isfinite(c).all().item())


def load_models(torch, root: Path, report: dict) -> dict:
    """Load each model once on the GPU and free it; no inference."""
    result: dict = {}
    sam3 = report["components"]["sam3_weights"]
    if sam3:
        try:
            from transformers import Sam3VideoModel, Sam3VideoProcessor

            started = time.perf_counter()
            model = Sam3VideoModel.from_pretrained(sam3, dtype=torch.bfloat16).to("cuda").eval()
            Sam3VideoProcessor.from_pretrained(sam3)
            torch.cuda.synchronize()
            result["sam3_load_s"] = round(time.perf_counter() - started, 1)
            result["sam3_params_m"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)
            result["sam3_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
            del model
        except Exception as exc:
            result["sam3_error"] = f"{type(exc).__name__}: {exc}"[:400]
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    vda_root = root / "src" / "video-depth-anything"
    weights = root / "models" / "video-depth-anything" / "video_depth_anything_vits.pth"
    if vda_root.is_dir() and weights.is_file():
        try:
            sys.path.insert(0, str(vda_root))
            from video_depth_anything.video_depth import VideoDepthAnything

            started = time.perf_counter()
            model = VideoDepthAnything(**VDA_SMALL)
            model.load_state_dict(torch.load(weights, map_location="cpu"), strict=True)
            model = model.to("cuda").eval()
            torch.cuda.synchronize()
            result["vda_small_load_s"] = round(time.perf_counter() - started, 1)
            result["vda_small_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
            del model
        except Exception as exc:
            result["vda_error"] = f"{type(exc).__name__}: {exc}"[:400]
        torch.cuda.empty_cache()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--load-models", action="store_true", help="also load SAM 3 and VDA Small on the GPU")
    args = parser.parse_args(argv)
    root = gpu_dir()
    min_free = float(os.getenv("GPU_MIN_FREE_GB", "30"))
    report: dict = {"memory_available_gb": free_memory_gb(), "min_free_gb": min_free, "components": components(root)}

    def done(code: int, error: str | None = None) -> int:
        if error:
            report["error"] = error
        print(json.dumps(report, indent=2))
        return code

    try:
        import torch
    except ImportError as exc:
        return done(1, f"torch not importable: {exc}")
    report["torch"] = torch.__version__
    report["torch_cuda"] = torch.version.cuda
    report["cuda"] = bool(torch.cuda.is_available())
    if not report["cuda"]:
        return done(1, "CUDA is not available to torch")
    available = report["memory_available_gb"]
    if available is not None and available < min_free:
        return done(3, f"only {available} GB of memory available, need {min_free} GB")
    capability = torch.cuda.get_device_capability(0)
    report["gpu"] = torch.cuda.get_device_name(0)
    report["capability"] = f"{capability[0]}.{capability[1]}"
    report["arch_list"] = torch.cuda.get_arch_list()
    report["capability_supported"] = capability_supported(capability, report["arch_list"])
    try:
        report["matmul_4096_ms"], report["matmul_finite"] = matmul_ms(torch)
    except Exception as exc:  # missing kernel image, out of memory, ...
        return done(1, f"matmul failed: {exc}")
    if args.load_models:
        report["models"] = load_models(torch, root, report)
    ok = report["capability_supported"] and report["matmul_finite"]
    return done(0 if ok else 1, None if ok else "torch build cannot run on this GPU")


if __name__ == "__main__":
    sys.exit(main())
