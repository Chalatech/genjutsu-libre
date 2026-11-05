#!/usr/bin/env python3
"""Separate voice and music with Demucs htdemucs on the GPU host.

    python separate.py --in <dir> --out <dir> [--job-id ID]

Reads ``original-audio.wav`` from --in (or the first ``upload.*`` there) and writes
the same files as the hosted ``separate()``: ``original-audio.wav``, one ``.wav``
per stem (``vocals``, ``drums``, ``bass``, ``other``), ``music.wav`` (every stem
but vocals) and ``audio-manifest.json`` with the same keys, plus
``report-separate.json``. Audio I/O uses PyAV and soundfile (not torchaudio).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import common

RATE = 44100


def manifest(job_id: str) -> dict:
    """Same keys and values as audio_workflow.separate(); prediction_id is the local job id."""
    return {
        "prediction_id": job_id,
        "seedance_video": "composite-silent.mp4",
        "seedance_audio": "vocals.wav",
        "final_audio": "upload.mp4 (entire original audio track)",
        "discard_seedance_audio": True,
        "separation": "Demucs htdemucs; residual music or vocal bleed is possible",
    }


def job_id(folder: Path) -> str:
    """Name of the job folder: ``jobs/<id>/in`` -> ``<id>``, ``jobs/<id>`` -> ``<id>``."""
    folder = Path(folder).resolve()
    return folder.parent.name if folder.name in ("in", "input") else folder.name


def source_audio(folder: Path) -> Path:
    wav = folder / "original-audio.wav"
    if wav.is_file():
        return wav
    uploads = sorted(folder.glob("upload.*"))
    if uploads:
        return uploads[0]
    raise FileNotFoundError(f"no original-audio.wav or upload.* in {folder}")


def read_audio(path: Path):
    """Decode any audio/video file to float32 stereo 44.1 kHz, shape (2, samples)."""
    import av
    import numpy as np

    parts = []
    resampler = av.AudioResampler(format="fltp", layout="stereo", rate=RATE)
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError("Source has no audio track.")
        for frame in container.decode(audio=0):
            parts.extend(x.to_ndarray() for x in resampler.resample(frame))
        parts.extend(x.to_ndarray() for x in resampler.resample(None))
    return np.concatenate(parts, axis=1).astype(np.float32)


def write_audio(path: Path, audio) -> None:
    """(2, samples) float -> 16-bit PCM stereo WAV, like audio_workflow.write_audio."""
    import numpy as np
    import soundfile

    soundfile.write(str(path), np.clip(audio, -1, 1).T, RATE, subtype="PCM_16")


def music_mix(stems: dict, length: int):
    """Sum of every stem except vocals, clipped to [-1, 1]."""
    import numpy as np

    music = np.zeros((2, length), dtype=np.float32)
    for name, audio in stems.items():
        if name != "vocals":
            n = min(length, audio.shape[1])
            music[:, :n] += audio[:, :n]
    return np.clip(music, -1, 1)


def work(args, report, torch, root):
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    inp, out = Path(args.inp), Path(args.out)
    src = source_audio(inp)
    with report.stage("read"):
        original = read_audio(src)
        target = out / "original-audio.wav"
        if src.name == "original-audio.wav" and src.resolve() != target.resolve():
            shutil.copyfile(src, target)
        elif src.resolve() != target.resolve():
            write_audio(target, original)
    report["seconds"] = round(original.shape[1] / RATE, 2)
    def load():
        with torch.device("cuda"), common.skip_weight_init(torch):  # weights go straight to the GPU
            loaded = get_model(args.model)
        return loaded.to("cuda").eval()

    with report.stage("load"):
        model, report["model_cached"] = common.cached(args, f"demucs:{args.model}", load)
    if model.samplerate != RATE or model.audio_channels != 2:
        raise RuntimeError(f"unexpected model format {model.samplerate} Hz x {model.audio_channels}")
    with report.stage("separate"):
        wav = torch.from_numpy(original)
        ref = wav.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        with torch.no_grad():
            sources = apply_model(model, ((wav - mean) / std)[None], device="cuda", shifts=args.shifts,
                                  split=True, overlap=0.25, progress=False)[0]
        sources = (sources * std + mean).cpu().numpy()
    with report.stage("write"):
        stems = {name: sources[i] for i, name in enumerate(model.sources)}
        for name, audio in stems.items():
            write_audio(out / f"{name}.wav", audio)
        write_audio(out / "music.wav", music_mix(stems, original.shape[1]))
        (out / "audio-manifest.json").write_text(json.dumps(manifest(args.job_id or job_id(inp)), indent=2))
    report.update(model=args.model, stems=list(model.sources), source=src.name,
                  vocals_rms=round(float((stems["vocals"] ** 2).mean() ** 0.5), 4))


def make_parser():
    parser = common.worker_parser(__doc__.splitlines()[0])
    parser.add_argument("--job-id", help="id recorded as prediction_id (default: name of the job folder)")
    parser.add_argument("--model", default="htdemucs")
    parser.add_argument("--shifts", type=int, default=1, help="random shifts averaged by demucs (more = slower, slightly better)")
    return parser


def main(argv: list[str] | None = None) -> int:
    return common.run_worker("separate", make_parser().parse_args(argv), work)


if __name__ == "__main__":
    sys.exit(main())
