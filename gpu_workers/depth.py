#!/usr/bin/env python3
"""Colored depth with Video-Depth-Anything Small on the GPU host.

    python depth.py --in <dir> --out <dir> [--input-size 518] [--fp32]

Reads ``original.mp4`` (already normalized to 24 fps, long side <= 960) from --in
and writes ``depth-raw.mp4``: same size and frame count, 24 fps, inferno colormap
normalized over the whole clip (near = bright), plus ``report-depth.json``.
Model: Video-Depth-Anything Small (Apache-2.0) from ``$GPU_DIR/src/video-depth-anything``
with weights in ``$GPU_DIR/models/video-depth-anything``. Frames are read with PyAV,
so the upstream ``utils.dc_utils`` (decord) is never imported.
"""
from __future__ import annotations

import sys
from pathlib import Path

import common

VDA_SMALL = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}


def inferno_lut():
    """256x3 uint8 RGB inferno table (the same OpenCV table pipeline.grayscale() inverts)."""
    import cv2
    import numpy as np

    return cv2.applyColorMap(np.arange(256, dtype=np.uint8)[:, None], cv2.COLORMAP_INFERNO)[:, 0, ::-1].copy()


def to_uint8(depths):
    """Normalize relative depth over the whole clip to 0..255 (a constant clip maps to 0)."""
    import numpy as np

    low, high = float(depths.min()), float(depths.max())
    scale = 255.0 / (high - low) if high > low else 0.0
    return ((depths - low) * scale).clip(0, 255).astype(np.uint8)


def colorize(depths, lut=None):
    """(T, H, W) float depth -> generator of (H, W, 3) uint8 inferno frames."""
    lut = inferno_lut() if lut is None else lut
    levels = to_uint8(depths)
    for frame in levels:
        yield lut[frame]


def use_fused_attention(torch) -> bool:
    """Run DINOv2's attention through torch's scaled_dot_product_attention.

    Upstream falls back to softmax(q @ k.T) @ v when xFormers is missing, which
    materializes a 32-frame x heads x tokens^2 matrix (~12 GB of GPU memory at 518 px).
    The fused kernel computes the same thing without that matrix.
    """
    from video_depth_anything.dinov2_layers import attention

    functional = torch.nn.functional
    if getattr(attention, "XFORMERS_AVAILABLE", False) or not hasattr(functional, "scaled_dot_product_attention"):
        return False

    def forward(self, x, attn_bias=None):
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, channels // self.num_heads).permute(2, 0, 3, 1, 4)
        x = functional.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])  # default scale = head_dim ** -0.5
        return self.proj_drop(self.proj(x.transpose(1, 2).reshape(batch, tokens, channels)))

    attention.Attention.forward = forward
    attention.MemEffAttention.forward = forward
    return True


def work(args, report, torch, root):
    import numpy as np

    src = Path(args.inp) / "original.mp4"
    out = Path(args.out) / "depth-raw.mp4"
    vda_root = root / "src" / "video-depth-anything"
    weights = root / "models" / "video-depth-anything" / "video_depth_anything_vits.pth"
    with report.stage("read"):
        frames = np.stack(common.read_frames(src))
    report.update(frames_in=int(frames.shape[0]), size=[int(frames.shape[2]), int(frames.shape[1])])
    with report.stage("load"):
        sys.path.insert(0, str(vda_root))
        from video_depth_anything.video_depth import VideoDepthAnything

        report["fused_attention"] = args.attention == "fused" and use_fused_attention(torch)

        with common.skip_weight_init(torch):
            model = VideoDepthAnything(**VDA_SMALL)
        model.load_state_dict(torch.load(weights, map_location="cpu"), strict=True)
        model = model.to("cuda").eval()
    with report.stage("infer"):
        depths, _ = model.infer_video_depth(frames, 24, input_size=args.input_size, device="cuda", fp32=args.fp32)
    del frames, model
    torch.cuda.empty_cache()
    with report.stage("write"):
        written = common.write_video(out, colorize(depths))
    report.update(model="Video-Depth-Anything Small", input_size=args.input_size, fp32=args.fp32,
                  frames_out=written, depth_min=round(float(depths.min()), 4), depth_max=round(float(depths.max()), 4),
                  depth_std=round(float(depths.std()), 4))
    if written != report["frames_in"]:
        raise RuntimeError(f"wrote {written} depth frames for {report['frames_in']} input frames")
    if report["depth_max"] <= report["depth_min"]:
        raise RuntimeError("depth is constant over the whole clip")


def main(argv: list[str] | None = None) -> int:
    parser = common.worker_parser(__doc__.splitlines()[0])
    parser.add_argument("--input-size", type=int, default=518, help="model input size (multiple of 14)")
    parser.add_argument("--fp32", action="store_true", help="full precision (slower; default uses autocast)")
    parser.add_argument("--attention", choices=["fused", "upstream"], default="fused",
                        help="fused: torch scaled_dot_product_attention (low memory); upstream: the original code")
    return common.run_worker("depth", parser.parse_args(argv), work)


if __name__ == "__main__":
    sys.exit(main())
