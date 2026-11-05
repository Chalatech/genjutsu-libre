#!/usr/bin/env python3
"""Track every person in a clip with SAM 3 on the GPU host, one mask sequence per instance.

    python segment.py --in <dir> --out <dir> [--prompt person] [--video original.mp4]
    python segment.py --in <dir> --out <dir> --select 1,3   # union of the chosen ids

Writes to --out:
  instances/<id>/mask_00000.png...  one complete sequence per instance (blank where absent)
  instances.json                    id, first/last frame, frames present, coverage, mean area
  instances-contact.jpg             one card per id (its largest frame) + a timeline of samples
  raw-masks/mask_00000.png...       only with --select: union of the chosen ids, ready for composite()
  report-segment.json               times and peak memory

Ids are 1-based, ordered by first appearance (then left to right). SAM 3's detector
runs on every frame, so people entering late get their own ids. With --select and an
existing instances.json in --out, nothing runs on the GPU: only the union is written.
SEGMENT_MODEL=sam2 (SAM 2.1 + a person detector) is not implemented in this fork.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import common

# 16 high-contrast colors (RGB); id n uses PALETTE[(n - 1) % 16].
PALETTE = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
           (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
           (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195)]


def color(ident: int) -> tuple[int, int, int]:
    return PALETTE[(ident - 1) % len(PALETTE)]


# ---- pure bookkeeping (tested on machines without a GPU) ---------------------

def instance_stats(areas: dict[int, int], frame_px: int, scores: dict[int, float] | None = None,
                   centers: dict[int, float] | None = None) -> dict:
    """Summary of one tracked object from its per-frame mask areas (frames with area 0 are absent)."""
    present = sorted(f for f, a in areas.items() if a > 0)
    if not present:
        return {"frames_present": 0}
    first, last = present[0], present[-1]
    best = max(present, key=lambda f: (areas[f], -f))
    gaps = [[a + 1, b - 1] for a, b in zip(present, present[1:]) if b - a > 1]
    stats = {
        "first_frame": first,
        "last_frame": last,
        "frames_present": len(present),
        "coverage": round(len(present) / (last - first + 1), 4),  # exits and occlusions count as gaps
        "gaps": gaps,
        "mean_area": round(sum(areas[f] for f in present) / len(present) / frame_px, 5),
        "best_frame": best,
    }
    if scores:
        values = [scores[f] for f in present if f in scores]
        stats["mean_score"] = round(sum(values) / len(values), 3) if values else None
    if centers and first in centers:
        stats["first_center_x"] = round(centers[first], 1)
    return stats


def drop_reason(stats: dict, min_frames: int, min_area: float) -> str | None:
    if stats.get("frames_present", 0) < max(1, min_frames):
        return f"present in fewer than {min_frames} frames"
    if stats["mean_area"] < min_area:
        return f"mean area below {min_area:.4%} of the frame"
    return None


def assign_ids(stats_by_object: dict) -> dict:
    """SAM object id -> 1-based instance id, by first appearance then left to right."""
    order = sorted(stats_by_object, key=lambda k: (stats_by_object[k]["first_frame"],
                                                   stats_by_object[k].get("first_center_x", 0.0), k))
    return {obj: index + 1 for index, obj in enumerate(order)}


def select(out: Path, ids: list[int]) -> dict:
    result = common.union_masks(out / "instances", ids, out / "raw-masks")
    (out / "selection.json").write_text(json.dumps(result, indent=2))
    return result


# ---- SAM 3 -------------------------------------------------------------------

def load_sam3(weights: str, torch):
    """(model, processor) of SAM 3 on the GPU."""
    from transformers import Sam3VideoModel, Sam3VideoProcessor

    try:  # straight to the GPU: no 3 GB CPU copy on a host short of memory
        model = Sam3VideoModel.from_pretrained(weights, dtype=torch.bfloat16, device_map="cuda")
    except (ImportError, ValueError):  # older transformers without accelerate
        model = Sam3VideoModel.from_pretrained(weights, dtype=torch.bfloat16).to("cuda")
    model.eval()
    return model, Sam3VideoProcessor.from_pretrained(weights)


def sam3_frames(frames, prompt: str, weights: str, torch, report, chunk: int = 16, args=None):
    """Yield (frame_idx, object_ids, masks (N, H, W) bool, scores) for every frame."""
    import numpy as np

    with report.stage("load"):
        (model, processor), report["model_cached"] = common.cached(args, f"sam3:{weights}", lambda: load_sam3(weights, torch))
    with report.stage("preprocess"):
        session = processor.init_video_session(inference_device="cuda", processing_device="cpu",
                                               video_storage_device="cpu", dtype=torch.bfloat16)
        session.video_height, session.video_width = (int(x) for x in frames[0].shape[:2])
        for start in range(0, len(frames), chunk):  # chunked: never holds the whole clip in float32
            pixels = processor.video_processor(videos=np.stack(frames[start:start + chunk]), device="cpu",
                                               return_tensors="pt").pixel_values_videos[0]
            for offset, value in enumerate(pixels):
                session.add_new_frame(value, start + offset)
        processor.add_text_prompt(inference_session=session, text=prompt)
    for output in model.propagate_in_video_iterator(inference_session=session):
        post = processor.postprocess_outputs(session, output)
        yield (int(output.frame_idx), [int(i) for i in post["object_ids"].tolist()],
               post["masks"].cpu().numpy().astype(bool), [float(s) for s in post["scores"].float().tolist()])


def work(args, report, torch, root):
    import numpy as np

    inp, out = Path(args.inp), Path(args.out)
    model = (args.model or "sam3").lower()
    if model != "sam3":
        raise NotImplementedError(f"SEGMENT_MODEL={model} is not implemented; use sam3")
    weights = os.getenv("SAM3_WEIGHTS") or str(root / "models" / "sam3")
    with report.stage("read"):
        frames = common.read_frames(inp / args.video)
    total, (height, width) = len(frames), frames[0].shape[:2]
    report.update(model="SAM 3 (transformers Sam3VideoModel, bf16)", prompt=args.prompt, video=args.video,
                  frames=total, size=[int(width), int(height)])
    instances = out / "instances"
    if instances.exists():
        shutil.rmtree(instances)
    staging = instances / ".sam"
    staging.mkdir(parents=True)
    areas: dict[int, dict[int, int]] = {}
    scores: dict[int, dict[int, float]] = {}
    centers: dict[int, dict[int, float]] = {}
    seen = set()
    with report.stage("track"):
        for frame_idx, ids, masks, frame_scores in sam3_frames(frames, args.prompt, weights, torch, report, args=args):
            seen.add(frame_idx)
            if len(seen) % 48 == 0:
                print(f"[segment] tracked {len(seen)}/{total} frames, {len(areas)} objects so far", file=sys.stderr, flush=True)
            for obj, mask, score in zip(ids, masks, frame_scores):
                area = int(mask.sum())
                if not area:
                    continue
                folder = staging / str(obj)
                folder.mkdir(exist_ok=True)
                common.write_mask(folder / common.mask_name(frame_idx), mask)
                areas.setdefault(obj, {})[frame_idx] = area
                scores.setdefault(obj, {})[frame_idx] = score
                centers.setdefault(obj, {})[frame_idx] = float(np.nonzero(mask.any(axis=0))[0].mean())
    if len(seen) != total:
        raise RuntimeError(f"SAM 3 returned {len(seen)} of {total} frames")
    with report.stage("write"):
        stats = {obj: instance_stats(areas[obj], width * height, scores.get(obj), centers.get(obj)) for obj in areas}
        dropped = []
        for obj in list(stats):
            reason = drop_reason(stats[obj], args.min_frames, args.min_area)
            if reason:
                dropped.append({"sam_id": obj, **stats.pop(obj), "reason": reason})
        mapping = assign_ids(stats)
        blank = None
        rows = []
        for obj, ident in sorted(mapping.items(), key=lambda item: item[1]):
            folder = instances / str(ident)
            (staging / str(obj)).rename(folder)
            for index in range(total):
                path = folder / common.mask_name(index)
                if not path.exists():
                    if blank is None:
                        common.write_mask(path, np.zeros((height, width), bool))
                        blank = path.read_bytes()
                    else:
                        path.write_bytes(blank)
            rows.append({"id": ident, "sam_id": obj, **stats[obj]})
        shutil.rmtree(staging)
        listing = {"video": args.video, "prompt": args.prompt, "model": model, "frames": total,
                   "width": int(width), "height": int(height), "instances": rows, "dropped": dropped}
        (out / "instances.json").write_text(json.dumps(listing, indent=2))
    with report.stage("contact"):
        contact_sheet(frames, listing, instances, out / "instances-contact.jpg")
    report.update(instances=len(rows), dropped=len(dropped),
                  coverage_min=min((r["coverage"] for r in rows), default=None))
    if args.select:
        report["selection"] = select(out, common.parse_ids(args.select))


# ---- contact sheet -----------------------------------------------------------

def _overlay(frame, masks: dict, focus: int | None = None):
    """Tint each mask with its id color and outline it; with ``focus``, dim everything else."""
    import cv2
    import numpy as np

    image = frame.astype(np.float32)
    if focus is not None and focus in masks:
        image[~masks[focus]] *= 0.3
    for ident, mask in masks.items():
        alpha = 0.55 if focus in (None, ident) else 0.2
        image[mask] = image[mask] * (1 - alpha) + np.array(color(ident), np.float32) * alpha
    image = image.clip(0, 255).astype(np.uint8)
    for ident, mask in masks.items():
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, color(ident), 2 if focus in (None, ident) else 1)
    return image


def _label(image, text: str, x: int, y: int, fill, scale: float = 0.7):
    import cv2

    font, thick = cv2.FONT_HERSHEY_DUPLEX, 2 if scale >= 0.7 else 1
    (w, h), base = cv2.getTextSize(text, font, scale, thick)
    x = max(0, min(int(x), image.shape[1] - w - 8))
    y = max(h + 8, min(int(y), image.shape[0] - 2))
    cv2.rectangle(image, (x, y - h - 8), (x + w + 8, y + base - 2), fill, -1)
    ink = (0, 0, 0) if sum(fill) > 380 else (255, 255, 255)
    cv2.putText(image, text, (x + 4, y - 4), font, scale, ink, thick, cv2.LINE_AA)


def _thumb(frame, masks: dict, width: int, focus: int | None = None, big: bool = False):
    import cv2
    import numpy as np

    image = _overlay(frame, masks, focus)
    scale = width / image.shape[1]
    image = cv2.resize(image, (width, round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    for ident, mask in masks.items():
        if focus is not None and ident != focus:
            continue
        ys, xs = np.nonzero(mask)
        if len(xs):
            _label(image, f"#{ident}", xs.mean() * scale - 14, ys.min() * scale - 2, color(ident),
                   1.0 if big else 0.6)
    return image


def contact_sheet(frames, listing: dict, instances: Path, path: Path, columns: int = 3, card: int = 460,
                  samples: int = 8) -> None:
    """Cards (one per id, at its largest frame, others dimmed) followed by a timeline of sampled frames."""
    import cv2
    import numpy as np

    rows = listing["instances"]
    total = listing["frames"]
    gap, caption, background = 12, 34, (24, 24, 24)
    width = columns * card + (columns + 1) * gap

    def masks_at(index: int, only: int | None = None) -> dict:
        found = {}
        for row in rows:
            if only is not None and row["id"] != only:
                continue
            if row["first_frame"] <= index <= row["last_frame"]:
                mask = common.read_mask(instances / str(row["id"]) / common.mask_name(index))
                if mask.any():
                    found[row["id"]] = mask
        return found

    def text_band(text: str, height: int = 44, scale: float = 0.8):
        band = np.full((height, width, 3), background, np.uint8)
        cv2.putText(band, text, (gap, height - 14), cv2.FONT_HERSHEY_DUPLEX, scale, (235, 235, 235), 1, cv2.LINE_AA)
        return band

    def grid(tiles):
        out = []
        for start in range(0, len(tiles), columns):
            row_tiles = tiles[start:start + columns]
            height = max(t.shape[0] for t in row_tiles)
            band = np.full((height + gap, width, 3), background, np.uint8)
            for col, tile in enumerate(row_tiles):
                x = gap + col * (card + gap)
                band[:tile.shape[0], x:x + tile.shape[1]] = tile
            out.append(band)
        return out

    def with_caption(image, text: str, fill):
        tile = np.full((image.shape[0] + caption, image.shape[1], 3), (40, 40, 40), np.uint8)
        tile[:image.shape[0]] = image
        cv2.rectangle(tile, (0, image.shape[0]), (10, tile.shape[0]), fill, -1)
        cv2.putText(tile, text, (18, tile.shape[0] - 11), cv2.FONT_HERSHEY_DUPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
        return tile

    parts = [text_band(f"{len(rows)} instances of '{listing['prompt']}' in {total} frames ({total / 24:.1f} s at 24 fps)"
                       f" - choose ids for:  segment.py --select 1,2", 52, 0.75)]
    cards = []
    for row in rows:
        frame = row["best_frame"]
        image = _thumb(frames[frame], masks_at(frame), card, focus=row["id"], big=True)
        text = (f"#{row['id']}  f{row['first_frame']}-{row['last_frame']} ({row['first_frame'] / 24:.1f}-"
                f"{row['last_frame'] / 24:.1f} s)  area {row['mean_area']:.1%}  cov {row['coverage']:.0%}")
        cards.append(with_caption(image, text, color(row["id"])))
    if cards:
        parts.append(text_band("Instances: each at its largest frame (other people dimmed)", 40, 0.6))
        parts.extend(grid(cards))
    timeline = []
    for index in sorted({int(round(x)) for x in np.linspace(0, total - 1, samples)}):
        found = masks_at(index)
        timeline.append(with_caption(_thumb(frames[index], found, card), f"frame {index} ({index / 24:.1f} s)  ids "
                                     + (",".join(str(i) for i in sorted(found)) or "-"), (90, 90, 90)))
    parts.append(text_band("Timeline: all ids on evenly spaced frames", 40, 0.6))
    parts.extend(grid(timeline))
    sheet = np.concatenate(parts, axis=0)
    cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])


def make_parser():
    parser = common.worker_parser(__doc__.splitlines()[0])
    parser.add_argument("--prompt", default="person", help="SAM 3 text prompt")
    parser.add_argument("--video", default="original.mp4", help="video inside --in to segment")
    parser.add_argument("--model", default=os.getenv("SEGMENT_MODEL", "sam3"), help="sam3 (sam2 is not implemented)")
    parser.add_argument("--select", help="comma-separated instance ids to merge into raw-masks/")
    parser.add_argument("--min-frames", type=int, default=8, help="drop objects present in fewer frames")
    parser.add_argument("--min-area", type=float, default=0.0005, help="drop objects with a smaller mean area (fraction of the frame)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    out = Path(args.out)
    if args.select and (out / "instances.json").is_file():
        try:  # selection only: no model, no GPU
            result = select(out, common.parse_ids(args.select))
        except ValueError as exc:
            print(json.dumps({"worker": "segment", "exit": common.EXIT_ERROR, "error": str(exc)}), flush=True)
            return common.EXIT_ERROR
        print(json.dumps({"worker": "segment", "exit": common.EXIT_OK, **result}), flush=True)
        return common.EXIT_OK
    return common.run_worker("segment", args, work)


if __name__ == "__main__":
    sys.exit(main())
