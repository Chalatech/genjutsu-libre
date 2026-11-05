import importlib.util, io, json, os, sys, tempfile, types, unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

WORKERS = Path(__file__).resolve().parent.parent / 'gpu_workers'
sys.path.insert(0, str(WORKERS))


def load(name):
    spec = importlib.util.spec_from_file_location(f'gpu_{name}', WORKERS / f'{name}.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


import common  # noqa: E402  (the same module object the workers import)
segment, separate, depth = load('segment'), load('separate'), load('depth')


def write_sequence(folder, masks):
    folder.mkdir(parents=True, exist_ok=True)
    for index, mask in enumerate(masks):
        common.write_mask(folder / common.mask_name(index), mask)


def box(y0, y1, x0, x1, shape=(6, 8)):
    mask = np.zeros(shape, bool); mask[y0:y1, x0:x1] = True
    return mask


class NamingTests(unittest.TestCase):
    def test_mask_names_match_composite(self):
        self.assertEqual(common.mask_name(0), 'mask_00000.png')
        self.assertEqual(common.mask_name(359), 'mask_00359.png')
        # composite() parses int(p.stem.split('_')[-1]) and sorts the names as strings
        names = [common.mask_name(i) for i in (0, 9, 10, 100, 1000)]
        self.assertEqual(names, sorted(names))
        self.assertEqual([int(Path(n).stem.split('_')[-1]) for n in names], [0, 9, 10, 100, 1000])
        with self.assertRaises(ValueError):
            common.mask_name(-1)

    def test_mask_files_reject_gaps(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            for i in (0, 1, 3):
                common.write_mask(folder / common.mask_name(i), box(0, 1, 0, 1))
            with self.assertRaises(ValueError):
                common.mask_files(folder)
            common.write_mask(folder / common.mask_name(2), box(0, 1, 0, 1))
            self.assertEqual([p.name for p in common.mask_files(folder)], [common.mask_name(i) for i in range(4)])

    def test_parse_ids(self):
        self.assertEqual(common.parse_ids('1,3'), [1, 3])
        self.assertEqual(common.parse_ids(' 3, 1,3,'), [1, 3])
        for bad in ('', 'a', '0', '1,-2'):
            with self.assertRaises(ValueError):
                common.parse_ids(bad)


class UnionTests(unittest.TestCase):
    def test_union_of_selected_instances(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_sequence(root / 'instances' / '1', [box(0, 2, 0, 2), box(0, 2, 0, 2), np.zeros((6, 8), bool)])
            write_sequence(root / 'instances' / '2', [box(4, 6, 6, 8), np.zeros((6, 8), bool), np.zeros((6, 8), bool)])
            write_sequence(root / 'instances' / '3', [box(0, 6, 0, 8)] * 3)
            result = common.union_masks(root / 'instances', [1, 2], root / 'raw-masks')
            self.assertEqual(result, {'ids': [1, 2], 'frames': 3, 'nonempty_frames': 2})
            files = sorted((root / 'raw-masks').glob('mask_*.png'))
            self.assertEqual([int(p.stem.split('_')[-1]) for p in files], [0, 1, 2])
            first = np.asarray(Image.open(files[0]).convert('RGB'))
            self.assertEqual(set(np.unique(first)), {0, 255})
            np.testing.assert_array_equal(first[:, :, 0] > 127, box(0, 2, 0, 2) | box(4, 6, 6, 8))
            self.assertFalse(common.read_mask(files[2]).any())
            # re-selecting replaces the previous union
            common.union_masks(root / 'instances', [3], root / 'raw-masks')
            self.assertTrue(common.read_mask(root / 'raw-masks' / 'mask_00002.png').all())

    def test_union_rejects_unknown_ids_and_mismatched_lengths(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_sequence(root / 'instances' / '1', [box(0, 1, 0, 1)] * 3)
            write_sequence(root / 'instances' / '2', [box(0, 1, 0, 1)] * 2)
            with self.assertRaisesRegex(ValueError, 'unknown instance ids'):
                common.union_masks(root / 'instances', [1, 7], root / 'raw-masks')
            with self.assertRaisesRegex(ValueError, 'different'):
                common.union_masks(root / 'instances', [1, 2], root / 'raw-masks')

    def test_select_only_needs_no_gpu(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_sequence(root / 'instances' / '1', [box(0, 2, 0, 2)] * 2)
            write_sequence(root / 'instances' / '2', [box(3, 4, 3, 4)] * 2)
            (root / 'instances.json').write_text('{"instances": []}')
            out = io.StringIO()
            with patch.dict(sys.modules, {'torch': None}), redirect_stdout(out):
                code = segment.main(['--in', d, '--out', d, '--select', '2,1'])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads((root / 'selection.json').read_text())['ids'], [1, 2])
            self.assertEqual(int(common.read_mask(root / 'raw-masks' / 'mask_00001.png').sum()), 5)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(segment.main(['--in', d, '--out', d, '--select', '9']), 1)


class InstanceTests(unittest.TestCase):
    def test_stats_cover_the_span_between_first_and_last_frame(self):
        stats = segment.instance_stats({3: 10, 4: 0, 5: 30, 6: 20}, frame_px=100, scores={3: 0.9, 5: 0.7, 6: 0.8},
                                       centers={3: 42.0})
        self.assertEqual((stats['first_frame'], stats['last_frame'], stats['frames_present']), (3, 6, 3))
        self.assertEqual((stats['coverage'], stats['mean_area'], stats['best_frame']), (0.75, 0.2, 5))
        self.assertEqual((stats['mean_score'], stats['first_center_x']), (0.8, 42.0))
        self.assertEqual(segment.instance_stats({1: 5, 2: 5, 5: 5, 6: 0, 7: 5}, 100)['gaps'], [[3, 4], [6, 6]])
        self.assertEqual(segment.instance_stats({1: 0}, 100), {'frames_present': 0})

    def test_ids_follow_first_appearance_then_left_to_right(self):
        stats = {7: {'first_frame': 50, 'first_center_x': 10}, 2: {'first_frame': 0, 'first_center_x': 700},
                 4: {'first_frame': 0, 'first_center_x': 100}}
        self.assertEqual(segment.assign_ids(stats), {4: 1, 2: 2, 7: 3})

    def test_short_or_tiny_objects_are_dropped(self):
        self.assertIsNone(segment.drop_reason({'frames_present': 20, 'mean_area': 0.05}, 8, 0.0005))
        self.assertIn('fewer', segment.drop_reason({'frames_present': 3, 'mean_area': 0.05}, 8, 0.0005))
        self.assertIn('area', segment.drop_reason({'frames_present': 30, 'mean_area': 0.0001}, 8, 0.0005))

    def test_contact_sheet_renders(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); shape = (54, 96)
            frames = [np.full(shape + (3,), 90, np.uint8) for _ in range(4)]
            write_sequence(root / 'instances' / '1', [box(5, 40, 5, 30, shape)] * 4)
            write_sequence(root / 'instances' / '2', [np.zeros(shape, bool)] * 2 + [box(10, 50, 60, 90, shape)] * 2)
            listing = {'prompt': 'person', 'frames': 4, 'instances': [
                {'id': 1, 'first_frame': 0, 'last_frame': 3, 'best_frame': 0, 'mean_area': 0.17, 'coverage': 1.0},
                {'id': 2, 'first_frame': 2, 'last_frame': 3, 'best_frame': 2, 'mean_area': 0.23, 'coverage': 1.0}]}
            segment.contact_sheet(frames, listing, root / 'instances', root / 'sheet.jpg', card=200, samples=4)
            sheet = np.asarray(Image.open(root / 'sheet.jpg'))
            self.assertEqual(sheet.shape[1], 3 * 200 + 4 * 12)
            self.assertGreater(sheet.shape[0], 2 * 112)


class MemoryGuardTests(unittest.TestCase):
    def test_waits_and_rechecks_until_enough(self):
        readings, clock, naps = iter([12.0, 18.0, 21.5]), iter([0, 0, 120, 240]), []
        ok, available, waited = common.wait_for_memory(20, read=lambda: next(readings), wait_s=1800, poll_s=120,
                                                       sleep=naps.append, clock=lambda: next(clock), log=lambda m: None)
        self.assertEqual((ok, available, waited, naps), (True, 21.5, 240, [120, 120]))

    def test_gives_up_after_the_wait_limit(self):
        now = [0.0]
        def nap(seconds): now[0] += seconds
        ok, available, waited = common.wait_for_memory(20, read=lambda: 9.5, wait_s=1800, poll_s=120,
                                                       sleep=nap, clock=lambda: now[0], log=lambda m: None)
        self.assertEqual((ok, available, waited), (False, 9.5, 1800))

    def test_huge_pages_are_off_before_torch_or_numpy_load_in_a_worker(self):
        for name, value in common.NO_HUGE_PAGES.items():
            self.assertIn(name, os.environ)
        for worker in ('separate', 'depth', 'segment', 'check'):
            lines = [l for l in (WORKERS / f'{worker}.py').read_text().splitlines() if l.startswith(('import ', 'from '))]
            first_third_party = next(l for l in lines if not l.startswith(('from __future__', 'import argparse', 'import importlib',
                                                                        'import json', 'import os', 'import re', 'import shutil',
                                                                        'import sys', 'import time', 'from pathlib')))
            self.assertIn('common', first_third_party, worker)

    def test_unknown_memory_does_not_block(self):
        self.assertTrue(common.wait_for_memory(20, read=lambda: None, wait_s=0, poll_s=120)[0])

    def test_worker_exits_3_with_a_report_and_never_imports_torch(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(sys.modules, {'torch': None}), \
                patch.object(common, 'free_memory_gb', return_value=4.0), patch.object(common, 'lower_priority', return_value=10), \
                patch.dict(os.environ, {'GPU_MIN_FREE_GB': '20', 'GPU_MEM_WAIT_MIN': '0', 'GPU_DIR': d}), \
                redirect_stdout(io.StringIO()):
            code = depth.main(['--in', d, '--out', d])
            report = json.loads((Path(d) / 'report-depth.json').read_text())
        self.assertEqual(code, 3)
        self.assertEqual((report['memory_available_gb'], report['min_free_gb']), (4.0, 20.0))
        self.assertIn('need 20.0 GB', report['error'])


class AudioAndDepthTests(unittest.TestCase):
    def test_manifest_keeps_the_hosted_keys(self):
        data = separate.manifest('job-1')
        self.assertEqual(set(data), {'prediction_id', 'seedance_video', 'seedance_audio', 'final_audio',
                                     'discard_seedance_audio', 'separation'})
        self.assertEqual((data['prediction_id'], data['seedance_audio']), ('job-1', 'vocals.wav'))
        self.assertEqual(separate.job_id(Path('/x/jobs/abc/in')), 'abc')
        self.assertEqual(separate.job_id(Path('/x/jobs/abc')), 'abc')

    def test_music_is_every_stem_but_vocals(self):
        stems = {'vocals': np.full((2, 4), 0.5, np.float32), 'drums': np.full((2, 4), 0.25, np.float32),
                 'bass': np.full((2, 3), 0.9, np.float32)}
        music = separate.music_mix(stems, 4)
        np.testing.assert_allclose(music[:, :3], 1.0); np.testing.assert_allclose(music[:, 3], 0.25)

    def test_depth_colors_span_the_clip(self):
        depths = np.linspace(0, 1, 2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
        levels = depth.to_uint8(depths)
        self.assertEqual((levels.min(), levels.max()), (0, 255))
        self.assertFalse(depth.to_uint8(np.ones((2, 3, 4), np.float32)).any())
        lut = depth.inferno_lut()
        frames = list(depth.colorize(depths, lut))
        self.assertEqual((len(frames), frames[0].shape, frames[0].dtype), (2, (3, 4, 3), np.uint8))
        self.assertLess(int(lut[0].sum()), int(lut[255].sum()))  # far is dark, near is bright


if __name__ == '__main__':
    unittest.main()
