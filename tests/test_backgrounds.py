import json, os, tempfile, unittest
from pathlib import Path
from unittest import mock

import av
import numpy as np
from PIL import Image

import pipeline
import audio_workflow


def _video(path, frames):
    c, s = pipeline.writer(path, frames[0].shape[1], frames[0].shape[0])
    for i, a in enumerate(frames):
        pipeline.put(c, s, a, i)
    pipeline.finish(c, s)


def _frames(path):
    return list(pipeline.frames(path))


class CompositeBackgroundTest(unittest.TestCase):
    def _run(self, background):
        d = Path(tempfile.mkdtemp())
        h, w = 64, 64
        rgb = [np.full((h, w, 3), (10, 200, 30), np.uint8) for _ in range(3)]
        depth = [np.full((h, w, 3), (220, 40, 160), np.uint8) for _ in range(3)]
        _video(d/'original.mp4', rgb)
        _video(d/'depth.mp4', depth)
        masks = d/'raw-masks'; masks.mkdir()
        for i in range(3):
            m = np.zeros((h, w), np.uint8); m[16:48, 16:48] = 255
            Image.fromarray(m).save(masks/f'mask_{i:05d}.png')
        with mock.patch.dict(os.environ, {'COMPOSITE_BACKGROUND': background}):
            pipeline.composite(d/'original.mp4', d/'depth.mp4', masks, d, d/'original.mp4')
        return _frames(d/'composite-silent.mp4')[1]

    def test_original_keeps_source_pixels_outside_mask(self):
        f = self._run('original')
        self.assertLess(np.abs(f[2, 2].astype(int) - (10, 200, 30)).max(), 12)

    def test_plain_has_no_source_pixels(self):
        f = self._run('plain')
        self.assertLess(np.abs(f[2, 2].astype(int) - 128).max(), 12)

    def test_depth_fills_the_frame_with_depth(self):
        f = self._run('depth')
        self.assertLess(np.abs(f[2, 2].astype(int) - (220, 40, 160)).max(), 14)

    def test_invalid_value_rejected(self):
        with self.assertRaises(ValueError):
            self._run('blur')


class GuideTrackTest(unittest.TestCase):
    def test_guide_track_replaces_source_vocals(self):
        d = Path(tempfile.mkdtemp())
        _video(d/'composite-silent.mp4', [np.zeros((32, 32, 3), np.uint8) for _ in range(24)])
        rate = audio_workflow.RATE
        audio_workflow.write_audio(d/'vocals.wav', np.zeros((2, rate), np.float32))
        guide = d/'mi-voz.wav'
        audio_workflow.write_audio(guide, (0.2*np.sin(np.linspace(0, 2000, rate))).astype(np.float32)[None].repeat(2, 0))
        env = {'SEEDANCE_EMBED_VOCALS': 'true', 'SEEDANCE_GUIDE_AUDIO': str(guide), 'SEEDANCE_GUIDE_SEMITONES': '0'}
        with mock.patch.dict(os.environ, env):
            out = audio_workflow.prepare_seedance_video(d)
        m = json.loads((d/'audio-manifest.json').read_text())
        self.assertTrue(m['guide_track'])
        self.assertEqual(m['embedded_vocals'], 'guide-plus0.wav')
        with av.open(str(out)) as c:
            self.assertTrue(c.streams.audio)
        embedded = audio_workflow.read_audio(out)
        self.assertGreater(np.abs(embedded).max(), 0.05)   # the guide, not the silent source vocals

    def test_missing_guide_is_an_error(self):
        d = Path(tempfile.mkdtemp())
        _video(d/'composite-silent.mp4', [np.zeros((32, 32, 3), np.uint8) for _ in range(4)])
        with mock.patch.dict(os.environ, {'SEEDANCE_EMBED_VOCALS': 'true', 'SEEDANCE_GUIDE_AUDIO': str(d/'nope.wav')}):
            with self.assertRaises(ValueError):
                audio_workflow.prepare_seedance_video(d)


if __name__ == '__main__':
    unittest.main()
