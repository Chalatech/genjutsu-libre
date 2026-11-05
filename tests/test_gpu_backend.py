import json, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

import gpu_backend, raw_masks
from gpu_backend import Gpu, GpuError, GpuMemoryError, GpuUnreachable

MEM = 'MemAvailable:   {} kB\n'


def completed(code=0, out='', err=''):
    return SimpleNamespace(returncode=code, stdout=out, stderr=err)


class FakeProc:
    def __init__(self, lines, code):
        self.stdout = iter(lines);self.code = code
    def wait(self):
        return self.code


class FakeHost:
    """Stands in for ssh/rsync: records command lines; the 'host' is a local temp folder."""
    def __init__(self, root, free_gb=64, code=0, lines=None):
        self.root = Path(root);self.free_gb = free_gb;self.code = code;self.lines = lines;self.calls = [];self.popen_calls = []
    def run(self, cmd, **kw):
        self.calls.append(cmd)
        if cmd[0] == 'rsync':
            src, dst = cmd[-2], cmd[-1]
            if ':' in dst:   # push: files... host:dir/
                target = self.root/dst.split(':', 1)[1]
                target.mkdir(parents=True, exist_ok=True)
                for f in cmd[cmd.index('-e')+2:-1]:(target/Path(f).name).write_bytes(Path(f).read_bytes())
            else:            # pull: host:dir/ local/
                source = self.root/src.split(':', 1)[1]
                import shutil;shutil.copytree(source, dst, dirs_exist_ok=True)
            return completed()
        text = cmd[-1]
        if 'MemAvailable' in text:return completed(0, MEM.format(int(self.free_gb*1024*1024)))
        if text.startswith('mkdir'):
            for part in text.split()[2:]:(self.root/part.strip('"').replace('$HOME/', '').lstrip('/')).mkdir(parents=True, exist_ok=True)
        return completed()
    def popen(self, cmd, **kw):
        self.popen_calls.append(cmd)
        return FakeProc(self.lines if self.lines is not None else ['[separate] 10:00:00 demucs...\n', json.dumps({'worker': 'separate', 'exit': 0})+'\n'], self.code)


def make_gpu(host, **kw):
    return Gpu(host='gpuhost', directory='~/work', python='~/work/.venv/bin/python', status=lambda s: None, sleep=lambda s: None, **kw)


class Patched(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root/'job'/'abc123').mkdir(parents=True)
        self.folder = self.root/'job'/'abc123'
    def host(self, **kw):
        h = FakeHost(self.root/'remote', **kw)
        for target, fn in (('subprocess.run', h.run), ('subprocess.Popen', h.popen)):
            p = patch('gpu_backend.'+target, side_effect=fn);p.start();self.addCleanup(p.stop)
        return h


class CommandLines(Patched):
    def test_worker_command_line(self):
        h = self.host();g = make_gpu(h)
        summary = g.run('separate', 'abc123', ['--job-id', 'x y'], self.folder)
        cmd = h.popen_calls[0]
        self.assertEqual(cmd[:5], ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10'])
        remote = cmd[-1]
        for part in ('nice -n 10', 'timeout -k 30 900', 'workers/separate.py', '"$HOME"/work/jobs/abc123/in', '"$HOME"/work/jobs/abc123/out', "'x y'"):
            self.assertIn(part, remote)
        self.assertEqual(summary['exit'], 0)
        self.assertIn('demucs', (self.folder/'gpu-logs'/'separate.log').read_text())

    def test_stage_pushes_pulls_and_always_cleans(self):
        h = self.host();g = make_gpu(h)
        (self.folder/'original-audio.wav').write_bytes(b'wav')
        out = h.root/'~/work/jobs/abc123/out'
        with patch.object(Gpu, 'run', side_effect=lambda *a, **k: (out.mkdir(parents=True, exist_ok=True), (out/'report-separate.json').write_text('{}'), {'exit': 0})[2]):
            g.stage('separate', self.folder, ['original-audio.wav'])
        self.assertTrue((self.folder/'gpu-reports'/'report-separate.json').exists())
        self.assertIn('rm -rf', h.calls[-1][-1]);self.assertIn('jobs/abc123', h.calls[-1][-1])

    def test_cleanup_even_when_worker_fails(self):
        h = self.host(code=1, lines=[json.dumps({'worker': 'depth', 'exit': 1, 'error': 'RuntimeError: boom'})+'\n']);g = make_gpu(h)
        (self.folder/'original.mp4').write_bytes(b'v')
        with self.assertRaises(GpuError) as ctx:g.stage('depth', self.folder, ['original.mp4'])
        self.assertIn('boom', str(ctx.exception))
        self.assertIn('rm -rf', h.calls[-1][-1])

    def test_unsafe_job_names_rejected(self):
        g = make_gpu(self.host())
        for bad in ('../x', 'a b', '', 'a;rm'):
            with self.assertRaises(ValueError):g.job_dir(bad)


class Failures(Patched):
    def test_low_memory_rejected_without_running_anything(self):
        h = self.host(free_gb=3);g = make_gpu(h)
        ticks = iter(range(0, 100000, 130))
        g.clock = lambda: next(ticks)
        with patch.dict('os.environ', {'GPU_MEM_WAIT_MIN': '10', 'GPU_MEM_POLL_S': '120', 'GPU_MIN_FREE_GB': '20'}):
            with self.assertRaises(GpuMemoryError) as ctx:g.run('segment', 'abc123')
        self.assertIn('3.0 GB', str(ctx.exception).replace('3 GB', '3.0 GB'))
        self.assertEqual(h.popen_calls, [])

    def test_memory_recovers_while_waiting(self):
        h = self.host(free_gb=3);g = make_gpu(h)
        def sleeper(s):h.free_gb = 40
        g.sleep = sleeper
        with patch.dict('os.environ', {'GPU_MEM_WAIT_MIN': '10', 'GPU_MEM_POLL_S': '120'}):
            g.run('separate', 'abc123')
        self.assertEqual(len(h.popen_calls), 1)

    def test_exit_codes(self):
        cases = {3: GpuMemoryError, 255: GpuUnreachable, 124: GpuError, 1: GpuError}
        for code, exc in cases.items():
            g = make_gpu(self.host(code=code, lines=['oops\n']))
            with self.assertRaises(exc):g.run('depth', 'abc123')
        g = make_gpu(self.host(code=124, lines=[]))
        with self.assertRaisesRegex(GpuError, 'limit'):g.run('depth', 'abc123')

    def test_unreachable_ssh(self):
        with patch('gpu_backend.subprocess.run', return_value=completed(255, '', 'ssh: Could not resolve hostname\n')):
            with self.assertRaises(GpuUnreachable):make_gpu(None).free_gb()


class Routing(unittest.TestCase):
    def test_enabled(self):
        with patch('gpu_backend.setting', return_value='gpu'):
            self.assertTrue(gpu_backend.enabled());self.assertFalse(gpu_backend.enabled('replicate'));self.assertTrue(gpu_backend.enabled('gpu'))
        with patch('gpu_backend.setting', return_value='replicate'):
            self.assertFalse(gpu_backend.enabled());self.assertTrue(gpu_backend.enabled('gpu'))

    def test_remote_path_keeps_home_expandable(self):
        self.assertEqual(gpu_backend.remote_path('~/a b'), '"$HOME"/\'a b\'')
        self.assertEqual(gpu_backend.remote_path('/srv/x'), '/srv/x')

    def test_requires_configuration(self):
        with patch('gpu_backend.setting', return_value=''):
            with self.assertRaises(ValueError):Gpu()

    def test_replicate_path_does_not_touch_gpu_backend(self):
        import audio_workflow
        with tempfile.TemporaryDirectory() as d, patch('gpu_backend.setting', return_value='replicate'), patch('audio_workflow.read_audio', return_value=np.zeros((2, 10), np.float32)), \
                patch('audio_workflow.write_audio'), patch.dict('os.environ', {'REPLICATE_API_TOKEN': ''}), patch('gpu_backend.Gpu') as gpu:
            with self.assertRaisesRegex(ValueError, 'REPLICATE_API_TOKEN'):audio_workflow.separate('x.mp4', d, lambda s: None)
            gpu.assert_not_called()

    def test_separate_routes_to_gpu(self):
        import audio_workflow
        with tempfile.TemporaryDirectory() as d, patch('audio_workflow.read_audio', return_value=np.zeros((2, 10), np.float32)), patch('audio_workflow.write_audio'), \
                patch('gpu_backend.separate') as remote, patch.dict('os.environ', {'REPLICATE_API_TOKEN': ''}):
            audio_workflow.separate('x.mp4', d, lambda s: None, 'gpu')
            remote.assert_called_once()


def write_instance(root, ident, frames):
    d = Path(root)/'instances'/str(ident);d.mkdir(parents=True, exist_ok=True)
    for i, box in enumerate(frames):
        a = np.zeros((8, 12), np.uint8)
        if box:a[box[0]:box[1], box[2]:box[3]] = 255
        Image.fromarray(a).save(d/f'mask_{i:05d}.png')


class Instances(unittest.TestCase):
    def test_union_of_selected_instances(self):
        with tempfile.TemporaryDirectory() as d:
            write_instance(d, 1, [(0, 2, 0, 2), None, None])
            write_instance(d, 2, [None, (4, 6, 4, 6), None])
            write_instance(d, 3, [(0, 8, 0, 12)] * 3)   # e.g. a mural face: selected only on request
            out = raw_masks.union_instances(d, '1,2')
            m = [np.asarray(Image.open(out/f'mask_{i:05d}.png')) for i in range(3)]
            self.assertEqual([int((x > 127).sum()) for x in m], [4, 4, 0])
            self.assertEqual(json.loads((Path(d)/'instance-selection.json').read_text())['ids'], [1, 2])
            self.assertEqual(sorted(json.loads((Path(d)/'instance-selection.json').read_text())['available']), [1, 2, 3])
            raw_masks.union_instances(d)   # default: everything
            self.assertEqual(int((np.asarray(Image.open(out/'mask_00002.png')) > 127).sum()), 96)

    def test_unknown_or_bad_ids(self):
        with tempfile.TemporaryDirectory() as d:
            write_instance(d, 1, [(0, 2, 0, 2)])
            with self.assertRaisesRegex(ValueError, 'Unknown'):raw_masks.union_instances(d, '1,9')
            for bad in ('', 'a', '0', '-1'):
                with self.assertRaises(ValueError):raw_masks.parse_ids(bad)

    def test_reselect_clears_review_and_updates_manifest(self):
        import mask_review
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            write_instance(d, 1, [(0, 2, 0, 2)]);write_instance(d, 2, [(4, 6, 4, 6)])
            for name in mask_review.FILES:(p/name).write_bytes(b'x')
            mask_review.review(p, True, 'ok', mask_review.fingerprint(p))
            (p/'manifest.json').write_text('{"instances":[1]}')
            def fake_composite(*a, **k):(p/'mask.mp4').write_bytes(b'new mask')
            with patch('pipeline.composite', side_effect=fake_composite), patch('audio_workflow.prepare_seedance_video'):
                self.assertEqual(raw_masks.reselect(p, '1,2', lambda s: None), [1, 2])
            self.assertFalse(mask_review.state(p).get('approved'))
            self.assertFalse((p/'mask-review.json').exists())
            self.assertEqual(json.loads((p/'manifest.json').read_text())['instances'], [1, 2])


class Doctor(unittest.TestCase):
    def check(self, code, out='', setting_value=None):
        import doctor
        values = {'GPU_HOST': 'h', 'GPU_DIR': '~/d', 'GPU_PYTHON': ''}
        with patch('doctor.setting', side_effect=lambda k, d='': values.get(k, d)), patch('subprocess.run', return_value=completed(code, out)):
            return doctor.gpu_checks()[0]

    def test_ready(self):
        c = self.check(0, json.dumps({'gpu': 'GB10', 'memory_available_gb': 40}));self.assertTrue(c.ok);self.assertNotIn('/', c.message.replace('GPU host:', ''))
    def test_low_memory_is_not_fatal(self):
        c = self.check(3, json.dumps({'memory_available_gb': 5, 'min_free_gb': 20}));self.assertFalse(c.ok);self.assertFalse(c.fatal)
    def test_unreachable_is_fatal(self):
        c = self.check(255);self.assertTrue(c.fatal)
    def test_no_replicate_token_needed(self):
        import doctor
        with patch('doctor.setting', side_effect=lambda k, d='': 'gpu' if k == 'PREP_BACKEND' else ''), patch('doctor.gpu_checks', return_value=[]):
            self.assertEqual([c.name for c in doctor.prep_checks()], [])


if __name__ == '__main__':
    unittest.main()
