import io, json, tempfile, unittest, zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

import gpu_backend, raw_masks
from gpu_api import GpuApiError
from gpu_backend import Gpu, GpuError, GpuMemoryError, GpuUnreachable


class FakeApi:
    """Stands in for GpuApi: records calls; 'downloads' write the canned outputs into the folder."""
    def __init__(self, outputs=None, error=None):
        self.outputs = outputs or {};self.error = error;self.calls = [];self.deleted = [];self.last_metrics = {'stages': {'infer': 1.5}}
    def upload(self, path, name=None):
        self.calls.append(('upload', Path(path).name, name));return 'f-in'
    def predict(self, model, data, timeout=0, on_status=None):
        self.calls.append(('predict', model, data, timeout))
        if on_status:on_status('processing')
        if self.error:raise self.error
        return {n: 'f-'+n for n in self.outputs}
    def download_all(self, output, folder):
        self.calls.append(('download', sorted(output)))
        for name in output:
            target = Path(folder)/name;target.parent.mkdir(parents=True, exist_ok=True)
            content = self.outputs[name]
            if isinstance(content, dict):   # a folder (the real client unzips it)
                for rel, data in content.items():(target/rel).parent.mkdir(parents=True, exist_ok=True);(target/rel).write_bytes(data)
            else:target.write_bytes(content)
        return {}
    def delete(self, file_id):
        self.deleted.append(file_id)


class Stages(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)/'abc123';self.folder.mkdir()
        (self.folder/'original-audio.wav').write_bytes(b'wav');(self.folder/'original.mp4').write_bytes(b'mp4')
    def gpu(self, **kw):
        self.api = FakeApi(**kw);return Gpu(api=self.api, status=lambda s: None)

    def test_separate_uploads_audio_downloads_stems_and_cleans(self):
        gpu = self.gpu(outputs={'vocals.wav': b'v', 'music.wav': b'm', 'drums.wav': b'd', 'original-audio.wav': b'x',
                                'audio-manifest.json': b'{"prediction_id": "abc123"}', 'report-separate.json': b'{}'})
        gpu_backend.separate(self.folder, lambda s: None, gpu)
        self.assertEqual(self.api.calls[0], ('upload', 'original-audio.wav', None))
        self.assertEqual(self.api.calls[1][:3], ('predict', 'separate', {'audio': 'f-in', 'job_id': 'abc123'}))
        for n in ('vocals.wav', 'music.wav', 'drums.wav', 'audio-manifest.json'):self.assertTrue((self.folder/n).exists(), n)
        self.assertEqual((self.folder/'original-audio.wav').read_bytes(), b'wav')   # the local one is not overwritten
        self.assertFalse((self.folder/'gpu-out').exists())
        self.assertTrue((self.folder/'gpu-reports'/'report-separate.json').exists())
        self.assertEqual(json.loads((self.folder/'gpu-reports'/'api-separate.json').read_text())['stages']['infer'], 1.5)

    def test_depth_keeps_file_name(self):
        gpu = self.gpu(outputs={'depth-raw.mp4': b'depth'})
        gpu_backend.depth(self.folder, lambda s: None, gpu)
        self.assertEqual((self.folder/'depth-raw.mp4').read_bytes(), b'depth')
        self.assertEqual(self.api.calls[1][2], {'video': 'f-in'})

    def test_segment_extracts_instances_and_listing(self):
        listing = json.dumps({'instances': [{'id': 1}]}).encode()
        gpu = self.gpu(outputs={'instances': {'1/mask_00000.png': b'png'}, 'instances.json': listing, 'instances-contact.jpg': b'jpg'})
        (self.folder/'instances').mkdir();(self.folder/'instances'/'stale').write_bytes(b'old')
        result = gpu_backend.segment(self.folder/'original.mp4', self.folder, 'person', lambda s: None, gpu)
        self.assertEqual(result['instances'][0]['id'], 1)
        self.assertTrue((self.folder/'instances'/'1'/'mask_00000.png').exists());self.assertFalse((self.folder/'instances'/'stale').exists())
        self.assertEqual(self.api.calls[0], ('upload', 'original.mp4', 'original.mp4'))
        self.assertEqual(self.api.calls[1][2], {'video': 'f-in', 'prompt': 'person'})

    def test_missing_output_is_an_error(self):
        gpu = self.gpu(outputs={'other.mp4': b'x'})
        with self.assertRaisesRegex(GpuError, 'depth-raw.mp4 missing'):gpu_backend.depth(self.folder, lambda s: None, gpu)
        self.assertFalse((self.folder/'gpu-out').exists())

    def test_errors_are_translated_and_the_upload_is_deleted(self):
        for code, kind in (('low_memory', GpuMemoryError), ('server_down', GpuUnreachable), ('worker_error', GpuError), ('timeout', GpuError)):
            gpu = self.gpu(error=GpuApiError(code, 'boom'))
            with self.assertRaises(kind) as ctx:gpu_backend.depth(self.folder, lambda s: None, gpu)
            self.assertIn('boom', str(ctx.exception));self.assertEqual(self.api.deleted, ['f-in'])
        self.assertNotIsInstance(ctx.exception, (GpuMemoryError, GpuUnreachable))

    def test_failed_download_deletes_the_server_outputs(self):
        gpu = self.gpu(outputs={'depth-raw.mp4': b'd'})
        self.api.download_all = MagicMock(side_effect=GpuApiError('http_404', 'gone'))
        with self.assertRaises(GpuError):gpu_backend.depth(self.folder, lambda s: None, gpu)
        self.assertEqual(self.api.deleted, ['f-depth-raw.mp4'])

    def test_timeout_includes_the_memory_wait(self):
        with patch('gpu_backend.setting', return_value='30'):self.assertEqual(Gpu(api=FakeApi()).timeout('depth'), 1800+1800)


class Session(unittest.TestCase):
    def fake_gpu(self):
        api = MagicMock();api.tunnel.return_value.__enter__ = MagicMock();api.tunnel.return_value.__exit__ = MagicMock(return_value=False)
        return Gpu(api=api, status=lambda s: None), api

    def test_one_server_check_and_one_tunnel_for_nested_stages(self):
        gpu, api = self.fake_gpu()
        with patch('gpu_backend.Gpu', return_value=gpu), patch('gpu_backend.server_env', return_value={'GPU_MEM_WAIT_MIN': '5'}):
            with gpu_backend.session(lambda s: None) as a, gpu_backend.session(lambda s: None) as b:
                self.assertIs(a, b)
            api.ensure_server.assert_called_once_with(env={'GPU_MEM_WAIT_MIN': '5'})
            api.tunnel.assert_called_once()
            api.tunnel.return_value.__exit__.assert_called_once()
            with gpu_backend.session(lambda s: None):pass   # a later job opens its own
            self.assertEqual(api.tunnel.call_count, 2)

    def test_server_failure_is_translated_and_leaves_no_session(self):
        gpu, api = self.fake_gpu();api.ensure_server.side_effect = GpuApiError('server_down', 'no start')
        with patch('gpu_backend.Gpu', return_value=gpu):
            with self.assertRaises(GpuUnreachable):
                with gpu_backend.session(lambda s: None):pass
        self.assertEqual(gpu_backend._state['users'], 0)

    def test_pipeline_run_opens_the_session_only_for_gpu(self):
        import pipeline
        with patch('pipeline._run', return_value='ok') as inner, patch('gpu_backend.session') as sess, patch('gpu_backend.setting', return_value='replicate'):
            pipeline.run('a', 'b', provider='replicate');sess.assert_not_called()
            pipeline.run('a', 'b', provider='gpu');sess.assert_called_once()
            self.assertEqual(inner.call_count, 2)


class Routing(unittest.TestCase):
    def test_enabled(self):
        with patch('gpu_backend.setting', return_value='gpu'):
            self.assertTrue(gpu_backend.enabled());self.assertFalse(gpu_backend.enabled('replicate'));self.assertTrue(gpu_backend.enabled('gpu'))
        with patch('gpu_backend.setting', return_value='replicate'):
            self.assertFalse(gpu_backend.enabled());self.assertTrue(gpu_backend.enabled('gpu'))

    def test_requires_configuration(self):
        with patch('gpu_backend.setting', return_value=''):
            with self.assertRaises(ValueError):Gpu()

    def test_server_env_only_passes_configured_limits(self):
        values = {'GPU_MEM_WAIT_MIN': '5', 'GPU_API_TTL_MIN': '30'}
        with patch('gpu_backend.setting', side_effect=lambda k, d='': values.get(k, d)):
            self.assertEqual(gpu_backend.server_env(), values)

    def test_no_rsync_transport_left(self):
        for module in ('gpu_backend', 'gpu_api', 'doctor', 'pipeline', 'raw_masks', 'audio_workflow'):
            self.assertNotIn('rsync', (Path(__file__).parent.parent/f'{module}.py').read_text(), module)

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
    def check(self, ssh_code=0, health=None, ensure_error=None, tunnel_error=None):
        import doctor
        values = {'GPU_HOST': 'h', 'GPU_DIR': '~/d', 'GPU_PYTHON': ''}
        api = MagicMock();api.ssh.return_value = SimpleNamespace(returncode=ssh_code, stdout='', stderr='')
        api.health.return_value = health or {'memory_available_gb': 40, 'models_loaded': []}
        if ensure_error:api.ensure_server.side_effect = ensure_error
        if tunnel_error:api.tunnel.side_effect = tunnel_error
        with patch('doctor.setting', side_effect=lambda k, d='': values.get(k, d)), patch('gpu_backend.setting', side_effect=lambda k, d='': values.get(k, d)), \
                patch('gpu_api.GpuApi.from_env', return_value=api):
            return doctor.gpu_checks()[0], api

    def test_ready(self):
        c, api = self.check();self.assertTrue(c.ok);api.ensure_server.assert_called_once();api.health.assert_called_once()
        self.assertNotIn('~', c.message)
    def test_low_memory_is_not_fatal(self):
        c, _ = self.check(health={'memory_available_gb': 5, 'models_loaded': []});self.assertFalse(c.ok);self.assertFalse(c.fatal)
    def test_unreachable_is_fatal(self):
        c, api = self.check(ssh_code=255);self.assertTrue(c.fatal);api.ensure_server.assert_not_called()
    def test_server_that_will_not_start_is_fatal(self):
        c, _ = self.check(ensure_error=GpuApiError('server_down', 'no health'));self.assertTrue(c.fatal);self.assertIn('server_down', c.message)
    def test_tunnel_failure_is_fatal(self):
        c, _ = self.check(tunnel_error=GpuApiError('server_down', 'tunnel exited'));self.assertTrue(c.fatal)
    def test_no_replicate_token_needed(self):
        import doctor
        with patch('doctor.setting', side_effect=lambda k, d='': 'gpu' if k == 'PREP_BACKEND' else ''), patch('doctor.gpu_checks', return_value=[]):
            self.assertEqual([c.name for c in doctor.prep_checks()], [])


class CleanStatus(unittest.TestCase):
    def test_drops_library_warnings_and_keeps_the_last_real_line(self):
        import gpu_backend
        text='[transformers] The `use_cache` argument is deprecated\nUserWarning: something\nsegment: 45% of frames tracked\n[transformers] noisy again'
        self.assertEqual(gpu_backend.clean_status(text),'segment: 45% of frames tracked')
    def test_all_noise_still_returns_something_short(self):
        import gpu_backend
        self.assertEqual(gpu_backend.clean_status('[transformers] only a warning'),'[transformers] only a warning')
        self.assertLessEqual(len(gpu_backend.clean_status('x'*1000)),300)
    def test_translate_cleans_error_text(self):
        import gpu_backend
        e=gpu_backend.translate(GpuApiError('worker_error','[transformers] warn\nCUDA failed: bad thing'),'depth')
        self.assertIn('CUDA failed: bad thing',str(e));self.assertNotIn('[transformers]',str(e))


if __name__ == '__main__':
    unittest.main()
