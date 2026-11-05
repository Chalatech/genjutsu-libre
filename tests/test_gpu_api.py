import json, os, sys, tempfile, threading, time, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'gpu_workers'))
sys.path.insert(0, str(ROOT))

import common  # noqa: E402
import server  # noqa: E402
import gpu_api  # noqa: E402


def fake_runner(calls):
    """Stands in for a worker: writes a file and a folder, records the argv."""
    def run(model, argv, cache):
        calls.append((model, argv))
        out = Path(argv[argv.index('--out') + 1]); inp = Path(argv[argv.index('--in') + 1])
        cache.get('fake:' + model, lambda: object())
        if model == 'segment' and (inp / 'original.mp4').read_bytes() == b'lowmem':
            return 3, {'worker': model, 'error': 'only 5 GB of memory available'}
        (out / 'result.txt').write_text(','.join(sorted(p.name for p in inp.iterdir())))
        (out / 'instances' / '1').mkdir(parents=True)
        (out / 'instances' / '1' / 'mask_00000.png').write_bytes(b'png')
        return 0, {'worker': model, 'wall_s': 0.1, 'started': 'x'}
    return run


class ServerCase(unittest.TestCase):
    token = None

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.calls = []
        self.service = server.Service(Path(self.tmp.name), runner=fake_runner(self.calls), ttl_min=60, unload_min=10, exit_min=30)
        self.httpd = server.ThreadingHTTPServer(('127.0.0.1', 0), server.make_handler(self.service, self.token))
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        threading.Thread(target=self.service.worker_loop, daemon=True).start()
        self.addCleanup(lambda: (self.service.stop.set(), self.httpd.shutdown(), self.httpd.server_close()))
        self.api = gpu_api.GpuApi('h', '/g', '/g/py', port=1, local_port=self.httpd.server_address[1], token=self.token, sleep=lambda s: time.sleep(0.01))
        self.media = Path(self.tmp.name) / 'in.bin'; self.media.write_bytes(b'video-bytes')


class RoundTrip(ServerCase):
    def test_health_upload_predict_download_and_cleanup(self):
        self.assertEqual(self.api.health()['models_loaded'], [])
        fid = self.api.upload(self.media, 'original.mp4')
        output = self.api.predict('depth', {'video': fid, 'fp32': True, 'input_size': 392}, timeout=10)
        self.assertEqual(sorted(output), ['instances.zip', 'result.txt'])
        model, argv = self.calls[0]
        self.assertEqual(model, 'depth'); self.assertIn('--fp32', argv); self.assertEqual(argv[argv.index('--input-size') + 1], '392')
        folder = Path(self.tmp.name) / 'got'
        got = self.api.download_all(output, folder)
        self.assertEqual((folder / 'result.txt').read_text(), 'original.mp4')
        self.assertEqual((folder / 'instances' / '1' / 'mask_00000.png').read_bytes(), b'png')
        self.assertFalse((folder / 'instances.zip').exists())
        # input consumed and deleted, outputs deleted after download, job folder gone
        self.assertEqual(self.service.files, {})
        self.assertEqual(list((self.service.api / 'jobs').iterdir()), [])
        self.assertEqual(list((self.service.api / 'files').iterdir()), [])
        self.assertEqual(self.api.health()['models_loaded'], ['fake:depth'])

    def test_separate_names_audio_and_segment_options(self):
        a = self.api.upload(self.media, 'original-audio.wav')
        self.api.predict('separate', {'audio': a, 'job_id': 'j1'}, timeout=10)
        b = self.api.upload(self.media, 'clip.mp4')
        self.api.predict('segment', {'video': b, 'prompt': 'dog', 'select': '1,2'}, timeout=10)
        self.assertEqual(self.calls[0][1][-2:], ['--job-id', 'j1'])
        self.assertIn('--select', self.calls[1][1])

    def test_raw_body_upload(self):
        r = self.api.request('POST', '/v1/files?name=x.wav', data=b'abc')
        self.assertEqual(r.status_code, 201); self.assertEqual(r.json()['size'], 3)

    def test_bad_inputs(self):
        for body, code in [({'model': 'nope', 'input': {}}, 'bad_input'), ({'model': 'depth', 'input': {'video': 'zz'}}, 'bad_input')]:
            with self.assertRaises(gpu_api.GpuApiError) as ctx:
                self.api.request('POST', '/v1/predictions', json=body)
            self.assertEqual(ctx.exception.code, code)
        fid = self.api.upload(self.media)
        with self.assertRaises(gpu_api.GpuApiError):
            self.api.request('POST', '/v1/predictions', json={'model': 'depth', 'input': {'video': fid, 'evil': 1}})

    def test_low_memory_is_reported_with_its_own_code(self):
        low = Path(self.tmp.name) / 'low'; low.write_bytes(b'lowmem')
        with self.assertRaises(gpu_api.GpuApiError) as ctx:
            self.api.predict('segment', {'video': self.api.upload(low)}, timeout=10)
        self.assertEqual(ctx.exception.code, 'low_memory')
        self.assertEqual(list((self.service.api / 'jobs').iterdir()), [])

    def test_fifo_one_job_at_a_time(self):
        active, peak = [0], [0]
        def slow(model, argv, cache):
            active[0] += 1; peak[0] = max(peak[0], active[0]); time.sleep(0.15); active[0] -= 1
            return 0, {}
        self.service.runner = slow
        ids = [self.api.request('POST', '/v1/predictions', json={'model': 'depth', 'input': {'video': self.api.upload(self.media)}}).json()['id'] for _ in range(3)]
        deadline = time.time() + 10
        while time.time() < deadline and any(self.service.predictions[i]['status'] not in ('succeeded', 'failed') for i in ids):
            time.sleep(0.05)
        self.assertEqual(peak[0], 1)
        self.assertTrue(all(self.service.predictions[i]['status'] == 'succeeded' for i in ids))

    def test_cancel_queued(self):
        gate = threading.Event()
        self.service.runner = lambda m, a, c: (gate.wait(5), (0, {}))[1]
        mk = lambda: self.api.request('POST', '/v1/predictions', json={'model': 'depth', 'input': {'video': self.api.upload(self.media)}}).json()['id']
        first, second = mk(), mk()
        time.sleep(0.2)
        self.assertTrue(self.api.request('DELETE', f'/v1/predictions/{second}').json()['canceled'])
        gate.set()
        time.sleep(0.4)
        self.assertEqual(self.service.predictions[second]['status'], 'canceled')

    def test_unknown_routes(self):
        with self.assertRaises(gpu_api.GpuApiError) as ctx:
            self.api.request('GET', '/v1/files/' + 'a' * 32)
        self.assertEqual(ctx.exception.code, 'not_found')


class Token(ServerCase):
    token = 's3cret'

    def test_requires_bearer(self):
        self.assertEqual(self.api.health()['status'], 'ok')
        bad = gpu_api.GpuApi('h', '/g', '/g/py', local_port=self.api.local_port, token='nope')
        with self.assertRaises(gpu_api.GpuApiError) as ctx:
            bad.health()
        self.assertEqual(ctx.exception.code, 'unauthorized')


class Lifecycle(unittest.TestCase):
    def test_unload_then_exit_when_idle_and_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = server.Service(Path(tmp), runner=fake_runner([]), unload_min=10, exit_min=30)
            s.unload_hook = MagicMock(return_value=['m'])
            s.cache.items['m'] = 1
            s.add_file('a.txt', b'x')
            s.sweep(); s.unload_hook.assert_not_called(); self.assertFalse(s.stop.is_set())
            s.last_work -= 11 * 60; s.last_request -= 11 * 60          # 11 idle minutes: unload, keep running
            s.sweep(); s.unload_hook.assert_called_once(); self.assertFalse(s.stop.is_set())
            s.cache.items.clear()
            s.last_work -= 25 * 60; s.last_request -= 25 * 60          # 36 idle minutes: exit
            s.sweep(); self.assertTrue(s.stop.is_set())
            s.cleanup(); self.assertFalse((Path(tmp) / 'api').exists())

    def test_busy_server_never_unloads_or_exits(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = server.Service(Path(tmp), unload_min=0, exit_min=0)
            s.cache.items['m'] = 1; s.unload_hook = MagicMock(); s.running = 'job'
            time.sleep(0.05); s.sweep()
            s.unload_hook.assert_not_called(); self.assertFalse(s.stop.is_set())

    def test_files_expire(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = server.Service(Path(tmp), ttl_min=0)
            info = s.add_file('a.txt', b'x'); time.sleep(0.01); s.sweep()
            self.assertIsNone(s.file(info['id'])); self.assertFalse((s.files_dir / info['id']).exists())

    def test_model_cache(self):
        cache = common.ModelCache(); n = []
        loader = lambda: n.append(1) or object()
        a, cached_a = cache.get('k', loader); b, cached_b = cache.get('k', loader)
        self.assertIs(a, b); self.assertEqual((cached_a, cached_b, len(n)), (False, True, 1))
        self.assertEqual(cache.clear(), ['k']); self.assertEqual(cache.keys(), [])
        self.assertEqual(common.cached(SimpleNamespace(), 'x', loader)[1], False)

    def test_error_codes(self):
        self.assertEqual(server.error_code(3, None), 'low_memory')
        self.assertEqual(server.error_code(1, 'CUDA out of memory'), 'cuda_error')
        self.assertEqual(server.error_code(1, 'boom'), 'worker_error')

    def test_worker_parsers_expose_defaults(self):
        import depth, segment, separate
        self.assertEqual(depth.make_parser().parse_args(['--in', 'a', '--out', 'b']).input_size, 518)
        self.assertEqual(segment.make_parser().parse_args(['--in', 'a', '--out', 'b']).prompt, 'person')
        self.assertEqual(separate.make_parser().parse_args(['--in', 'a', '--out', 'b']).shifts, 1)


class ClientSsh(unittest.TestCase):
    def make(self, outputs, **kw):
        run = MagicMock(side_effect=[SimpleNamespace(returncode=0, stdout=o, stderr='') for o in outputs])
        return gpu_api.GpuApi('gpu', '/home/u/gpu', '/home/u/gpu/.venv/bin/python', port=18770, run=run, sleep=lambda s: None, **kw), run

    def test_ensure_server_reuses_a_live_one(self):
        api, run = self.make(['{"status": "ok", "pid": 7}'])
        self.assertEqual(api.ensure_server()['pid'], 7); self.assertEqual(run.call_count, 1)

    def test_ensure_server_starts_detached_and_waits(self):
        api, run = self.make(['', 'started\n', '', '{"status": "ok"}'], token='t0k')
        api.ensure_server(env={'GPU_API_EXIT_IDLE_MIN': '1'})
        start = run.call_args_list[1].args[0]
        self.assertEqual(start[:5], ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10'])
        text = start[-1]
        for part in ('nohup setsid nice -n 10', 'workers/server.py --port 18770', 'GPU_API_EXIT_IDLE_MIN=1', 'GPU_API_TOKEN=t0k', '< /dev/null &) > /dev/null 2>&1 < /dev/null ; echo started'):
            self.assertIn(part, text)

    def test_ensure_server_fails_clearly(self):
        api, run = self.make(['', 'started'] + [''] * 60)
        with self.assertRaises(gpu_api.GpuApiError) as ctx:
            api.ensure_server(wait_s=0)
        self.assertEqual(ctx.exception.code, 'server_down')

    def test_stop_checks_process_name_before_kill(self):
        api, run = self.make(['stopped\n'])
        self.assertTrue(api.stop_server())
        self.assertIn('/proc/$p/comm', run.call_args.args[0][-1]); self.assertIn('genjutsu-gpu-ap', run.call_args.args[0][-1])

    def test_tunnel_command_and_close(self):
        proc = MagicMock(); proc.poll.return_value = None
        popen = MagicMock(return_value=proc)
        api = gpu_api.GpuApi('gpu', '/g', '/p', port=18770, popen=popen, sleep=lambda s: None)
        import socket
        srv = socket.socket(); srv.bind(('127.0.0.1', 0)); srv.listen(1); api.local_port = srv.getsockname()[1]
        with api.tunnel():
            cmd = popen.call_args.args[0]
            self.assertIn(f'{api.local_port}:127.0.0.1:18770', cmd); self.assertIn('-N', cmd); self.assertEqual(cmd[-1], 'gpu')
        proc.terminate.assert_called_once(); srv.close()

    def test_server_down_message(self):
        api = gpu_api.GpuApi('gpu', '/g', '/p', local_port=gpu_api.free_port())
        with self.assertRaises(gpu_api.GpuApiError) as ctx:
            api.health()
        self.assertEqual(ctx.exception.code, 'server_down')

    def test_zip_slip_rejected(self):
        import zipfile
        with tempfile.TemporaryDirectory() as tmp:
            z = Path(tmp) / 'o.zip'
            with zipfile.ZipFile(z, 'w') as h: h.writestr('../evil', 'x')
            api = gpu_api.GpuApi('g', '/g', '/p')
            api.download = lambda fid, dest, delete=True: z
            with self.assertRaises(gpu_api.GpuApiError):
                api.download_all({'o.zip': 'x'}, Path(tmp) / 'out')


if __name__ == '__main__':
    unittest.main()
