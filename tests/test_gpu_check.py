import importlib.util, io, json, os, subprocess, sys, tempfile, types, unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

WORKERS = Path(__file__).resolve().parent.parent / 'gpu_workers'
spec = importlib.util.spec_from_file_location('gpu_check', WORKERS / 'check.py')
check = importlib.util.module_from_spec(spec); spec.loader.exec_module(check)


def fake_torch(cuda=True, capability=(12, 1), arch_list=('sm_80', 'sm_90', 'sm_120')):
    cuda_ns = types.SimpleNamespace(
        is_available=lambda: cuda, get_device_capability=lambda index: capability,
        get_device_name=lambda index: 'Test GPU', get_arch_list=lambda: list(arch_list))
    return types.SimpleNamespace(__version__='0.0+test', version=types.SimpleNamespace(cuda='13.0'), cuda=cuda_ns)


def run_main(torch_module, available=64.0, argv=()):
    out = io.StringIO()
    with patch.dict(sys.modules, {'torch': torch_module}), patch.object(check, 'free_memory_gb', return_value=available), \
            patch.object(check, 'matmul_ms', return_value=(7.0, True)) as matmul, patch.dict(os.environ, {'GPU_MIN_FREE_GB': '30'}), \
            redirect_stdout(out):
        code = check.main(list(argv))
    return code, json.loads(out.getvalue()), matmul


class CapabilityTests(unittest.TestCase):
    def test_same_major_native_code_runs_on_newer_minor(self):
        self.assertTrue(check.capability_supported((12, 1), ['sm_80', 'sm_90', 'sm_100', 'sm_110', 'sm_120']))

    def test_other_major_native_code_does_not_run(self):
        self.assertFalse(check.capability_supported((12, 1), ['sm_80', 'sm_90a', 'sm_100']))
        self.assertFalse(check.capability_supported((8, 0), ['sm_86']))

    def test_ptx_runs_on_equal_or_newer_gpus(self):
        self.assertTrue(check.capability_supported((12, 1), ['compute_90']))
        self.assertFalse(check.capability_supported((8, 0), ['compute_90']))
        self.assertFalse(check.capability_supported((12, 1), ['garbage', '']))


class MemoryTests(unittest.TestCase):
    def test_reads_available_memory(self):
        with tempfile.NamedTemporaryFile('w', suffix='meminfo', delete=False) as handle:
            handle.write('MemTotal:       127000000 kB\nMemAvailable:   31457280 kB\n')
        try:
            self.assertEqual(check.free_memory_gb(handle.name), 30.0)
        finally:
            os.unlink(handle.name)

    def test_missing_meminfo_is_unknown(self):
        self.assertIsNone(check.free_memory_gb('/nonexistent/meminfo'))


class MainTests(unittest.TestCase):
    def test_exits_1_without_torch(self):
        code, report, _ = run_main(None)
        self.assertEqual(code, 1); self.assertIn('torch not importable', report['error'])

    def test_exits_1_without_cuda(self):
        code, report, matmul = run_main(fake_torch(cuda=False))
        self.assertEqual(code, 1); self.assertFalse(report['cuda']); matmul.assert_not_called()

    def test_exits_3_and_skips_gpu_work_when_memory_is_low(self):
        code, report, matmul = run_main(fake_torch(), available=12.5)
        self.assertEqual(code, 3); self.assertIn('12.5', report['error']); matmul.assert_not_called()

    def test_exits_1_when_build_has_no_kernels_for_the_gpu(self):
        code, report, _ = run_main(fake_torch(arch_list=('sm_80', 'sm_90')))
        self.assertEqual(code, 1); self.assertFalse(report['capability_supported'])

    def test_exits_0_and_reports_the_gpu(self):
        code, report, _ = run_main(fake_torch())
        self.assertEqual(code, 0)
        self.assertEqual((report['gpu'], report['capability'], report['matmul_4096_ms']), ('Test GPU', '12.1', 7.0))
        self.assertTrue(report['capability_supported'])

    def test_components_found_in_gpu_dir(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); sam3 = root / 'models' / 'sam3'; sam3.mkdir(parents=True)
            (sam3 / 'config.json').write_text('{"model_type": "sam3_video"}'); (sam3 / 'model.safetensors').write_bytes(b'x')
            (root / 'models' / 'video-depth-anything').mkdir(); (root / 'models' / 'video-depth-anything' / 'video_depth_anything_vits.pth').write_bytes(b'x')
            with patch.dict(os.environ, {'GPU_DIR': d, 'SAM3_WEIGHTS': ''}):
                found = check.components(check.gpu_dir())
        self.assertEqual((found['sam3_weights'], found['sam3_model_type']), (str(sam3), 'sam3_video'))
        self.assertTrue(found['vda_small_weights']); self.assertFalse(found['vda_source'])


class SetupScriptTests(unittest.TestCase):
    def test_setup_script_is_valid_user_space_bash(self):
        script = WORKERS / 'setup_gpu.sh'
        subprocess.run(['bash', '-n', str(script)], check=True)
        text = script.read_text()
        for forbidden in ('sudo ', 'apt ', 'apt-get', '~/.cache', 'vitl.pth'):
            self.assertNotIn(forbidden, text)

    def test_requirements_never_replace_cuda_torch(self):
        names = [line.split('#')[0].strip().lower() for line in (WORKERS / 'requirements-gpu.txt').read_text().splitlines()]
        self.assertFalse([n for n in names if n.startswith(('torch', 'torchvision'))])
        self.assertTrue(any(n.startswith('transformers') for n in names))


if __name__ == '__main__':
    unittest.main()
