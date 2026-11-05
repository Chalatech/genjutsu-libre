"""Preparation stages on the GPU host through its inference API (PREP_BACKEND=gpu).

Everything runs on this machine except the inference. A preparation job opens one session:
``ensure_server()`` (starts the server on the host if it is not alive) and one ssh tunnel to its
loopback-only port. Each stage uploads its normalized input, runs the model, downloads the
outputs into the job folder under the same file names as always, and deletes the host copies.

Configuration (.env): GPU_HOST, GPU_DIR, GPU_PYTHON, GPU_API_PORT, GPU_API_TOKEN, and the
server's own limits GPU_MIN_FREE_GB / GPU_MEM_POLL_S / GPU_MEM_WAIT_MIN (low memory is waited
out on the host, then reported as GpuMemoryError) and GPU_API_UNLOAD_IDLE_MIN /
GPU_API_EXIT_IDLE_MIN / GPU_API_TTL_MIN. See ``gpu_api.py`` for the client itself.
"""
from __future__ import annotations

import contextlib
import json
import shutil
import threading
from pathlib import Path

from config import setting
from gpu_api import GpuApi, GpuApiError

TIMEOUTS = {'separate': 900, 'depth': 1800, 'segment': 3600}   # seconds of compute; segment ~0.8 s/frame with many objects
SERVER_SETTINGS = ('GPU_MIN_FREE_GB', 'GPU_MEM_POLL_S', 'GPU_MEM_WAIT_MIN',
                   'GPU_API_UNLOAD_IDLE_MIN', 'GPU_API_EXIT_IDLE_MIN', 'GPU_API_TTL_MIN')


class GpuError(RuntimeError):
    """A stage failed on the GPU host (the message carries the worker's own error when it printed one)."""


class GpuUnreachable(GpuError):
    """The GPU host or its server could not be reached."""


class GpuMemoryError(GpuError):
    """Not enough free memory on the GPU host, even after waiting."""


def enabled(provider=None):
    """True when preparation must run on the GPU host (provider 'gpu', or PREP_BACKEND=gpu by default)."""
    if provider in (None, ''):
        return setting('PREP_BACKEND', 'replicate') == 'gpu'
    return provider == 'gpu'


def translate(error: GpuApiError, what: str) -> GpuError:
    if error.code == 'low_memory':return GpuMemoryError(f'{what}: the GPU host ran out of free memory ({error})')
    if error.code in ('server_down', 'unauthorized'):return GpuUnreachable(f'{what}: {error}')
    if error.code == 'timeout':return GpuError(f'{what}: timed out ({error})')
    return GpuError(f'{what} failed on the GPU host ({error.code}): {error}')


def server_env():
    """Limits the server is started with (only the ones configured; the server has its own defaults)."""
    return {k: setting(k) for k in SERVER_SETTINGS if setting(k)}


class Gpu:
    def __init__(self, api=None, status=print):
        if api is None:
            if not setting('GPU_HOST') or not setting('GPU_DIR'):
                raise ValueError('Set GPU_HOST and GPU_DIR in this project’s .env file (see docs/PLAN-BACKENDS.md).')
            api = GpuApi.from_env(python=setting('GPU_PYTHON') or setting('GPU_DIR').rstrip('/')+'/.venv/bin/python')
        self.api, self.status = api, status

    def timeout(self, model):
        return TIMEOUTS.get(model, 1800)+float(setting('GPU_MEM_WAIT_MIN', '30'))*60

    def stage(self, model, source, data=None, upload_name=None, folder=None):
        """upload ``source`` -> predict -> download into ``<folder>/gpu-out`` -> delete the host copies. Returns that folder.

        ``data`` are the extra model options; the uploaded file id goes in the model's file key."""
        key = {'separate': 'audio'}.get(model, 'video')
        out = Path(folder)/'gpu-out'
        shutil.rmtree(out, ignore_errors=True)
        file_id, output = None, {}
        try:
            file_id = self.api.upload(Path(source), upload_name)
            output = self.api.predict(model, {key: file_id, **(data or {})}, timeout=self.timeout(model),
                                      on_status=lambda s: self.status(f'GPU host · {model} {s}'))
            file_id = None   # consumed by the job
            self.api.download_all(output, out)
            output = {}
        except GpuApiError as exc:
            raise translate(exc, model) from exc
        finally:
            for leftover in ([file_id] if file_id else [])+list(output.values()):self.api.delete(leftover)
        self.record(model, folder, out)
        return out

    def record(self, model, folder, out):
        reports = Path(folder)/'gpu-reports';reports.mkdir(exist_ok=True)
        for report in out.glob('report-*.json'):shutil.copy2(report, reports/report.name)
        if self.api.last_metrics:(reports/f'api-{model}.json').write_text(json.dumps(self.api.last_metrics, indent=2))


_lock = threading.Lock()
_state = {'gpu': None, 'users': 0, 'stack': None}


@contextlib.contextmanager
def session(status=print):
    """One server check and one tunnel for everything inside the block; nested/concurrent users share it."""
    with _lock:
        if _state['users'] == 0:
            gpu = Gpu(status=status)
            stack = contextlib.ExitStack()
            try:
                status('GPU host · connecting')
                try:
                    gpu.api.ensure_server(env=server_env())
                    stack.enter_context(gpu.api.tunnel())
                except GpuApiError as exc:raise translate(exc, 'GPU host') from exc
            except BaseException:
                stack.close();raise
            _state.update(gpu=gpu, stack=stack)
        _state['users'] += 1
        gpu = _state['gpu']
    try:
        yield gpu
    finally:
        with _lock:
            _state['users'] -= 1
            if _state['users'] == 0:
                stack, _state['gpu'], _state['stack'] = _state['stack'], None, None
                stack.close()


# ---- the three routed stages ---------------------------------------------------------
def separate(folder, status=print, gpu=None):
    """Demucs on the host. Expects ``original-audio.wav`` in the job folder; writes the stems and audio-manifest.json."""
    folder = Path(folder)
    status('Audio · separating vocals and music on the GPU host')
    with contextlib.ExitStack() as stack:
        gpu = gpu or stack.enter_context(session(status))
        out = gpu.stage('separate', folder/'original-audio.wav', {'job_id': folder.name}, folder=folder)
    try:
        for name in ('vocals.wav', 'music.wav', 'audio-manifest.json'):
            if not (out/name).exists():raise GpuError(f'separate: {name} missing from the host output.')
        for wav in out.glob('*.wav'):
            if wav.name != 'original-audio.wav':shutil.copy2(wav, folder/wav.name)
        shutil.copy2(out/'audio-manifest.json', folder/'audio-manifest.json')
    finally:
        shutil.rmtree(out, ignore_errors=True)


def depth(folder, status=print, gpu=None):
    """Video-Depth-Anything Small on the host: ``original.mp4`` -> ``depth-raw.mp4``."""
    folder = Path(folder)
    with contextlib.ExitStack() as stack:
        gpu = gpu or stack.enter_context(session(status))
        out = gpu.stage('depth', folder/'original.mp4', folder=folder)
    try:
        if not (out/'depth-raw.mp4').exists():raise GpuError('depth: depth-raw.mp4 missing from the host output.')
        shutil.copy2(out/'depth-raw.mp4', folder/'depth-raw.mp4')
    finally:
        shutil.rmtree(out, ignore_errors=True)


def segment(video, folder, prompt='person', status=print, gpu=None):
    """SAM 3 on the host: one mask sequence per instance in ``<folder>/instances`` plus ``instances.json`` and the contact sheet."""
    folder = Path(folder);folder.mkdir(parents=True, exist_ok=True)
    with contextlib.ExitStack() as stack:
        gpu = gpu or stack.enter_context(session(status))
        out = gpu.stage('segment', video, {'prompt': prompt}, upload_name='original.mp4', folder=folder)
    try:
        for name in ('instances.json', 'instances-contact.jpg'):
            if not (out/name).exists():raise GpuError(f'segment: {name} missing from the host output.')
            shutil.copy2(out/name, folder/name)
        shutil.rmtree(folder/'instances', ignore_errors=True)
        shutil.move(str(out/'instances'), str(folder/'instances'))
    finally:
        shutil.rmtree(out, ignore_errors=True)
    return json.loads((folder/'instances.json').read_text())
