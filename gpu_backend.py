"""Preparation workers on a remote GPU host, driven over ssh/rsync (PREP_BACKEND=gpu).

Configuration (.env): GPU_HOST (ssh alias), GPU_DIR (the only folder written on the host),
GPU_PYTHON (the venv python on the host), GPU_MIN_FREE_GB / GPU_MEM_POLL_S / GPU_MEM_WAIT_MIN.

Every stage is self-contained and leaves nothing behind on the host:
push the inputs to ``$GPU_DIR/jobs/<job>/in`` -> run one worker (``nice -n 10``, one at a time,
remote ``timeout``) -> pull ``jobs/<job>/out`` -> delete ``jobs/<job>`` (always, even on failure).
Nothing here kills remote processes; the only limit is the remote ``timeout`` around our own worker.
"""
from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from config import setting

SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10']
JOB_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$')
TIMEOUTS = {'separate': 900, 'depth': 1800, 'segment': 3600}   # seconds; segment ~0.8 s/frame with many objects


class GpuError(RuntimeError):
    """A remote stage failed (the message carries the worker's own error when it printed one)."""


class GpuUnreachable(GpuError):
    """ssh could not reach the GPU host."""


class GpuMemoryError(GpuError):
    """Not enough free memory on the GPU host, even after waiting."""


def enabled(provider=None):
    """True when preparation must run on the GPU host (provider 'gpu', or PREP_BACKEND=gpu by default)."""
    if provider in (None, ''):
        return setting('PREP_BACKEND', 'replicate') == 'gpu'
    return provider == 'gpu'


def remote_path(path):
    """Quote a remote path for the shell, keeping a leading ``~/`` expandable."""
    path = str(path)
    if path == '~':return '"$HOME"'
    if path.startswith('~/'):return '"$HOME"/'+shlex.quote(path[2:])
    return shlex.quote(path)


class Gpu:
    def __init__(self, host=None, directory=None, python=None, status=print, sleep=time.sleep, clock=time.monotonic):
        self.host = host or setting('GPU_HOST')
        self.dir = (directory or setting('GPU_DIR')).rstrip('/')
        if not self.host or not self.dir:raise ValueError('Set GPU_HOST and GPU_DIR in this project’s .env file (see docs/PLAN-BACKENDS.md).')
        self.python = python or setting('GPU_PYTHON') or self.dir+'/.venv/bin/python'
        self.status, self.sleep, self.clock = status, sleep, clock

    # ---- plumbing -----------------------------------------------------------------
    def job_dir(self, job):
        if not JOB_RE.match(str(job)):raise ValueError(f'Unsafe job name {job!r}.')
        return f'{self.dir}/jobs/{job}'

    def ssh(self, command, timeout=60):
        try:r = subprocess.run(SSH+[self.host, command], capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:raise GpuUnreachable(f'GPU host did not answer within {timeout} s.') from exc
        if r.returncode == 255:
            lines = (r.stderr or '').strip().splitlines()
            raise GpuUnreachable('Cannot reach the GPU host over ssh'+(': '+lines[-1] if lines else '.'))
        return r

    def rsync(self, args, timeout=900):
        r = subprocess.run(['rsync', '-az', '-e', ' '.join(SSH)]+args, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:raise GpuError(f'rsync failed ({r.returncode}): '+(r.stderr or '').strip()[-300:])

    # ---- memory -------------------------------------------------------------------
    def free_gb(self):
        r = self.ssh("grep MemAvailable /proc/meminfo")
        try:return round(int(r.stdout.split()[1])/1024/1024, 1)
        except (IndexError, ValueError):return None

    def wait_for_memory(self):
        """Same policy as the workers: re-check every GPU_MEM_POLL_S for up to GPU_MEM_WAIT_MIN."""
        need = float(setting('GPU_MIN_FREE_GB', '20'))
        poll = float(setting('GPU_MEM_POLL_S', '120'));limit = float(setting('GPU_MEM_WAIT_MIN', '30'))*60
        started = self.clock()
        while True:
            free = self.free_gb()
            if free is None or free >= need:return free
            waited = self.clock()-started
            if waited+poll > limit:
                raise GpuMemoryError(f'GPU host has {free} GB of memory available, {need:g} GB needed; gave up after {waited:.0f} s. Retry later.')
            self.status(f'GPU host · only {free} GB free, waiting for {need:g} GB')
            self.sleep(poll)

    # ---- one stage ----------------------------------------------------------------
    def push(self, job, folder, names):
        base = self.job_dir(job)
        self.ssh(f'mkdir -p {remote_path(base+"/in")} {remote_path(base+"/out")}')
        self.rsync([str(Path(folder)/n) for n in names]+[f'{self.host}:{base}/in/'])

    def run(self, worker, job, args=(), folder=None, timeout=None):
        """Run one worker; return its JSON summary. Progress lines go to ``status`` and ``gpu-logs/<worker>.log``."""
        base = self.job_dir(job);timeout = int(timeout or TIMEOUTS.get(worker, 1800))
        self.wait_for_memory()
        command = (f'cd {remote_path(self.dir)} && nice -n 10 timeout -k 30 {timeout} {remote_path(self.python)} workers/{worker}.py '
                   f'--in {remote_path(base+"/in")} --out {remote_path(base+"/out")} '+' '.join(shlex.quote(str(a)) for a in args))
        log = None
        if folder is not None:
            (Path(folder)/'gpu-logs').mkdir(exist_ok=True);log = (Path(folder)/'gpu-logs'/f'{worker}.log').open('w')
        summary = {};tail = []
        try:
            proc = subprocess.Popen(SSH+[self.host, command], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                line = line.rstrip('\n')
                if log:log.write(line+'\n');log.flush()
                if line.startswith('{'):
                    try:summary = json.loads(line);continue
                    except ValueError:pass
                tail.append(line);tail[:] = tail[-8:]
                if line.startswith('['):self.status('GPU host · '+line)
            code = proc.wait()
        finally:
            if log:log.close()
        if code == 0:return summary
        error = summary.get('error') or ' | '.join(tail[-3:])
        if code == 3:raise GpuMemoryError(f'{worker}: GPU host ran out of free memory ({error})')
        if code == 255:raise GpuUnreachable(f'{worker}: lost the ssh connection ({error})')
        if code == 124:raise GpuError(f'{worker}: stopped after the {timeout} s limit')
        raise GpuError(f'{worker} failed on the GPU host (exit {code}): {error}')

    def pull(self, job, dest):
        Path(dest).mkdir(parents=True, exist_ok=True)
        self.rsync([f'{self.host}:{self.job_dir(job)}/out/', str(dest)+'/'])

    def cleanup(self, job):
        base = self.job_dir(job)
        if not base.startswith(self.dir+'/jobs/'):return
        try:self.ssh(f'rm -rf {remote_path(base)}', timeout=120)
        except GpuError:pass

    def stage(self, worker, folder, names, args=(), timeout=None, source=None):
        """push (from ``source``, default ``folder``) -> run -> pull into ``<folder>/gpu-out`` -> delete the host copy."""
        folder = Path(folder);job = job_name(folder);out = folder/'gpu-out'
        try:
            self.push(job, source or folder, names)
            summary = self.run(worker, job, args, folder, timeout)
            self.pull(job, out)
        finally:
            self.cleanup(job)
        reports = folder/'gpu-reports';reports.mkdir(exist_ok=True)
        for report in out.glob('report-*.json'):shutil.copy2(report, reports/report.name)
        return summary, out


def job_name(folder):
    """Host job id: the job folder name (the repair pass lives in ``<job>/rgb-repair``)."""
    folder = Path(folder)
    return f'{folder.parent.name}-{folder.name}' if folder.name == 'rgb-repair' else folder.name


# ---- the three routed stages ---------------------------------------------------------
def separate(folder, status=print, gpu=None):
    """Demucs on the host. Expects ``original-audio.wav`` in the job folder; writes the stems and audio-manifest.json."""
    folder = Path(folder);gpu = gpu or Gpu(status=status)
    status('Audio · separating vocals and music on the GPU host')
    _, out = gpu.stage('separate', folder, ['original-audio.wav'])
    for name in ('vocals.wav', 'music.wav', 'audio-manifest.json'):
        if not (out/name).exists():raise GpuError(f'separate: {name} missing from the host output.')
    for wav in out.glob('*.wav'):
        if wav.name != 'original-audio.wav':shutil.copy2(wav, folder/wav.name)
    shutil.copy2(out/'audio-manifest.json', folder/'audio-manifest.json')
    shutil.rmtree(out, ignore_errors=True)


def depth(folder, status=print, gpu=None):
    """Video-Depth-Anything Small on the host: ``original.mp4`` -> ``depth-raw.mp4``."""
    folder = Path(folder);gpu = gpu or Gpu(status=status)
    _, out = gpu.stage('depth', folder, ['original.mp4'])
    if not (out/'depth-raw.mp4').exists():raise GpuError('depth: depth-raw.mp4 missing from the host output.')
    shutil.copy2(out/'depth-raw.mp4', folder/'depth-raw.mp4')
    shutil.rmtree(out, ignore_errors=True)


def segment(video, folder, prompt='person', status=print, gpu=None):
    """SAM 3 on the host: one mask sequence per instance in ``<folder>/instances`` plus ``instances.json`` and the contact sheet."""
    folder = Path(folder);gpu = gpu or Gpu(status=status)
    folder.mkdir(parents=True, exist_ok=True)
    staging = folder/'.segment-in'
    shutil.rmtree(staging, ignore_errors=True);staging.mkdir()
    try:
        shutil.copy2(video, staging/'original.mp4')
        _, out = gpu.stage('segment', folder, ['original.mp4'], ['--prompt', prompt], source=staging)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    for name in ('instances.json', 'instances-contact.jpg'):
        if not (out/name).exists():raise GpuError(f'segment: {name} missing from the host output.')
        shutil.copy2(out/name, folder/name)
    shutil.rmtree(folder/'instances', ignore_errors=True)
    shutil.move(str(out/'instances'), str(folder/'instances'))
    shutil.rmtree(out, ignore_errors=True)
    return json.loads((folder/'instances.json').read_text())
