"""Client for the GPU host inference API (``gpu_workers/server.py``).

Everything runs on this machine except the inference: ``ensure_server()`` starts the server on
the GPU host over ssh if it is not alive, ``tunnel()`` forwards a local port to the host's
loopback-only port, and ``upload`` / ``predict`` / ``download`` talk plain HTTP through it.
No rsync: job data travels through the API and is deleted from the host after download.

    api = GpuApi.from_env()           # GPU_HOST, GPU_DIR, GPU_PYTHON, GPU_API_PORT, GPU_API_TOKEN
    api.ensure_server()
    with api.tunnel():
        out = api.predict("depth", {"video": api.upload(Path("original.mp4"))})
        api.download(out["depth-raw.mp4"], Path("depth-raw.mp4"))
"""
from __future__ import annotations

import contextlib
import json
import shlex
import socket
import subprocess
import time
import zipfile
from pathlib import Path

import requests

DEFAULT_PORT = 18770
SERVER_NAME = "genjutsu-gpu-api"  # process name the server sets (checked before any kill)


class GpuApiError(RuntimeError):
    """``code``: low_memory, bad_input, cuda_error, worker_error, timeout, server_down, ..."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code, self.message = code, message


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class GpuApi:
    def __init__(self, host: str, gpu_dir: str, python: str, *, port: int = DEFAULT_PORT, token: str | None = None,
                 local_port: int | None = None, run=subprocess.run, popen=subprocess.Popen, session=None, sleep=time.sleep):
        if not (host and gpu_dir and python):
            raise ValueError("GPU_HOST, GPU_DIR and GPU_PYTHON must be set")
        self.host, self.gpu_dir, self.python = host, gpu_dir.rstrip("/"), python
        self.port, self.token, self.local_port = int(port), token or None, local_port
        self._run, self._popen, self._sleep = run, popen, sleep
        self.session = session or requests.Session()

    @classmethod
    def from_env(cls, **overrides) -> "GpuApi":
        from config import setting

        values = dict(host=setting("GPU_HOST"), gpu_dir=setting("GPU_DIR"), python=setting("GPU_PYTHON"),
                      port=int(setting("GPU_API_PORT", str(DEFAULT_PORT))), token=setting("GPU_API_TOKEN") or None)
        values.update(overrides)
        return cls(**values)

    # ---- ssh -----------------------------------------------------------------
    def ssh_command(self, remote: str) -> list[str]:
        return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", self.host, remote]

    def ssh(self, remote: str, timeout: float = 60):
        try:
            return self._run(self.ssh_command(remote), capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise GpuApiError("timeout", f"ssh to the GPU host timed out after {timeout:.0f} s")

    def _remote_health(self) -> dict | None:
        """Health JSON read on the host itself (no tunnel needed); None if nothing answers."""
        script = (f"import json,urllib.request as u;r=u.Request('http://127.0.0.1:{self.port}/health'"
                  + (f",headers={{'Authorization':'Bearer '+{self.token!r}}}" if self.token else "")
                  + ");print(u.urlopen(r,timeout=5).read().decode())")
        result = self.ssh(f"{shlex.quote(self.python)} -c {shlex.quote(script)} 2>/dev/null", timeout=30)
        try:
            return json.loads((result.stdout or "").strip().splitlines()[-1])
        except (ValueError, IndexError):
            return None

    def ensure_server(self, env: dict[str, str] | None = None, wait_s: float = 90) -> dict:
        """Start the server on the GPU host unless it is alive (pid file + /health); returns its health."""
        health = self._remote_health()
        if health:
            return health
        assigns = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in {"GPU_DIR": self.gpu_dir, "GPU_API_PORT": self.port,
                                                                    **({"GPU_API_TOKEN": self.token} if self.token else {}),
                                                                    **(env or {})}.items())
        root = shlex.quote(self.gpu_dir)
        # the subshell keeps ssh's stdout/stderr pipes from being inherited by the detached server
        start = (f"mkdir -p {root}/run && (cd {root} && env {assigns} nohup setsid nice -n 10 "
                 f"{shlex.quote(self.python)} {root}/workers/server.py --port {self.port} "
                 f">> {root}/run/server.log 2>&1 < /dev/null &) > /dev/null 2>&1 < /dev/null ; echo started")
        result = self.ssh(start)
        if result.returncode != 0:
            raise GpuApiError("server_down", f"could not start the server: {(result.stderr or '').strip()[-300:]}")
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            self._sleep(2)
            health = self._remote_health()
            if health:
                return health
        raise GpuApiError("server_down", f"the server did not answer /health within {wait_s:.0f} s (see run/server.log on the host)")

    def stop_server(self) -> bool:
        """Ask the server to exit (SIGTERM) only if the pid file names a process called genjutsu-gpu-api."""
        script = (f"p=$(python3 -c \"import json;print(json.load(open('{self.gpu_dir}/run/server.pid'))['pid'])\" 2>/dev/null); "
                  f"[ -n \"$p\" ] && [ \"$(cat /proc/$p/comm 2>/dev/null)\" = {SERVER_NAME[:15]} ] && kill -TERM $p && echo stopped")
        return "stopped" in (self.ssh(script).stdout or "")

    # ---- tunnel --------------------------------------------------------------
    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.local_port or self.port}"

    @contextlib.contextmanager
    def tunnel(self, wait_s: float = 20):
        """ssh -N -L <local>:127.0.0.1:<port> for the duration of the block."""
        self.local_port = self.local_port or free_port()
        command = ["ssh", "-N", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ExitOnForwardFailure=yes",
                   "-o", "ServerAliveInterval=30", "-L", f"{self.local_port}:127.0.0.1:{self.port}", self.host]
        process = self._popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic() + wait_s
            while True:
                if process.poll() is not None:
                    raise GpuApiError("server_down", f"ssh tunnel exited with code {process.returncode}")
                try:
                    socket.create_connection(("127.0.0.1", self.local_port), timeout=1).close()
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise GpuApiError("timeout", "the ssh tunnel did not open")
                    self._sleep(0.3)
            yield self
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except Exception:
                process.kill()

    # ---- HTTP ----------------------------------------------------------------
    def request(self, method: str, path: str, **kwargs):
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        try:
            response = self.session.request(method, self.base + path, headers=headers, timeout=kwargs.pop("timeout", 120), **kwargs)
        except requests.ConnectionError as exc:
            raise GpuApiError("server_down", f"cannot reach the GPU server ({type(exc).__name__}); is the tunnel open?")
        except requests.Timeout:
            raise GpuApiError("timeout", f"{method} {path} timed out")
        if response.status_code >= 400:
            try:
                body = response.json()
            except ValueError:
                body = {}
            raise GpuApiError(body.get("error_code") or f"http_{response.status_code}", body.get("error") or response.text[:300])
        return response

    def health(self) -> dict:
        return self.request("GET", "/health").json()

    def upload(self, path: Path, name: str | None = None) -> str:
        path = Path(path)
        with path.open("rb") as handle:
            response = self.request("POST", "/v1/files", files={"file": (name or path.name, handle)}, timeout=600)
        return response.json()["id"]

    def predict(self, model: str, data: dict, timeout: float = 1800, poll: float = 2.0, on_status=None) -> dict:
        """Create a prediction and wait for it; returns ``output`` ({file name: file id}) or raises GpuApiError."""
        prediction = self.request("POST", "/v1/predictions", json={"model": model, "input": data}).json()
        deadline = time.monotonic() + timeout
        last = None
        while True:
            state = self.request("GET", f"/v1/predictions/{prediction['id']}").json()
            if state["status"] != last and on_status:
                on_status(state["status"])
            last = state["status"]
            if state["status"] == "succeeded":
                self.last_metrics = state.get("metrics")
                return state["output"]
            if state["status"] in ("failed", "canceled"):
                self.last_metrics = state.get("metrics")
                raise GpuApiError(state.get("error_code") or "worker_error", state.get("error") or state["status"])
            if time.monotonic() > deadline:
                with contextlib.suppress(GpuApiError):
                    self.request("DELETE", f"/v1/predictions/{prediction['id']}")
                raise GpuApiError("timeout", f"{model} prediction still {state['status']} after {timeout:.0f} s")
            self._sleep(poll)

    last_metrics: dict | None = None

    def download(self, file_id: str, dest: Path, delete: bool = True) -> Path:
        """Stream a file to ``dest`` then delete it on the host."""
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        response = self.request("GET", f"/v1/files/{file_id}", stream=True, timeout=600)
        with dest.open("wb") as handle:
            for chunk in response.iter_content(1 << 20):
                handle.write(chunk)
        if delete:
            self.delete(file_id)
        return dest

    def download_all(self, output: dict, folder: Path, unzip: bool = True) -> dict[str, Path]:
        """Download every output file into ``folder``; ``*.zip`` are extracted (``instances.zip`` -> instances/...)."""
        folder = Path(folder)
        result = {}
        for name, file_id in output.items():
            path = self.download(file_id, folder / name)
            if unzip and name.endswith(".zip"):
                with zipfile.ZipFile(path) as archive:
                    root = folder.resolve()
                    for member in archive.namelist():  # never write outside the folder
                        if not (folder / member).resolve().is_relative_to(root):
                            raise GpuApiError("bad_output", f"unsafe path in {name}: {member}")
                    archive.extractall(folder)
                path.unlink()
            result[name] = path
        return result

    def delete(self, file_id: str) -> None:
        with contextlib.suppress(GpuApiError):
            self.request("DELETE", f"/v1/files/{file_id}")
