#!/usr/bin/env python3
"""Inference API for the GPU host: separate, depth and segment behind a small HTTP server.

    python server.py            # listens on 127.0.0.1:$GPU_API_PORT (default 18770)

Replicate-style endpoints (standard library only, so it runs in the workers' venv):

    POST   /v1/files                  multipart upload (field ``file``) or a raw body with ?name= -> {id, name, size}
    GET    /v1/files/{id}[?delete=1]  download (optionally delete after sending)
    DELETE /v1/files/{id}
    POST   /v1/predictions            {"model": "separate"|"depth"|"segment", "input": {...}} -> {id, status}
    GET    /v1/predictions/{id}       {status, output, error, error_code, metrics}
    DELETE /v1/predictions/{id}       cancel (queued only) / forget
    GET    /health                    loaded models, memory, queue, idle times

Inputs reference uploaded file ids. ``separate``: ``audio`` (wav or any media with audio), ``job_id``,
``shifts``. ``depth``: ``video`` (normalized original.mp4), ``input_size``, ``fp32``, ``attention``.
``segment``: ``video``, ``prompt``, ``select`` (ids to merge), ``min_frames``, ``min_area``.
The output maps each file the worker wrote to a file id; folders become ``<name>.zip``.

Respect for a shared host: it only listens on loopback (reach it through an ssh tunnel), loads a
model on first use, one GPU job at a time (FIFO), the same memory wait as the CLI workers, models
are dropped after GPU_API_UNLOAD_IDLE_MIN minutes without work and the process exits after
GPU_API_EXIT_IDLE_MIN minutes without requests, deleting everything under $GPU_DIR/api. Input
files are deleted when their job ends; outputs after download or after GPU_API_TTL_MIN minutes.
Error codes: bad_input, low_memory (worker exit 3), cuda_error, worker_error, canceled.
"""
from __future__ import annotations

import common  # first: sets the huge-page switches before numpy / torch load

import json
import os
import queue
import re
import shutil
import sys
import threading
import time
import uuid
import zipfile
from email.parser import BytesParser
from email.policy import HTTP
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

MAX_UPLOAD = 1 << 30
ID_RE = re.compile(r"^[0-9a-f]{32}$")
# model -> (module, {input key: (cli flag, kind)}, file inputs)
MODELS = {
    "separate": {"module": "separate", "files": {"audio": None},
                 "options": {"job_id": "--job-id", "shifts": "--shifts", "model": "--model"}},
    "depth": {"module": "depth", "files": {"video": "original.mp4"},
              "options": {"input_size": "--input-size", "fp32": "--fp32", "attention": "--attention"}},
    "segment": {"module": "segment", "files": {"video": "original.mp4"},
                "options": {"prompt": "--prompt", "select": "--select", "min_frames": "--min-frames",
                            "min_area": "--min-area", "model": "--model"}},
}
FLAG_OPTIONS = {"--fp32"}


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return float(default)


class BadInput(ValueError):
    pass


class Service:
    """State of the server: files, predictions, the FIFO queue and the model cache."""

    def __init__(self, root: Path, *, runner=None, ttl_min=None, unload_min=None, exit_min=None):
        self.root = Path(root)
        self.api = self.root / "api"
        self.files_dir = self.api / "files"
        self.jobs_dir = self.api / "jobs"
        self.run_dir = self.root / "run"
        for folder in (self.files_dir, self.jobs_dir, self.run_dir):
            folder.mkdir(parents=True, exist_ok=True)
        self.ttl_s = 60 * (env_float("GPU_API_TTL_MIN", 60) if ttl_min is None else ttl_min)
        self.unload_s = 60 * (env_float("GPU_API_UNLOAD_IDLE_MIN", 10) if unload_min is None else unload_min)
        self.exit_s = 60 * (env_float("GPU_API_EXIT_IDLE_MIN", 30) if exit_min is None else exit_min)
        self.runner = runner or run_job
        self.cache = common.ModelCache()
        self.files: dict[str, dict] = {}
        self.predictions: dict[str, dict] = {}
        self.queue: queue.Queue = queue.Queue()
        self.lock = threading.RLock()
        self.running: str | None = None
        self.started = time.time()
        self.last_request = time.monotonic()
        self.last_work = time.monotonic()
        self.stop = threading.Event()
        self.unloaded_at: float | None = None
        self.unload_hook = lambda cache: cache.clear(sys.modules.get("torch"))

    # ---- files ----
    def add_file(self, name: str, data: bytes | None = None, path: Path | None = None) -> dict:
        ident = uuid.uuid4().hex
        folder = self.files_dir / ident
        folder.mkdir()
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", Path(name or "file").name)[:120] or "file"
        target = folder / safe
        if data is not None:
            target.write_bytes(data)
        elif path is not None:
            shutil.move(str(path), target)
        entry = {"id": ident, "name": safe, "path": target, "size": target.stat().st_size,
                 "expires": time.monotonic() + self.ttl_s}
        with self.lock:
            self.files[ident] = entry
        return {k: entry[k] for k in ("id", "name", "size")}

    def file(self, ident: str) -> dict | None:
        with self.lock:
            return self.files.get(ident) if ID_RE.match(ident or "") else None

    def delete_file(self, ident: str) -> bool:
        with self.lock:
            entry = self.files.pop(ident, None)
        if entry:
            shutil.rmtree(self.files_dir / ident, ignore_errors=True)
        return bool(entry)

    # ---- predictions ----
    def submit(self, body: dict) -> dict:
        model, data = body.get("model"), body.get("input") or {}
        if model not in MODELS:
            raise BadInput(f"unknown model {model!r}; use one of {sorted(MODELS)}")
        if not isinstance(data, dict):
            raise BadInput("input must be an object")
        spec = MODELS[model]
        for key in spec["files"]:
            if not self.file(str(data.get(key, ""))):
                raise BadInput(f"input.{key} must be the id of an uploaded file")
        unknown = sorted(set(data) - set(spec["files"]) - set(spec["options"]))
        if unknown:
            raise BadInput(f"unknown input keys {unknown}")
        ident = uuid.uuid4().hex
        prediction = {"id": ident, "model": model, "status": "starting", "input": data, "output": None,
                      "error": None, "error_code": None, "metrics": None, "created": time.time(),
                      "expires": None}
        with self.lock:
            self.predictions[ident] = prediction
        self.queue.put(ident)
        self.touch()
        return self.public(prediction)

    @staticmethod
    def public(prediction: dict) -> dict:
        return {k: prediction[k] for k in ("id", "model", "status", "output", "error", "error_code", "metrics")}

    def cancel(self, ident: str) -> bool:
        with self.lock:
            prediction = self.predictions.get(ident)
            if prediction and prediction["status"] == "starting":
                prediction.update(status="canceled", error_code="canceled", expires=time.monotonic() + self.ttl_s)
                return True
        return False

    def worker_loop(self) -> None:
        while not self.stop.is_set():
            try:
                ident = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue
            with self.lock:
                prediction = self.predictions.get(ident)
                if not prediction or prediction["status"] != "starting":
                    continue
                prediction["status"] = "processing"
                self.running = ident
            try:
                self.process(prediction)
            except Exception as exc:  # never let the loop die
                prediction.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:1000], error_code="worker_error")
            finally:
                with self.lock:
                    self.running = None
                    self.last_work = time.monotonic()
                    prediction["expires"] = time.monotonic() + self.ttl_s
                    self.unloaded_at = None

    def process(self, prediction: dict) -> None:
        spec = MODELS[prediction["model"]]
        job = self.jobs_dir / prediction["id"]
        inp, out = job / "in", job / "out"
        inp.mkdir(parents=True)
        out.mkdir()
        consumed = []
        try:
            for key, fixed in spec["files"].items():
                entry = self.file(prediction["input"][key])
                if not entry:
                    raise BadInput(f"input file for {key} expired or was deleted")
                consumed.append(entry["id"])
                name = fixed or ("original-audio.wav" if entry["name"] == "original-audio.wav" else "upload" + Path(entry["name"]).suffix)
                try:
                    os.link(entry["path"], inp / name)
                except OSError:
                    shutil.copyfile(entry["path"], inp / name)
            argv = ["--in", str(inp), "--out", str(out)]
            for key, flag in spec["options"].items():
                if key not in prediction["input"]:
                    continue
                value = prediction["input"][key]
                if flag in FLAG_OPTIONS:
                    if value:
                        argv.append(flag)
                else:
                    argv += [flag, str(value)]
            code, report = self.runner(prediction["model"], argv, self.cache)
            prediction["metrics"] = {k: v for k, v in report.items() if k != "started"}
            if code != 0:
                prediction.update(status="failed", error=report.get("error") or f"worker exit {code}",
                                  error_code=error_code(code, report.get("error")))
                return
            prediction["output"] = self.collect(out)
            prediction["status"] = "succeeded"
        except BadInput as exc:
            prediction.update(status="failed", error=str(exc), error_code="bad_input")
        finally:
            shutil.rmtree(job, ignore_errors=True)  # inputs may hold faces: never keep them
            for ident in consumed:
                self.delete_file(ident)

    def collect(self, out: Path) -> dict:
        """Every top-level file of the worker output becomes a file id; folders are zipped."""
        output = {}
        for item in sorted(out.iterdir()):
            if item.is_dir():
                archive = out / f"{item.name}.zip"
                with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as handle:  # PNG/MP4 are already compressed
                    for path in sorted(item.rglob("*")):
                        if path.is_file():
                            handle.write(path, path.relative_to(out).as_posix())
                item = archive
            output[item.name] = self.add_file(item.name, path=item)["id"]
        return output

    # ---- lifecycle ----
    def touch(self) -> None:
        self.last_request = time.monotonic()

    def health(self) -> dict:
        with self.lock:
            return {"status": "ok", "pid": os.getpid(), "models_loaded": self.cache.keys(),
                    "memory_available_gb": common.free_memory_gb(), "queue": self.queue.qsize(),
                    "running": self.running, "uptime_s": round(time.time() - self.started, 1),
                    "idle_work_s": round(time.monotonic() - self.last_work, 1),
                    "idle_request_s": round(time.monotonic() - self.last_request, 1),
                    "files": len(self.files), "unloaded": self.unloaded_at is not None}

    def sweep(self) -> None:
        """Expire files and finished predictions; unload models when idle; flag exit when unused."""
        now = time.monotonic()
        with self.lock:
            for ident in [i for i, e in self.files.items() if e["expires"] < now]:
                self.delete_file(ident)
            for ident in [i for i, p in self.predictions.items() if p["expires"] and p["expires"] < now]:
                for file_id in (self.predictions.pop(ident).get("output") or {}).values():
                    self.delete_file(file_id)
            busy = self.running is not None or not self.queue.empty()
            if not busy and self.cache.keys() and now - self.last_work > self.unload_s:
                dropped = self.unload_hook(self.cache)
                self.unloaded_at = time.time()
                log(f"idle {self.unload_s:.0f} s: unloaded {dropped}")
            if not busy and now - max(self.last_request, self.last_work) > self.exit_s:
                log(f"no requests for {self.exit_s:.0f} s: exiting")
                self.stop.set()

    def cleanup(self) -> None:
        shutil.rmtree(self.api, ignore_errors=True)


def error_code(code: int, message: str | None) -> str:
    if code == common.EXIT_LOW_MEMORY:
        return "low_memory"
    text = (message or "").lower()
    if "cuda" in text and ("out of memory" in text or "not available" in text):
        return "cuda_error"
    return "worker_error"


def log(message: str) -> None:
    print(f"[server] {time.strftime('%H:%M:%S')} {message}", file=sys.stderr, flush=True)


def run_job(model: str, argv: list[str], cache) -> tuple[int, dict]:
    """Run a worker in this process with the shared model cache (needs torch + CUDA)."""
    import importlib

    module = importlib.import_module(MODELS[model]["module"])
    parser = module.make_parser()

    def fail(message):
        raise BadInput(message)

    parser.error = fail
    args = parser.parse_args(argv)
    args.cache = cache
    return common.execute(model, args, module.work)


# ---- HTTP -------------------------------------------------------------------

def parse_multipart(content_type: str, body: bytes) -> tuple[str, bytes]:
    message = BytesParser(policy=HTTP).parsebytes(b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + body)
    for part in message.iter_parts():
        if part.get_param("name", header="content-disposition") in ("file", "content"):
            return part.get_filename() or "file", part.get_payload(decode=True) or b""
    raise BadInput("multipart upload needs a 'file' field")


def make_handler(service: Service, token: str | None):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "genjutsu-gpu"

        def log_message(self, fmt, *args):  # quieter: one line per request on stderr
            log(f"{self.address_string()} {fmt % args}")

        def reply(self, status: int, payload=None, *, body: bytes | None = None, ctype="application/json", headers=None):
            if body is None:
                body = json.dumps(payload if payload is not None else {}).encode()
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def authorized(self) -> bool:
            if not token:
                return True
            if self.headers.get("Authorization") == f"Bearer {token}":
                return True
            self.reply(401, {"error": "missing or wrong bearer token", "error_code": "unauthorized"})
            return False

        def dispatch(self, method: str):
            service.touch()
            if not self.authorized():
                return
            url = urlparse(self.path)
            query = parse_qs(url.query)
            parts = [p for p in url.path.split("/") if p]
            try:
                if parts == ["health"] and method == "GET":
                    return self.reply(200, service.health())
                if parts[:2] == ["v1", "files"]:
                    return self.files(method, parts[2:], query)
                if parts[:2] == ["v1", "predictions"]:
                    return self.predictions(method, parts[2:])
            except BadInput as exc:
                return self.reply(400, {"error": str(exc), "error_code": "bad_input"})
            except Exception as exc:
                log(f"error {type(exc).__name__}: {exc}")
                return self.reply(500, {"error": f"{type(exc).__name__}: {exc}"[:500], "error_code": "server_error"})
            self.reply(404, {"error": "not found", "error_code": "not_found"})

        def read_body(self) -> bytes:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_UPLOAD:
                raise BadInput("upload too large")
            return self.rfile.read(length)

        def files(self, method, rest, query):
            if method == "POST" and not rest:
                body = self.read_body()
                ctype = self.headers.get("Content-Type", "")
                if ctype.startswith("multipart/form-data"):
                    name, data = parse_multipart(ctype, body)
                else:
                    name, data = (query.get("name") or ["file"])[0], body
                return self.reply(201, service.add_file(name, data))
            entry = service.file(rest[0]) if len(rest) == 1 else None
            if not entry:
                return self.reply(404, {"error": "file not found or expired", "error_code": "not_found"})
            if method == "GET":
                data = Path(entry["path"]).read_bytes()
                self.reply(200, body=data, ctype="application/octet-stream",
                           headers={"Content-Disposition": f'attachment; filename="{entry["name"]}"'})
                if query.get("delete") == ["1"]:
                    service.delete_file(entry["id"])
                return
            if method == "DELETE":
                service.delete_file(entry["id"])
                return self.reply(200, {"deleted": True})
            self.reply(405, {"error": "method not allowed"})

        def predictions(self, method, rest):
            if method == "POST" and not rest:
                try:
                    body = json.loads(self.read_body() or b"{}")
                except json.JSONDecodeError as exc:
                    raise BadInput(f"invalid JSON: {exc}")
                return self.reply(201, service.submit(body))
            prediction = service.predictions.get(rest[0]) if len(rest) == 1 else None
            if not prediction:
                return self.reply(404, {"error": "prediction not found or expired", "error_code": "not_found"})
            if method == "GET":
                return self.reply(200, service.public(prediction))
            if method == "DELETE":
                return self.reply(200, {"canceled": service.cancel(prediction["id"])})
            self.reply(405, {"error": "method not allowed"})

        def do_GET(self): self.dispatch("GET")
        def do_POST(self): self.dispatch("POST")
        def do_DELETE(self): self.dispatch("DELETE")

    return Handler


def set_process_name(name: str) -> None:
    try:
        import ctypes

        ctypes.CDLL(None).prctl(15, name.encode()[:15], 0, 0, 0)  # PR_SET_NAME
    except (OSError, AttributeError):
        pass


def serve(service: Service, port: int, token: str | None = None, host: str = "127.0.0.1") -> int:
    pid_file = service.run_dir / "server.pid"
    httpd = ThreadingHTTPServer((host, port), make_handler(service, token))
    httpd.daemon_threads = True
    pid_file.write_text(json.dumps({"pid": os.getpid(), "port": port, "started": time.time()}))
    threading.Thread(target=service.worker_loop, daemon=True, name="gpu-worker").start()
    threading.Thread(target=httpd.serve_forever, daemon=True, name="http").start()
    log(f"listening on {host}:{port}, pid {os.getpid()}, unload {service.unload_s:.0f} s, exit {service.exit_s:.0f} s")
    try:
        while not service.stop.wait(2):
            service.sweep()
    except KeyboardInterrupt:
        pass
    finally:
        service.stop.set()
        httpd.shutdown()
        httpd.server_close()
        service.cleanup()
        try:
            pid_file.unlink()
        except FileNotFoundError:
            pass
    log("stopped")
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse
    import signal

    parser = argparse.ArgumentParser(description="GPU host inference API (loopback only)")
    parser.add_argument("--port", type=int, default=int(os.getenv("GPU_API_PORT", "18770")))
    args = parser.parse_args(argv)
    common.setup_env()
    common.lower_priority()
    set_process_name("genjutsu-gpu-api")
    service = Service(common.gpu_dir())
    pid_file = service.run_dir / "server.pid"
    if pid_file.is_file():
        try:
            old = json.loads(pid_file.read_text())["pid"]
            os.kill(old, 0)
            print(f"a server is already running (pid {old})", file=sys.stderr)
            return 1
        except (OSError, ValueError, KeyError):
            pass  # stale pid file
    signal.signal(signal.SIGTERM, lambda *_: service.stop.set())
    return serve(service, args.port, os.getenv("GPU_API_TOKEN") or None)


if __name__ == "__main__":
    sys.exit(main())
