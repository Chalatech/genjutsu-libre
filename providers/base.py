"""Common contract for generation providers.

A provider knows how to talk to ONE generation service. Everything that must behave the same for
every service (duplicate refusal, never retrying a paid submission, writing `seedance-request.json`
and `provider-task.json` before polling, restoring the original audio) lives in the facade,
`seedance_bridge.py`, not here.
"""
from __future__ import annotations
import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
import requests

IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg', '.webp')
VIDEO_SUFFIXES = ('.mp4', '.mov', '.webm', '.mkv')


class ProviderError(RuntimeError):
    """Any provider failure that is not a rejected submission."""


class SubmissionRejected(ValueError):
    """The service refused a submission. `body` is the raw response text kept for the user."""
    def __init__(self, message, body=''):
        super().__init__(message)
        self.body = body


class Unsupported(ValueError):
    """The chosen provider/model cannot do what this workflow needs. Raised before any upload."""


@dataclass(frozen=True)
class Capabilities:
    reference_video: bool = True            # accepts a video as reference/source input
    max_reference_images: int = 0
    draft: bool = False                     # cheap preview, then a final pass on the same task
    resolutions: tuple = ()                 # for the first (draft) request
    final_resolutions: tuple = ()           # for submit_final
    min_duration: float = 0                 # seconds of requested output
    max_duration: float = 0
    needs_webhook: bool = False
    accepts_real_faces: Optional[bool] = None   # None = unknown until measured


@dataclass
class Job:
    """Everything a provider needs to describe one generation, provider-neutral."""
    folder: Path
    prompt: str
    video: Path                              # prepared composite with isolated vocals embedded
    references: list = field(default_factory=list)   # character references; images or short videos
    mesh_mode: Optional[str] = None          # None unless the face-mesh workflow is active
    video_info: Optional[dict] = None        # filled by the facade when provider.needs_media_info
    reference_infos: list = field(default_factory=list)

    @property
    def seconds(self):
        return float((self.video_info or {}).get('duration', 0))


@dataclass
class PollResult:
    state: str                               # 'pending' | 'done' | 'failed'
    result_url: Optional[str] = None
    error: str = ''
    raw: dict = field(default_factory=dict)


@dataclass
class Check:
    """One line of `doctor.py` output contributed by a provider."""
    name: str
    ok: bool
    message: str
    fatal: bool = False                      # True = dependencies not ready, not just keys missing


@dataclass
class Env:
    """Injected configuration so the facade (and tests) control where settings come from."""
    setting: Callable
    credential: Callable
    upload: Optional[Callable] = None        # generic public-URL media host (used by enhancor)


def probe_video(path):
    import av
    path = Path(path)
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        rate = float(s.average_rate or 0)
        seconds = float(c.duration / av.time_base) if c.duration else 0.0
        return {'width': s.width, 'height': s.height, 'fps': rate, 'duration': seconds,
                'bytes': path.stat().st_size, 'suffix': path.suffix.lower(), 'kind': 'video'}


def probe_image(path):
    from PIL import Image
    path = Path(path)
    with Image.open(path) as im:
        return {'width': im.width, 'height': im.height, 'bytes': path.stat().st_size,
                'suffix': path.suffix.lower(), 'kind': 'image'}


def probe(path):
    return probe_image(path) if Path(path).suffix.lower() in IMAGE_SUFFIXES else probe_video(path)


class GenerationProvider(abc.ABC):
    name = ''
    needs_media_info = False                 # True if validate() needs probed width/height/fps/duration
    poll_interval = 10
    max_polls = 360

    def __init__(self, env: Env):
        self.env = env

    @abc.abstractmethod
    def capabilities(self) -> Capabilities: ...

    def preflight(self):
        """Cheap checks (credentials, webhook, catalog) run before any media is prepared or uploaded."""

    def doctor(self) -> list:
        """`Check`s for doctor.py: what this provider needs, without calling anything paid."""
        return []

    def validate(self, job: Job):
        """Raise ValueError listing every local problem. Runs before any upload."""

    def estimate_cost(self, job: Job):
        """Estimated USD for a draft/first request, or None when the provider has no known price."""
        return None

    def estimate_final_cost(self, seconds: float):
        return None

    @abc.abstractmethod
    def upload(self, path) -> str: ...

    @abc.abstractmethod
    def build_request(self, job: Job, urls: list) -> dict: ...

    @abc.abstractmethod
    def submit(self, payload: dict, final: bool = False):
        """Send ONE paid request, never retried. Returns (task_id or None, raw_response_dict).
        Raises SubmissionRejected when the service refuses it."""

    @abc.abstractmethod
    def poll(self, task_id: str) -> PollResult: ...

    def download(self, url: str, dest):
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        Path(dest).write_bytes(r.content)

    # --- second pass (HD) -------------------------------------------------------------------
    def is_draft_request(self, payload: dict) -> bool:
        return False

    def build_final(self, draft_task_id: str, original: dict, resolution: Optional[str] = None) -> dict:
        raise Unsupported(f'{self.name} has no draft/final flow.')
