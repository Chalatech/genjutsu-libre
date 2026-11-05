"""OpenRouter video generation (`POST /api/v1/videos`), tested with ByteDance Seedance 2.5.

Flow (docs read 2026-11-04): submit -> `{id, polling_url, status}`; poll `GET /api/v1/videos/{id}` until
`completed` (or `failed`/`cancelled`/`expired`); download `unsigned_urls[0]` with the API key in the
`Authorization` header; `usage.cost` (USD) is the real price.

Capabilities come from the free `GET /api/v1/videos/models` (the general `/api/v1/models` catalog does NOT
list video models), then from KNOWN_MODELS when that is unreachable. A model takes a REFERENCE VIDEO only
when OpenRouter documents it: "Audio and video references are only honored by providers that support them
(including BytePlus Seedance generation 2 and newer)". Anything else stops before any upload.

OpenRouter has no file storage, so media is hosted elsewhere (`OPENROUTER_MEDIA_HOST`): Kie's file API or tmpfiles.
There is no draft concept: the "final pass" is a second, separately billed generation at a higher resolution.
"""
import math, re, requests
from pathlib import Path
from .base import (GenerationProvider, Capabilities, PollResult, SubmissionRejected, ProviderError, Unsupported, Check,
                   IMAGE_SUFFIXES, VIDEO_SUFFIXES)
from .kie import KieProvider, REF_VIDEO_SUFFIXES, REF_IMAGE_SUFFIXES, MIN_PIXELS, MAX_PIXELS

BASE = 'https://openrouter.ai/api/v1'
DEFAULT_MODEL = 'bytedance/seedance-2.5'
FPS = 24                      # token count assumes 24 fps (ByteDance: width x height x fps x seconds / 1024)

# Used when /videos/models cannot be reached. Add a row only after reading the model's page.
KNOWN_MODELS = {
    'bytedance/seedance-2.5': {
        'reference_video': True, 'checked': '2026-11-04', 'durations': list(range(4, 31)),
        'resolutions': ['480p', '720p'], 'aspect_ratios': ['16:9', '4:3', '1:1', '3:4', '9:16', '21:9'],
        'sizes': ['854x480', '752x560', '640x640', '560x752', '480x854', '992x432',
                  '1280x720', '1112x834', '960x960', '834x1112', '720x1280', '1470x630'],
        'pricing': {'video_tokens': '0.0000107', 'video_tokens_with_video_input': '0.0000064'}},
    'alibaba/wan-3.0': {'reference_video': False, 'checked': '2026-11-04'},
}
TERMINAL_FAILURES = ('failed', 'cancelled', 'canceled', 'expired')


def _supports_reference_video(model_id):
    """Seedance generation 2 and newer, per OpenRouter's documentation."""
    m = re.search(r'seedance-(\d+)', model_id or '')
    return bool(m) and int(m.group(1)) >= 2


def _ratio(text):
    a, b = text.split(':')
    return float(a) / float(b)


class OpenRouterProvider(GenerationProvider):
    name = 'openrouter'
    needs_media_info = True
    max_polls = 360

    def __init__(self, env):
        super().__init__(env)
        self._info = None
        self._last = {}

    # --- configuration ------------------------------------------------------------------------
    def _api(self):
        return self.env.setting('OPENROUTER_BASE_URL', BASE).rstrip('/')

    def _model_id(self):
        return (self.env.setting('OPENROUTER_MODEL', '') or DEFAULT_MODEL).strip()

    def _key_name(self):
        for name in ('OPENROUTER_API_KEY', 'OPENROUTER_KEY'):
            if self.env.setting(name, '').strip():
                return name
        return None

    def _key(self):
        name = self._key_name()
        if not name:
            raise ValueError('Set OPENROUTER_API_KEY (or OPENROUTER_KEY) in this project’s .env file.')
        return self.env.setting(name).strip()

    def _headers(self):
        return {'Authorization': 'Bearer ' + self._key()}

    def _resolution(self):
        return (self.env.setting('OPENROUTER_RESOLUTION', '') or '480p').strip()

    def _int(self, name, default):
        try:
            return int(float(self.env.setting(name, '') or default))
        except ValueError:
            raise ValueError(f'{name} must be a number.')

    # --- capabilities -------------------------------------------------------------------------
    def _model_info(self):
        """Normalized facts about the configured model: {'source': 'api'|'table'|None, ...}. One free GET per provider."""
        if self._info is not None:
            return self._info
        model, error, entry = self._model_id(), '', None
        try:
            r = requests.get(self._api() + '/videos/models', timeout=20)
            r.raise_for_status()
            entry = next((m for m in r.json().get('data', []) if m.get('id') == model), None)
        except (requests.RequestException, ValueError, AttributeError) as exc:
            error = f'video model list unreachable ({type(exc).__name__})'
        known = KNOWN_MODELS.get(model)
        if entry:
            self._info = {'source': 'api', 'durations': entry.get('supported_durations') or [],
                          'resolutions': entry.get('supported_resolutions') or [],
                          'aspect_ratios': entry.get('supported_aspect_ratios') or [],
                          'sizes': entry.get('supported_sizes') or [], 'pricing': entry.get('pricing_skus') or {},
                          'reference_video': (known['reference_video'] if known else _supports_reference_video(model)),
                          'error': ''}
        elif known:
            self._info = {'source': 'table', 'durations': known.get('durations', []), 'resolutions': known.get('resolutions', []),
                          'aspect_ratios': known.get('aspect_ratios', []), 'sizes': known.get('sizes', []),
                          'pricing': known.get('pricing', {}), 'reference_video': known['reference_video'],
                          'checked': known['checked'], 'error': error}
        else:
            self._info = {'source': None, 'durations': [], 'resolutions': [], 'aspect_ratios': [], 'sizes': [],
                          'pricing': {}, 'reference_video': False, 'error': error}
        return self._info

    def capabilities(self):
        info = self._model_info()
        res = self._resolution()
        finals = tuple(r for r in info['resolutions'] if self._p(r) > self._p(res))
        durations = info['durations']
        return Capabilities(reference_video=info['reference_video'],
                            max_reference_images=self._int('OPENROUTER_MAX_REF_IMAGES', 30) if info['reference_video'] else 0,
                            draft=bool(finals), resolutions=(res,), final_resolutions=finals,
                            min_duration=min(durations) if durations else 0, max_duration=max(durations) if durations else 0,
                            needs_webhook=False, accepts_real_faces=None)

    @staticmethod
    def _p(resolution):
        m = re.match(r'(\d+)p', str(resolution))
        return int(m.group(1)) if m else 0

    def _explain(self):
        model, info = self._model_id(), self._model_info()
        if info['source'] is None:
            extra = f' ({info["error"]})' if info['error'] else ''
            return f'Model {model!r} is not in OpenRouter\'s video model list or in this provider\'s table{extra}; assuming reference video is not supported.'
        if not info['reference_video']:
            return f'Model {model!r} does not accept a reference video (OpenRouter honors video references only for Seedance generation 2 and newer).'
        return ''

    def preflight(self):
        if not self.capabilities().reference_video:
            raise Unsupported('OpenRouter cannot run this workflow: ' + self._explain() +
                              ' The workflow sends a reference video; pick another model or GEN_BACKEND=kie or enhancor. Nothing was uploaded.')
        self._key()
        self._host()

    def doctor(self):
        key = self._key_name()
        checks = [Check('OPENROUTER_API_KEY', bool(key), 'OPENROUTER_API_KEY: ' + (f'configured as {key} (not verified remotely)' if key else 'MISSING (or OPENROUTER_KEY)'))]
        ok = self.capabilities().reference_video
        checks.append(Check('OPENROUTER_MODEL', ok, f'OPENROUTER_MODEL {self._model_id()}: ' + ('accepts reference video' if ok else self._explain())))
        try:
            host = self._host()
            hosted = host == 'tmpfiles' or bool(self.env.setting('KIE_API_KEY', '').strip())
            checks.append(Check('OPENROUTER_MEDIA_HOST', hosted, f'OPENROUTER_MEDIA_HOST: {host}' + ('' if hosted else ' needs KIE_API_KEY')))
        except ValueError as exc:
            checks.append(Check('OPENROUTER_MEDIA_HOST', False, str(exc)))
        return checks

    # --- validation ---------------------------------------------------------------------------
    def validate(self, job):
        problems = []
        if job.video_info is None:
            raise ValueError('OpenRouter validation needs probed media info for the composite video.')
        min_px, max_px = self._int('OPENROUTER_MIN_PIXELS', MIN_PIXELS), self._int('OPENROUTER_MAX_PIXELS', MAX_PIXELS)
        max_total, max_images = self._int('OPENROUTER_MAX_REF_SECONDS', 30), self.capabilities().max_reference_images
        videos, images = [('composite video', job.video_info)], []
        for ref, info in zip(job.references, job.reference_infos):
            (videos if info.get('kind') == 'video' else images).append((Path(ref).name, info))
        if len(videos) > 10:
            problems.append('more than 10 reference videos')
        if len(images) > max_images:
            problems.append(f'more than {max_images} reference images')
        total = 0.0
        for label, v in videos:
            KieProvider._check_visual(label, v, problems, REF_VIDEO_SUFFIXES)
            px = v['width'] * v['height']
            if not min_px <= px <= max_px:
                problems.append(f'{label}: width x height must be {min_px}-{max_px} px total (got {v["width"]}x{v["height"]} = {px})')
            if not (23.99 <= v['fps'] <= 60.01):
                problems.append(f'{label}: frame rate must be 24-60 fps (got {v["fps"]:.2f})')
            if not 2 <= v['duration'] <= 30:
                problems.append(f'{label}: duration must be 2-30 s (got {v["duration"]:.2f})')
            if v['bytes'] > 200 * 1024 * 1024:
                problems.append(f'{label}: larger than 200 MB')
            total += v['duration']
        if total > max_total + 0.0001:
            problems.append(f'total reference video duration must be <= {max_total} s (got {total:.2f})')
        for label, im in images:
            KieProvider._check_visual(label, im, problems, REF_IMAGE_SUFFIXES)
            if im['bytes'] >= 30 * 1024 * 1024:
                problems.append(f'{label}: image must be under 30 MB')
        if problems:
            raise ValueError('OpenRouter Seedance rejected the inputs locally (nothing uploaded):\n- ' + '\n- '.join(problems))

    # --- request shape ------------------------------------------------------------------------
    def _duration(self, seconds):
        durations = self._model_info()['durations'] or list(range(4, 31))
        return min(durations, key=lambda d: (abs(d - round(seconds)), d))

    def _aspect(self, info):
        ratios = self._model_info()['aspect_ratios'] or ['16:9']
        ratio = info['width'] / info['height']
        return min(ratios, key=lambda a: abs(math.log(_ratio(a) / ratio)))

    def _size(self, aspect, resolution):
        """(width, height) of the output: the listed size of that aspect closest in area to the resolution."""
        p = self._p(resolution) or 480
        target = p * p * 16 / 9
        sizes = []
        for s in self._model_info()['sizes']:
            w, h = (int(x) for x in s.split('x'))
            if abs(w / h - _ratio(aspect)) < 0.02:
                sizes.append((w, h))
        if sizes:
            return min(sizes, key=lambda s: abs(s[0] * s[1] - target))
        r = _ratio(aspect)
        return (round(p * r), p) if r >= 1 else (p, round(p / r))

    # --- cost ---------------------------------------------------------------------------------
    def _rates(self):
        pricing = self._model_info()['pricing']
        try:
            plain = float(pricing.get('video_tokens') or 0)
            with_video = float(pricing.get('video_tokens_with_video_input') or plain)
        except ValueError:
            return 0.0, 0.0
        return plain, with_video

    @staticmethod
    def _tokens(width, height, seconds):
        return width * height * FPS * seconds / 1024

    def _estimate(self, info, in_seconds, resolution, extra_video_seconds=0.0):
        """ESTIMATE only (ByteDance token formula at 24 fps, input + output seconds billed at the video-input rate);
        the real price is `usage.cost`, stored with the job after completion."""
        plain, with_video = self._rates()
        if not with_video:
            return None
        out_w, out_h = self._size(self._aspect(info), resolution)
        out_seconds = self._duration(in_seconds)
        tokens = self._tokens(out_w, out_h, out_seconds) + self._tokens(info['width'], info['height'], in_seconds + extra_video_seconds)
        return tokens * with_video

    def estimate_cost(self, job):
        extra = sum(i.get('duration', 0) for i in job.reference_infos if i.get('kind') == 'video')
        return self._estimate(job.video_info, job.seconds, self._resolution(), extra)

    def estimate_final_cost(self, seconds):
        finals = self.capabilities().final_resolutions
        plain, with_video = self._rates()
        if not finals or not with_video:
            return None
        # Only the length is known here: assume the 16:9 composite of this workflow (960x540 input).
        return self._estimate({'width': 960, 'height': 540}, float(seconds), finals[-1])

    # --- transport ----------------------------------------------------------------------------
    def _host(self):
        host = (self.env.setting('OPENROUTER_MEDIA_HOST', '') or '').strip().lower()
        if not host:
            host = 'kie' if self.env.setting('KIE_API_KEY', '').strip() else 'tmpfiles'
        if host not in ('kie', 'tmpfiles'):
            raise ValueError('OPENROUTER_MEDIA_HOST must be kie or tmpfiles.')
        return host

    def upload(self, path):
        """OpenRouter has no file storage. Kie's file API (24 h) or tmpfiles; never catbox, never falling back."""
        if self._host() == 'kie':
            return KieProvider(self.env).upload(path)
        import media_host
        return media_host.upload(path, hosts=['tmpfiles'])

    def build_request(self, job, urls):
        refs = list(job.references)
        videos = [urls[0]] + [u for ref, u in zip(refs, urls[1:]) if Path(ref).suffix.lower() in VIDEO_SUFFIXES]
        images = [u for ref, u in zip(refs, urls[1:]) if Path(ref).suffix.lower() in IMAGE_SUFFIXES]
        prompt = KieProvider(self.env).build_request(job, urls)['input']['prompt']   # same @Video1/@Image notes
        resolution = self._resolution()
        return {'model': self._model_id(), 'prompt': prompt, 'duration': self._duration(job.seconds),
                'resolution': resolution, 'aspect_ratio': self._aspect(job.video_info), 'generate_audio': False,
                'input_references': [{'type': 'video_url', 'video_url': {'url': u}} for u in videos] +
                                    [{'type': 'image_url', 'image_url': {'url': u}} for u in images]}

    def submit(self, payload, final=False):
        r = requests.post(self._api() + '/videos', headers=self._headers(), json=payload, timeout=90)
        if not r.ok:
            raise SubmissionRejected(f'OpenRouter rejected submission (HTTP {r.status_code}): {r.text[:500]}', r.text)
        j = r.json()
        return j.get('id'), j

    def poll(self, task_id):
        r = requests.get(f'{self._api()}/videos/{task_id}', headers=self._headers(), timeout=30)
        r.raise_for_status()
        j = r.json()
        status = str(j.get('status') or '').lower()
        self._last = j
        if status == 'completed':
            urls = j.get('unsigned_urls') or j.get('urls') or []
            if not urls or not str(urls[0]).startswith('https://'):
                raise ValueError('OpenRouter result URL missing.')
            return PollResult('done', urls[0], raw=j)
        if status in TERMINAL_FAILURES:
            return PollResult('failed', error=str(j.get('error') or status), raw=j)
        return PollResult('pending', raw=j)   # pending | in_progress | processing

    def download(self, url, dest):
        dest = Path(dest)
        headers = self._headers() if url.startswith(self._api() + '/') else {}   # signed URLs need no key
        r = requests.get(url, headers=headers, timeout=300)
        r.raise_for_status()
        dest.write_bytes(r.content)
        usage = self._last.get('usage')
        if usage:
            import json
            (dest.parent / 'provider-usage.json').write_text(json.dumps({'cost_usd': usage.get('cost'), 'usage': usage,
                                                                          'generation_id': self._last.get('generation_id')}, indent=2))

    # --- second pass: a NEW generation at a higher resolution (billed again) ----------------------
    def is_draft_request(self, payload):
        finals = self.capabilities().final_resolutions
        return bool(finals) and self._p(payload.get('resolution')) < self._p(finals[0])

    def build_final(self, draft_task_id, original, resolution=None):
        finals = self.capabilities().final_resolutions
        resolution = resolution or (finals[-1] if finals else None)
        if resolution not in finals:
            raise ValueError(f'HD resolution must be one of {", ".join(finals) or "(none)"}.')
        return {**original, 'resolution': resolution}
