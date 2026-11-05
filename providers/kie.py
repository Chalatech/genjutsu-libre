"""Kie.ai `bytedance/seedance-2-5` (multimodal reference scenario) and its HD completion.

Limits come from docs.kie.ai/market/bytedance/seedance-2-5 (read 2026-11-04). Kie answers HTTP 200 even
on failure, so every response is judged by its `code` field. Polling only; no webhook.
"""
from pathlib import Path
import json, uuid, requests
from .base import (GenerationProvider, Capabilities, PollResult, SubmissionRejected, ProviderError, Check,
                   IMAGE_SUFFIXES, VIDEO_SUFFIXES)

USD_PER_CREDIT = 0.005
REF_VIDEO_SUFFIXES = ('.mp4', '.mov', '.mkv')
REF_IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg', '.webp')
MIN_PIXELS, MAX_PIXELS = 409_600, 927_408
FINAL_MODEL = 'bytedance/seedance-2-5-draft'


class KieProvider(GenerationProvider):
    name = 'kie'
    needs_media_info = True
    max_polls = 540    # tasks may sit in `waiting` 30+ min before failing without charge

    def _api(self):
        return self.env.setting('KIE_BASE_URL', 'https://api.kie.ai').rstrip('/')

    def _headers(self):
        return {'Authorization': 'Bearer ' + self.env.credential('KIE_API_KEY')}

    def _model(self):
        return self.env.setting('KIE_MODEL', 'bytedance/seedance-2-5')

    def _is_mini(self):
        return self._model().endswith('seedance-2-mini')

    def capabilities(self):
        if self._is_mini():
            # Seedance 2 mini on Kie: no draft/final pass, 4-15 s, up to 3 reference videos of 2-15 s.
            return Capabilities(reference_video=True, max_reference_images=30, draft=False, resolutions=('480p', '720p'),
                                final_resolutions=(), min_duration=4, max_duration=15,
                                needs_webhook=False, accepts_real_faces=None)
        return Capabilities(reference_video=True, max_reference_images=30, draft=True, resolutions=('480p',),
                            final_resolutions=('720p', '1080p'), min_duration=4, max_duration=30,
                            needs_webhook=False, accepts_real_faces=None)

    def preflight(self):
        self.env.credential('KIE_API_KEY')

    def doctor(self):
        value = self.env.setting('KIE_API_KEY').strip()
        valid = bool(value) and not any(x in value.lower() for x in ['your_key', 'your_api', 'replace_me', 'placeholder'])
        return [Check('KIE_API_KEY', valid, 'KIE_API_KEY: ' + ('configured (not verified remotely)' if valid else 'MISSING'))]

    # --- validation ---------------------------------------------------------------------------
    @staticmethod
    def _check_visual(label, info, problems, suffixes):
        w, h = info['width'], info['height']
        if info['suffix'] not in suffixes:
            problems.append(f'{label}: format {info["suffix"] or "?"} not accepted (use {"/".join(s[1:] for s in suffixes)})')
        if not (300 <= w <= 6000 and 300 <= h <= 6000):
            problems.append(f'{label}: sides must be 300-6000 px (got {w}x{h})')
        if h and not (0.4 <= w / h <= 2.5):
            problems.append(f'{label}: aspect ratio {w / h:.2f} outside 0.4-2.5')

    def validate(self, job):
        problems = []
        if len(job.prompt) > 20480:
            problems.append('prompt longer than 20480 characters')
        if job.video_info is None:
            raise ValueError('Kie validation needs probed media info for the composite video.')
        videos = [('composite video', job.video_info)]
        images = []
        for ref, info in zip(job.references, job.reference_infos):
            if info.get('kind') == 'video':
                videos.append((Path(ref).name, info))
            else:
                images.append((Path(ref).name, info))
        if len(videos) > 10:
            problems.append('more than 10 reference videos')
        if len(images) > 30:
            problems.append('more than 30 reference images')
        total = 0.0
        for label, v in videos:
            self._check_visual(label, v, problems, REF_VIDEO_SUFFIXES)
            px = v['width'] * v['height']
            if not MIN_PIXELS <= px <= MAX_PIXELS:
                problems.append(f'{label}: width x height must be {MIN_PIXELS}-{MAX_PIXELS} px total (got {v["width"]}x{v["height"]} = {px})')
            if not (23.99 <= v['fps'] <= 60.01):
                problems.append(f'{label}: frame rate must be 24-60 fps (got {v["fps"]:.2f})')
            if not 2 <= v['duration'] <= 30:
                problems.append(f'{label}: duration must be 2-30 s (got {v["duration"]:.2f})')
            if v['bytes'] > 200 * 1024 * 1024:
                problems.append(f'{label}: larger than 200 MB')
            total += v['duration']
        if total > 30.0001:
            problems.append(f'total reference video duration must be <= 30 s (got {total:.2f})')
        for label, im in images:
            self._check_visual(label, im, problems, REF_IMAGE_SUFFIXES)
            if im['bytes'] >= 30 * 1024 * 1024:
                problems.append(f'{label}: image must be under 30 MB')
        if problems:
            raise ValueError('Kie Seedance 2.5 rejected the inputs locally (nothing uploaded):\n- ' + '\n- '.join(problems))

    # --- cost ---------------------------------------------------------------------------------
    def _unit(self):
        try:
            return float(self.env.setting('KIE_UNIT_CREDITS', '17'))
        except ValueError:
            raise ValueError('KIE_UNIT_CREDITS must be a number.')

    def estimate_cost(self, job):
        """credits per billable second x (reference seconds + output seconds) x $0.005.
        Output uses `duration: -1`, i.e. the length of the source video."""
        ref = job.seconds + sum(i.get('duration', 0) for i in job.reference_infos if i.get('kind') == 'video')
        return self._unit() * (ref + job.seconds) * USD_PER_CREDIT

    def estimate_final_cost(self, seconds):
        # Lower bound: the per-resolution price of the final pass is not published (measured in F6).
        return self._unit() * float(seconds) * USD_PER_CREDIT

    # --- transport ----------------------------------------------------------------------------
    def upload(self, path):
        path = Path(path)
        with path.open('rb') as f:
            r = requests.post('https://kieai.redpandaai.co/api/file-stream-upload', headers=self._headers(),
                              files={'file': (path.name, f)},
                              data={'uploadPath': 'genjutsu', 'fileName': uuid.uuid4().hex + path.suffix.lower()}, timeout=300)
        r.raise_for_status()
        j = r.json()
        url = (j.get('data') or {}).get('downloadUrl')
        if not j.get('success') or not url or not url.startswith('https://'):
            raise ProviderError(f'Kie file upload failed (code {j.get("code")}): {str(j.get("msg"))[:300]}')
        return url

    def build_request(self, job, urls):
        refs = list(job.references)
        video_urls = [urls[0]] + [u for ref, u in zip(refs, urls[1:]) if Path(ref).suffix.lower() in VIDEO_SUFFIXES]
        image_urls = [u for ref, u in zip(refs, urls[1:]) if Path(ref).suffix.lower() in IMAGE_SUFFIXES]
        from audio_workflow import embed_vocals
        from config import setting
        if not embed_vocals():
            first = '@Video1 is the source edit: a silent colored-depth composite; follow its motion and timing.'
        elif str(setting('SEEDANCE_GUIDE_AUDIO', '') or '').strip():
            first = '@Video1 is the source edit: a colored-depth composite with the performer\'s guide vocal embedded; use its embedded audio for lip synchronization.'
        else:
            first = '@Video1 is the source edit: a colored-depth composite with isolated vocals pitched +3 semitones embedded; use its embedded audio for performance and lip synchronization.'
        notes = [first]
        n_video = 1
        for ref in refs:
            if Path(ref).suffix.lower() in VIDEO_SUFFIXES:
                n_video += 1
                notes.append(f'@Video{n_video} is only a character visual identity reference.')
        if len(image_urls) > 2:
            # Many references (characters, props, a location): the prompt says what each one is.
            notes.append(f'@Image1 to @Image{len(image_urls)} are the reference images described in the prompt.')
        else:
            for i in range(len(image_urls)):
                notes.append(f'@Image{i + 1} is the {"primary" if i == 0 else "secondary"} character reference.' if len(image_urls) > 1 else '@Image1 is the only replacement character reference.')
        prompt = job.prompt.replace('@Audio1', 'the audio embedded in @Video1') + '\n' + ' '.join(notes)
        if job.mesh_mode is not None:
            desc = 'depth composite with face mesh overlay' if job.mesh_mode == 'depth_mesh' else 'original video with face mesh overlay'
            prompt = prompt.replace('colored-depth composite', desc) + ' Remove all face mesh lines and any depth colors from the output; use them only for expression and lip motion guidance. Replace the entire original identity including hair. Preserve the exact embedded dialogue and timing.'
        body = {'model': self._model(), 'input': {
            'prompt': prompt, 'reference_video_urls': video_urls, 'reference_image_urls': image_urls,
            'draft': True, 'resolution': '480p', 'duration': -1, 'generate_audio': False,
            'aspect_ratio': 'adaptive', 'output_format': 'mp4'}}
        if self._is_mini():
            for key in ('draft', 'output_format'):      # not in the Seedance 2 mini schema
                body['input'].pop(key, None)
        return body

    def submit(self, payload, final=False):
        r = requests.post(self._api() + '/api/v1/jobs/createTask', headers=self._headers(), json=payload, timeout=90)
        if not r.ok:
            raise SubmissionRejected(f'Kie rejected submission (HTTP {r.status_code}): {r.text[:500]}', r.text)
        j = r.json()
        if j.get('code') != 200:
            raise SubmissionRejected(f'Kie rejected submission (code {j.get("code")}): {str(j.get("msg"))[:500]}', r.text)
        return (j.get('data') or {}).get('taskId'), j

    def poll(self, task_id):
        r = requests.get(self._api() + '/api/v1/jobs/recordInfo', headers=self._headers(), params={'taskId': task_id}, timeout=30)
        r.raise_for_status()
        j = r.json()
        if j.get('code') != 200:
            raise ProviderError(f'Kie status check failed (code {j.get("code")}): {str(j.get("msg"))[:300]}')
        d = j.get('data') or {}
        state = d.get('state')
        if state == 'success':
            try:
                urls = json.loads(d.get('resultJson') or '{}').get('resultUrls') or []
            except ValueError:
                urls = []
            if not urls or not str(urls[0]).startswith('https://'):
                raise ValueError('Seedance result URL missing.')
            return PollResult('done', urls[0], raw=j)
        if state == 'fail':
            return PollResult('failed', error=str(d.get('failMsg') or d.get('failCode') or 'fail'), raw=j)
        return PollResult('pending', raw=j)   # waiting | queuing | generating

    # --- HD -----------------------------------------------------------------------------------
    def is_draft_request(self, payload):
        return bool((payload.get('input') or {}).get('draft'))

    def build_final(self, draft_task_id, original, resolution=None):
        resolution = resolution or self.env.setting('KIE_HD_RESOLUTION', '1080p')
        if resolution not in self.capabilities().final_resolutions:
            raise ValueError(f'HD resolution must be one of {", ".join(self.capabilities().final_resolutions)}.')
        return {'model': FINAL_MODEL, 'input': {'draft_task_id': draft_task_id, 'resolution': resolution, 'output_format': 'mp4'}}
