"""Enhancor Seedance 2.5 (the upstream behaviour, moved here unchanged)."""
from pathlib import Path
import json, requests
from .base import (GenerationProvider, Capabilities, PollResult, SubmissionRejected, Check)


class EnhancorProvider(GenerationProvider):
    name = 'enhancor'
    response_has_request_id = True   # the app and CLI resume from its raw `requestId`

    def _base(self):
        return self.env.setting('ENHANCOR_BASE_URL', 'https://apireq.enhancor.ai/api/seedance2.5/v1').rstrip('/') + '/'

    def capabilities(self):
        return Capabilities(reference_video=True, max_reference_images=2, draft=True, resolutions=('1080p',),
                            final_resolutions=('1080p',), min_duration=4, max_duration=30, needs_webhook=True,
                            accepts_real_faces=True)  # pass_faces=True is part of every Enhancor request

    def preflight(self):
        self.env.credential('ENHANCOR_API_KEY')
        if not self.env.setting('ENHANCOR_WEBHOOK_URL'):
            raise ValueError('Webhook URL required. Install cloudflared and restart the app, or set ENHANCOR_WEBHOOK_URL.')

    def doctor(self):
        checks = []
        value = self.env.setting('ENHANCOR_API_KEY').strip()
        valid = bool(value) and not any(x in value.lower() for x in ['your_key', 'your_api', 'replace_me', 'placeholder'])
        checks.append(Check('ENHANCOR_API_KEY', valid, 'ENHANCOR_API_KEY: ' + ('configured (not verified remotely)' if valid else 'MISSING')))
        if self.env.setting('MEDIA_HOST', 'tmpfiles') not in ['tmpfiles', 'catbox']:
            checks.append(Check('MEDIA_HOST', False, 'MEDIA_HOST must be tmpfiles or catbox', fatal=True))
        if not self.env.setting('ENHANCOR_WEBHOOK_URL'):
            checks.append(Check('ENHANCOR_WEBHOOK_URL', False, 'WEBHOOK: not running/configured — install cloudflared and launch the app, or set your own HTTPS callback'))
        return checks

    def upload(self, path):
        return self.env.upload(path)

    def build_request(self, job, urls):
        prompt, folder = job.prompt, Path(job.folder)
        sirio = job.references[0]
        tein = job.references[1] if len(job.references) > 1 else None
        prepared = job.video
        p = {'resolution': '1080p', 'output_format': 'mp4', 'pass_faces': True, 'webhook_url': self.env.setting('ENHANCOR_WEBHOOK_URL', '')}
        p.update(mode='edit', is_draft=True, aspect_ratio='adaptive', videos=[urls[0]], audios=[], images=urls[1:], prompt=prompt.replace('@Audio1', 'the audio embedded in @Video1') + '\n@Video1 contains the colored-depth composite and embedded isolated vocals pitched up three semitones, without the music bed; use its embedded audio for performance and lip synchronization. ' + ('@Image1 is the primary character reference; @Image2 is the secondary character reference.' if tein else '@Image1 is the only replacement character reference.')); p.pop('duration', None)
        if Path(sirio).suffix.lower() in ['.mp4', '.mov', '.webm']:
            import av
            with av.open(str(prepared)) as source:
                seconds = float(source.duration / av.time_base)
                stream = source.streams.video[0]
                ratio = stream.width / stream.height
            ratios = {'16:9': 16 / 9, '9:16': 9 / 16, '1:1': 1, '4:3': 4 / 3, '3:4': 3 / 4, '21:9': 21 / 9}
            p.update(mode='multi_reference', duration=str(max(4, min(30, round(seconds)))), aspect_ratio=min(ratios, key=lambda r: abs(ratios[r] - ratio)))
            p['videos'] = [urls[0], urls[1]]
            p['images'] = urls[2:]
            p['prompt'] = prompt + '\n@Video1 is the source edit and contains isolated vocals pitched +3 semitones. @Video2 is only the primary character visual identity reference. Use only @Video1 audio for speech timing.'
        if job.mesh_mode is not None:
            description = 'depth composite with face mesh overlay' if job.mesh_mode == 'depth_mesh' else 'original video with face mesh overlay'
            p['prompt'] = p['prompt'].replace('colored-depth composite', description) + ' Remove all face mesh lines and any depth colors from the output; use them only for expression and lip motion guidance. Replace the entire original identity including hair. Preserve the exact embedded dialogue and timing.'
        return p

    def submit(self, payload, final=False):
        r = requests.post(self._base() + 'queue', headers={'x-api-key': self.env.credential('ENHANCOR_API_KEY')}, json=payload, timeout=90)
        if final:
            r.raise_for_status()
        elif not r.ok:
            raise SubmissionRejected(f'Enhancor rejected submission ({r.status_code}): {r.text[:500]}', r.text)
        j = r.json()
        return j.get('requestId'), j

    def poll(self, task_id):
        r = requests.post(self._base() + 'status', headers={'x-api-key': self.env.credential('ENHANCOR_API_KEY')}, json={'request_id': task_id}, timeout=30)
        r.raise_for_status(); j = r.json()
        if j.get('status') == 'COMPLETED':
            url = j.get('result')
            if not isinstance(url, str) or not url.startswith('https://'):
                raise ValueError('Seedance result URL missing.')
            return PollResult('done', url, raw=j)
        if j.get('status') in ['FAILED', 'CANCELED']:
            return PollResult('failed', error=str(j.get('error') or j.get('status')), raw=j)
        return PollResult('pending', raw=j)

    def is_draft_request(self, payload):
        return bool(payload.get('is_draft'))

    def build_final(self, draft_task_id, original, resolution=None):
        # Enhancor reuses the draft's original parameters.
        return {'draft_id': draft_task_id, 'webhook_url': original['webhook_url']}
