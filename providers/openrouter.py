"""OpenRouter: capabilities come from its free public model catalog, never from assumption.

This workflow needs a model that takes a REFERENCE VIDEO and returns a video. At 2026-11-04 the catalog lists
no video-output model at all (Wan 3.0 on OpenRouter accepts reference images only), so every run stops with a
clear error before anything is uploaded. Submission is implemented only for what the catalog confirms:
nothing yet, so `submit` refuses. If the catalog is unreachable the provider assumes "video reference not
supported" and says so.
"""
import requests
from .base import GenerationProvider, Capabilities, PollResult, Unsupported, Check

CATALOG = 'https://openrouter.ai/api/v1/models'


class OpenRouterProvider(GenerationProvider):
    name = 'openrouter'

    def __init__(self, env):
        super().__init__(env)
        self._catalog = None

    def _models(self):
        """Free GET; returns (models by id, error text or '')."""
        if self._catalog is None:
            try:
                r = requests.get(self.env.setting('OPENROUTER_CATALOG_URL', CATALOG), timeout=20)
                r.raise_for_status()
                self._catalog = ({m['id']: m for m in r.json().get('data', [])}, '')
            except (requests.RequestException, ValueError, KeyError) as exc:
                self._catalog = ({}, f'catalog unreachable ({type(exc).__name__})')
        return self._catalog

    def _model_id(self):
        return self.env.setting('OPENROUTER_MODEL', '').strip()

    def capabilities(self):
        models, error = self._models()
        m = models.get(self._model_id())
        arch = (m or {}).get('architecture') or {}
        takes_video = 'video' in (arch.get('input_modalities') or [])
        makes_video = 'video' in (arch.get('output_modalities') or [])
        takes_images = 'image' in (arch.get('input_modalities') or [])
        return Capabilities(reference_video=bool(m and takes_video and makes_video),
                            max_reference_images=30 if (m and makes_video and takes_images) else 0,
                            draft=False, resolutions=(), final_resolutions=(), needs_webhook=False,
                            accepts_real_faces=None)

    def _explain(self):
        models, error = self._models()
        model = self._model_id()
        if not model:
            return 'Set OPENROUTER_MODEL in .env.'
        if error:
            return f'OpenRouter {error}; assuming video reference is not supported.'
        if model not in models:
            return f'Model {model!r} is not in the OpenRouter catalog.'
        return f'Model {model!r} does not accept a reference video and return a video.'

    def preflight(self):
        if not self.capabilities().reference_video:
            raise Unsupported('OpenRouter cannot run this workflow: ' + self._explain() +
                              ' The workflow sends a reference video; pick GEN_BACKEND=kie or enhancor. Nothing was uploaded.')
        self.env.credential('OPENROUTER_API_KEY')

    def doctor(self):
        if not self._model_id():
            return [Check('OPENROUTER_MODEL', False, 'OPENROUTER_MODEL: MISSING')]
        ok = self.capabilities().reference_video
        key = bool(self.env.setting('OPENROUTER_API_KEY').strip())
        checks = [Check('OPENROUTER_API_KEY', key, 'OPENROUTER_API_KEY: ' + ('configured (not verified remotely)' if key else 'MISSING'))]
        checks.append(Check('OPENROUTER_MODEL', ok, 'OPENROUTER_MODEL: ' + ('accepts reference video' if ok else self._explain())))
        return checks

    def upload(self, path):
        raise Unsupported('OpenRouter submission is not implemented: the catalog confirms no video-generation model with reference video.')

    def build_request(self, job, urls):
        self.upload(None)

    def submit(self, payload, final=False):
        self.upload(None)

    def poll(self, task_id):
        raise Unsupported('OpenRouter submission is not implemented.')
