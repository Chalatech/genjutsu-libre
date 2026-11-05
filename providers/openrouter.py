"""OpenRouter: capabilities are declared per configured model (`OPENROUTER_MODEL`), never assumed.

This workflow needs a model that takes a REFERENCE VIDEO and returns a video. The free public catalog
(`/api/v1/models`) does NOT list video models at all, so an absent entry proves nothing. Capabilities come
from, in order: the catalog when it does list the model, then KNOWN_MODELS below (read from the model's
public page), otherwise "unknown" which is treated as unsupported. At 2026-11-04 OpenRouter's Wan 3.0 accepts
reference IMAGES only, so every run stops with a clear error before anything is uploaded. Submission is
implemented only for what is confirmed to work: nothing yet, so `submit` refuses.
"""
import requests
from .base import GenerationProvider, Capabilities, PollResult, Unsupported, Check

CATALOG = 'https://openrouter.ai/api/v1/models'

# Models the catalog does not list, as read from their public pages. Add a row when one is verified.
KNOWN_MODELS = {
    'alibaba/wan-3.0': {'reference_video': False, 'reference_images': True, 'checked': '2026-11-04'},
}


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

    def _declared(self):
        """('catalog'|'table'|None, reference_video, reference_images) for the configured model."""
        if not self._model_id():
            return None, False, False
        models, _ = self._models()
        m = models.get(self._model_id())
        if m:
            arch = m.get('architecture') or {}
            takes, makes = arch.get('input_modalities') or [], arch.get('output_modalities') or []
            return 'catalog', 'video' in takes and 'video' in makes, 'video' in makes and 'image' in takes
        known = KNOWN_MODELS.get(self._model_id())
        if known:
            return 'table', known['reference_video'], known['reference_images']
        return None, False, False

    def capabilities(self):
        _, video, images = self._declared()
        return Capabilities(reference_video=video, max_reference_images=1 if images else 0,   # exact image limit not verified
                            draft=False, resolutions=(), final_resolutions=(), needs_webhook=False,
                            accepts_real_faces=None)

    def _explain(self):
        models, error = self._models()
        model = self._model_id()
        if not model:
            return 'Set OPENROUTER_MODEL in .env.'
        source, _, _ = self._declared()
        if source == 'table':
            return f'Model {model!r} accepts reference images only, not a reference video (checked {KNOWN_MODELS[model]["checked"]}).'
        if source is None:
            extra = f' ({error}; the catalog does not list video models anyway)' if error else ' (the public catalog does not list video models)'
            return f'Model {model!r} is not in the OpenRouter catalog or in this provider\'s table of known video models{extra}; assuming video reference is not supported.'
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
