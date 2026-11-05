"""Generation provider registry. Pick one with `GEN_BACKEND` in .env (default: enhancor)."""
from importlib import import_module
from .base import (GenerationProvider, Capabilities, Job, PollResult, Check, Env, ProviderError,
                   SubmissionRejected, Unsupported)

_REGISTRY = {
    'enhancor': ('providers.enhancor', 'EnhancorProvider'),
    'kie': ('providers.kie', 'KieProvider'),
    'openrouter': ('providers.openrouter', 'OpenRouterProvider'),
}


def register(name, module, cls):
    """Add a provider (module path + class name); it only has to implement GenerationProvider."""
    _REGISTRY[name] = (module, cls)


def available():
    return sorted(_REGISTRY)


def default_env():
    from config import setting, credential
    from media_host import upload
    return Env(setting=setting, credential=credential, upload=upload)


def get_provider(name=None, env=None):
    env = env or default_env()
    name = (name or env.setting('GEN_BACKEND', 'enhancor') or 'enhancor').strip().lower()
    if name not in _REGISTRY:
        raise ValueError(f'Unknown GEN_BACKEND {name!r}. Available: {", ".join(available())}.')
    module, cls = _REGISTRY[name]
    return getattr(import_module(module), cls)(env)
