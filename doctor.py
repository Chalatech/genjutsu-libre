"""Offline checks; never prints credential values or starts paid calls."""
import argparse,importlib,shutil,sys
from config import setting

def prep_checks():
    """Checks for the active PREP_BACKEND, as providers.Check objects."""
    from providers import Check
    backend=setting('PREP_BACKEND','replicate')
    checks=[]
    if backend=='replicate':
        value=setting('REPLICATE_API_TOKEN').strip()
        valid=bool(value) and not any(x in value.lower() for x in ['your_key','your_api','replace_me','placeholder'])
        checks.append(Check('REPLICATE_API_TOKEN',valid,'REPLICATE_API_TOKEN: '+('configured (not verified remotely)' if valid else 'MISSING')))
    elif backend=='gpu':checks.extend(gpu_checks())
    return checks

def gpu_checks():
    """GPU host: ssh reachable, inference server alive (or startable) and /health through the tunnel (host paths are never echoed)."""
    from providers import Check
    host,directory=setting('GPU_HOST'),setting('GPU_DIR')
    if not host or not directory:return [Check('GPU_HOST',False,'GPU_HOST / GPU_DIR: MISSING (PREP_BACKEND=gpu)',fatal=True)]
    import gpu_backend
    from gpu_api import GpuApi,GpuApiError
    api=GpuApi.from_env(python=setting('GPU_PYTHON') or directory.rstrip('/')+'/.venv/bin/python')
    try:
        ssh=api.ssh('true',timeout=30)
        if ssh.returncode!=0:return [Check('GPU host',False,'GPU host: unreachable over ssh',fatal=True)]
        api.ensure_server(env=gpu_backend.server_env())
        with api.tunnel():health=api.health()
    except GpuApiError as exc:return [Check('GPU host',False,f'GPU host: inference server not ready ({exc.code}: {str(exc)[:120]})',fatal=True)]
    free,need=health.get('memory_available_gb'),float(setting('GPU_MIN_FREE_GB','20'))
    loaded=', '.join(m.split(':',1)[0] if '/' in m else m for m in health.get('models_loaded') or []) or 'none'   # cache keys may carry host paths
    if free is not None and free<need:return [Check('GPU host memory',False,f'GPU host: server up but only {free} GB memory available (need {need:g} GB); stages will wait')]
    return [Check('GPU host',True,f'GPU host: server up through the tunnel, {free} GB memory available, models loaded: {loaded}')]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--require-keys',action='store_true');parser.add_argument('--mesh',action='store_true');args=parser.parse_args()
    failed=sys.version_info<(3,12)
    print('Python:',sys.version.split()[0], '(3.12+ required)')
    for name in ['av','numpy','PIL','requests','fastapi','uvicorn','multipart','gradio_client','scipy']:
        try:importlib.import_module(name);print(name+': OK')
        except Exception as exc:print(name+': FAILED ('+type(exc).__name__+')');failed=True
    binary=shutil.which('rubberband');print('rubberband:', 'OK' if binary else 'MISSING — see docs/SETUP-AGENT.md');failed=failed or not binary
    if args.mesh:
        from pathlib import Path
        import subprocess
        worker=Path(__file__).resolve().parent/'.venv-face/bin/python'
        ready=worker.exists() and subprocess.run([str(worker),'-c','import mediapipe,av,PIL,numpy'],capture_output=True).returncode==0
        print('Face mesh worker: '+('OK' if ready else 'MISSING — run bash setup-face-mesh.sh'))
        failed=failed or not ready
    missing=[]
    # Generation: ask the active provider what it needs (GEN_BACKEND) instead of hard-coding names.
    from providers import get_provider
    try:provider=get_provider()
    except ValueError as exc:print(str(exc));failed=True;provider=None
    for check in (provider.doctor() if provider else []):
        print(check.message)
        if not check.ok:
            if check.fatal:failed=True
            else:missing.append(check.name)
    # Preparation: each PREP_BACKEND contributes its own checks (the GPU host check is added here by the GPU phase).
    for check in prep_checks():
        print(check.message)
        if not check.ok:
            if check.fatal:failed=True
            else:missing.append(check.name)
    if failed:print('DEPENDENCIES NOT READY');return 1
    if missing:
        print('LOCAL UI READY; GENERATION BLOCKED until keys and the required callback are configured.')
        return 2 if args.require_keys else 0
    print('LOCAL SETUP READY; provider access/balance has not been verified by this offline check.')
    return 0
if __name__=='__main__':sys.exit(main())
