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
    return checks

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
