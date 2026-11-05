"""Facade between the app/CLI and the generation providers in `providers/`.

Public signatures are the upstream ones. Everything every provider must do the same way lives here:
refuse duplicates, write `seedance-request.json` before sending and `provider-task.json` before polling,
never retry a paid submission, check the spend cap, and restore the original audio at the end.
Provider-specific behaviour (request bodies, endpoints, limits) lives in `providers/<name>.py`.
"""
from pathlib import Path
import json,time,concurrent.futures
import requests  # noqa: F401  (kept importable here: tests and callers patch seedance_bridge.requests)
import config
from config import credential, setting
from media_host import upload
from audio_workflow import finish_preview
from providers import get_provider, Env, Job
from providers.base import SubmissionRejected, probe

def _env():
    # Looked up at call time so these module names stay patchable, exactly as in upstream.
    return Env(setting=setting,credential=credential,upload=upload)

def _task(folder):
    try:return json.loads((Path(folder)/'provider-task.json').read_text())
    except (OSError,ValueError):return None

def _provider_for(folder,name=None):
    """The provider that created this job (so resuming never switches service); else `name`, else GEN_BACKEND."""
    folder=Path(folder);saved=_task(folder) or (_task(folder.parent) if folder.name=='hd' else None)
    return get_provider((saved or {}).get('provider') or name or config.setting('GEN_BACKEND','enhancor') or 'enhancor',_env())

def estimate(folder,name=None):
    """What a draft (and its HD pass) would cost with this provider for the prepared input, before anything is sent.
    Reference images are assumed (a character reference video would add its seconds). Free: no upload, no request."""
    folder=Path(folder);prepared=folder/'seedance-input.mp4'
    if not prepared.exists():raise ValueError('Prepare the video first.')
    provider=_provider_for(folder,name);job=Job(folder=folder,prompt='',video=prepared,video_info=probe(prepared))
    try:limit=float(setting('GEN_MAX_USD','3') or 3)
    except ValueError:limit=3.0
    draft=provider.estimate_cost(job);final=provider.estimate_final_cost(job.seconds)
    return {'provider':provider.name,'seconds':round(job.seconds,2),'draft_usd':None if draft is None else round(draft,2),'final_usd':None if final is None else round(final,2),'cap_usd':limit,'over_cap':draft is not None and draft>limit}

def _write_task(folder,provider,task_id,raw,response_file):
    """Persist the task id before anything else can fail or poll. Non-upstream providers also get a
    `requestId` key so the app's and CLI's resume code keeps working."""
    folder=Path(folder)
    saved=raw if provider.response_has_request_id else {'requestId':task_id,**raw}
    (folder/response_file).write_text(json.dumps(saved))
    (folder/'provider-task.json').write_text(json.dumps({'provider':provider.name,'task_id':task_id}))

def _cap(provider,cost,status,what='request'):
    if cost is None:return
    try:limit=float(setting('GEN_MAX_USD','3') or 3)
    except ValueError:raise ValueError('GEN_MAX_USD must be a number of US dollars.')
    status(f'Estimated {provider.name} cost for this {what}: ${cost:.2f} (cap ${limit:.2f})')
    if cost>limit:raise ValueError(f'Estimated cost ${cost:.2f} exceeds GEN_MAX_USD=${limit:.2f}. Nothing was uploaded or submitted.')

def submit(folder,prompt,sirio,tein=None,status=print,provider_name=None,extra_references=None):
    folder=Path(folder)
    from mask_review import require_approved, is_mesh
    require_approved(folder)
    if (folder/'seedance-response.json').exists() or (folder/'provider-task.json').exists():raise ValueError('This job already has a provider response. Resume it; prepare a new job for a deliberate variation.')
    provider=_provider_for(folder,provider_name)
    provider.preflight()
    if not (folder/'audio-manifest.json').exists():raise ValueError('Audio separation must complete before submission.')
    from audio_workflow import prepare_seedance_video
    status('Pitching isolated vocals +3 semitones and embedding in colored-depth video')
    prepared=folder/'seedance-input.mp4' if is_mesh(folder) else prepare_seedance_video(folder)
    mesh_mode=json.loads((folder/'workflow-mode.json').read_text()).get('mode') if is_mesh(folder) else None
    job=Job(folder=folder,prompt=prompt,video=Path(prepared),references=[sirio]+([tein] if tein else [])+list(extra_references or []),mesh_mode=mesh_mode)
    if provider.needs_media_info:
        job.video_info=probe(job.video);job.reference_infos=[probe(r) for r in job.references]
    if extra_references and len(job.references)>provider.capabilities().max_reference_images:
        raise ValueError(f'{provider.name} accepts at most {provider.capabilities().max_reference_images} reference images; got {len(job.references)}.')
    provider.validate(job)
    _cap(provider,provider.estimate_cost(job),status)
    status('Uploading video with embedded vocals and character references')
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:urls=list(pool.map(provider.upload,[prepared]+job.references))
    p=provider.build_request(job,urls)
    (folder/'seedance-request.json').write_text(json.dumps(p,indent=2))
    try:ident,raw=provider.submit(p)   # one attempt: a paid submission is never retried
    except SubmissionRejected as e:
        (folder/'submission-error.json').write_text(e.body);raise ValueError(str(e))
    if not ident:
        (folder/'seedance-response.json').write_text(json.dumps(raw));raise ValueError('Seedance did not return a request ID.')
    _write_task(folder,provider,ident,raw,'seedance-response.json')
    generation={'request_id':ident,'prompt':p.get('prompt') or (p.get('input') or {}).get('prompt'),'video':'seedance-result.mp4','mode':p.get('mode','reference'),'provider':provider.name}
    if job.video_info:generation['seconds']=job.seconds
    (folder/'generation.json').write_text(json.dumps(generation))
    status('Seedance draft submitted: '+ident)
    return ident

def wait_and_finish(folder,request_id,status=print):
    folder=Path(folder)
    provider=_provider_for(folder)
    for _ in range(provider.max_polls):
        result=provider.poll(request_id)
        (folder/'provider-status.json').write_text(json.dumps(result.raw,indent=2))
        if result.state=='done':
            provider.download(result.result_url,folder/'seedance-result.mp4')
            status('Restoring entire original source audio · discarding generated audio')
            finish_preview(folder/'seedance-result.mp4',folder)
            return
        if result.state=='failed':raise RuntimeError('Seedance failed: '+result.error)
        time.sleep(provider.poll_interval)
    raise TimeoutError('Seedance is still pending. Request ID saved; do not resubmit automatically.')


def submit_hd(folder,resolution=None):
    """Complete the accepted draft with the same provider that made it."""
    folder=Path(folder)
    hd=folder/'hd'
    if (hd/'response.json').exists() or (hd/'provider-task.json').exists():raise ValueError('HD already submitted. Resume its saved request.')
    generation=json.loads((folder/'generation.json').read_text())
    original=json.loads((folder/'seedance-request.json').read_text())
    provider=get_provider(generation.get('provider') or (_task(folder) or {}).get('provider') or config.setting('GEN_BACKEND','enhancor') or 'enhancor',_env())
    if not provider.is_draft_request(original) or not (folder/'final-preview.mp4').exists():
        raise ValueError('A completed draft is required before generating 1080p.')
    payload=provider.build_final(generation['request_id'],original,resolution)
    if generation.get('seconds'):_cap(provider,provider.estimate_final_cost(generation['seconds']),print,'HD pass')
    hd.mkdir(exist_ok=True)
    # Keep the draft intact and use the exact source audio for the HD export.
    source=hd/'upload.mp4'
    if not source.exists():__import__('shutil').copy2(folder/'upload.mp4',source)
    (hd/'request.json').write_text(json.dumps(payload,indent=2))
    ident,raw=provider.submit(payload,final=True)
    if not ident:
        (hd/'response.json').write_text(json.dumps(raw));raise ValueError(f'{provider.name} did not return a request ID.')
    _write_task(hd,provider,ident,raw,'response.json')
    return ident
