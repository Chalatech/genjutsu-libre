from pathlib import Path
import os,json,time,zipfile,io
import requests

def parse_ids(value):
    """'1,2,4' or [1,2,4] -> sorted unique positive ints."""
    parts=value.split(',') if isinstance(value,str) else list(value)
    ids=set()
    for part in parts:
        part=str(part).strip()
        if not part:continue
        if not part.isdigit() or int(part)<1:raise ValueError(f'Instance ids are positive integers, got {part!r}.')
        ids.add(int(part))
    if not ids:raise ValueError('No instance ids given.')
    return sorted(ids)

def union_instances(folder,ids=None):
    """Write raw-masks/mask_NNNNN.png as the per-frame union of instances/<id>/ (all instances when ids is None)."""
    import numpy as np
    from PIL import Image
    folder=Path(folder);root=folder/'instances'
    available=sorted(int(p.name) for p in root.iterdir() if p.is_dir() and p.name.isdigit()) if root.is_dir() else []
    if not available:raise ValueError('No instance masks found; run the segmentation first.')
    ids=available if ids is None else parse_ids(ids)
    missing=[i for i in ids if i not in available]
    if missing:raise ValueError(f'Unknown instance ids {missing}; available: {available}')
    sequences=[sorted((root/str(i)).glob('mask_*.png')) for i in ids]
    if len({len(x) for x in sequences})!=1 or not sequences[0]:raise ValueError(f'Instances {ids} have different or empty frame counts.')
    out=folder/'raw-masks';out.mkdir(exist_ok=True)
    for stale in out.glob('mask_*.png'):stale.unlink()
    for i,paths in enumerate(zip(*sequences)):
        merged=np.logical_or.reduce([np.asarray(Image.open(p).convert('L'))>127 for p in paths])
        Image.fromarray(merged.astype(np.uint8)*255).save(out/f'mask_{i:05d}.png')
    (folder/'instance-selection.json').write_text(json.dumps({'ids':ids,'available':available,'frames':len(sequences[0])}))
    return out

def reselect(folder,ids,status=print):
    """Choose other instances for a GPU-prepared job: rebuild masks, composite and Seedance input.

    This changes mask.mp4 and the composite, so any earlier mask-review approval no longer matches
    (its fingerprint is bound to those files); the stale review file is also removed."""
    from pipeline import composite
    from audio_workflow import prepare_seedance_video
    folder=Path(folder)
    status('Instances · rebuilding the mask from the selected instances');union_instances(folder,ids)
    selected=json.loads((folder/'instance-selection.json').read_text())['ids']
    raw_dir=clean_border_fragments(folder/'raw-masks')
    status('Compositing · depth subject over original background')
    from pipeline import stretch_regions
    composite(folder/'original.mp4',folder/'depth.mp4',raw_dir,folder,folder/'upload.mp4',stretch_regions(folder))
    status('Embedding isolated vocals · +3 semitones');prepare_seedance_video(folder)
    (folder/'mask-review.json').unlink(missing_ok=True)
    manifest=folder/'manifest.json'
    if manifest.exists():
        meta=json.loads(manifest.read_text());meta['instances']=selected;manifest.write_text(json.dumps(meta,indent=2))
    return selected

def segment_raw(video,folder,prompt='person',provider=None,instances=None,status=print):
    import gpu_backend
    if gpu_backend.enabled(provider):
        folder=Path(folder)
        gpu_backend.segment(video,folder,prompt,status)
        status('SAM 3 · merging instances '+(str(instances) if instances else '(all)'))
        return union_instances(folder,instances)
    folder=Path(folder);h={'Authorization':'Bearer '+os.environ['REPLICATE_API_TOKEN']}
    with open(video,'rb') as f:r=requests.post('https://api.replicate.com/v1/files',headers=h,files={'content':f},timeout=120)
    r.raise_for_status();url=r.json()['urls']['get']
    payload={'version':'408f82fc20c300aac5d61d2e34ddb34cd0181e810e9f609d250aed14d5f81269','input':{'video':url,'prompt':prompt,'mask_only':True,'return_zip':True}}
    r=requests.post('https://api.replicate.com/v1/predictions',headers=h,json=payload,timeout=60);r.raise_for_status();j=r.json();(folder/'sam-raw-request.json').write_text(json.dumps(j,indent=2));print('SAM raw-mask job:',j['id'],flush=True)
    for _ in range(180):
        if j['status'] in ['succeeded','failed','canceled']:break
        time.sleep(5);r=requests.get(j['urls']['get'],headers=h,timeout=30);r.raise_for_status();j=r.json()
    if j['status']!='succeeded':raise RuntimeError(str(j.get('error') or j['status']))
    r=requests.get(j['output'],timeout=180);r.raise_for_status();(folder/'sam-masks.zip').write_bytes(r.content)
    out=folder/'raw-masks';out.mkdir(exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        for name in z.namelist():
            base=Path(name).name
            if base.startswith('mask_') and base.endswith('.png'):(out/base).write_bytes(z.read(name))
    return out

def clean_border_fragments(folder):
    """Remove tiny, narrow disconnected edge fragments; preserve large edge subjects."""
    import numpy as np
    from PIL import Image
    from scipy.ndimage import label
    folder=Path(folder);output=folder.parent/'clean-masks';output.mkdir(exist_ok=True)
    report=[]
    for p in sorted(folder.glob('mask_*.png')):
        a=np.asarray(Image.open(p).convert('L'))>127;labels,_=label(a);h,w=a.shape;removed=0
        for ident in (set(labels[:,0])|set(labels[:,-1]))-{0}:
            y,x=np.where(labels==ident)
            if len(x)<h*w*.03 and x.max()-x.min()+1<=max(6,int(w*.05)):
                a[labels==ident]=False;removed+=len(x)
        Image.fromarray(a.astype(np.uint8)*255).save(output/p.name)
        report.append({'frame':p.stem,'removed_border_pixels':removed,'mask_pixels':int(a.sum()),'left_right_edge_pixels':int(a[:,0].sum()+a[:,-1].sum())})
    (folder.parent/'mask-cleanup-report.json').write_text(json.dumps(report,indent=2))
    return output
