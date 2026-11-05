"""Optional per-subject depth stretch (`DEPTH_SUBJECT_STRETCH=1`, off by default).

Far subjects occupy a narrow slice of the depth colormap and come out as nearly flat single-colour
silhouettes, which gives the generator little to read for face orientation and limbs. This remaps the depth
colours INSIDE each selected subject's mask so its 2nd-98th percentile range spans the whole colormap.
It assumes the depth video uses the inferno colormap (what the GPU backend produces); colours are mapped
back to a scalar by nearest colour, so other palettes only work approximately. Percentile bounds are
smoothed over time so the colours do not flicker from frame to frame.
"""
from pathlib import Path
import numpy as np
from PIL import Image

LOW,HIGH,WINDOW,MIN_SPAN,MIN_PIXELS=2,98,9,8,50

def enabled():
    from config import setting
    return setting('DEPTH_SUBJECT_STRETCH','').strip().lower() in ('1','true','yes','on')

def inferno_lut():
    import cv2
    return cv2.applyColorMap(np.arange(256,dtype=np.uint8)[:,None],cv2.COLORMAP_INFERNO)[:,0,::-1].copy()

def to_scalar(pixels,lut):
    """(N,3) uint8 inferno colours -> (N,) float index 0..255 of the nearest colormap entry."""
    colours,inverse=np.unique(pixels,axis=0,return_inverse=True)
    best=np.empty(len(colours),dtype=np.float32);table=lut.astype(np.int32)
    for start in range(0,len(colours),4096):
        chunk=colours[start:start+4096].astype(np.int32)
        best[start:start+4096]=((chunk[:,None,:]-table[None,:,:])**2).sum(axis=2).argmin(axis=1)
    return best[inverse.reshape(-1)]

def _mask(path,size):
    image=Image.open(path).convert('L')
    if image.size!=size:image=image.resize(size,Image.Resampling.NEAREST)
    return np.asarray(image)>127

def _smooth(values):
    """Centered moving average ignoring NaN; frames with no subject take the nearest valid value."""
    values=np.asarray(values,dtype=np.float64);out=np.full_like(values,np.nan);half=WINDOW//2
    for i in range(len(values)):
        window=values[max(0,i-half):i+half+1]
        if np.isfinite(window).any():out[i]=np.nanmean(window)
    good=np.flatnonzero(np.isfinite(out))
    if len(good)==0:return out
    for i in np.flatnonzero(~np.isfinite(out)):out[i]=out[good[np.abs(good-i).argmin()]]
    return out

def _fix_edge(frame,m):
    """Border pixels blend subject and background depth; stretched they turn into a dark outline, so they copy the nearest interior pixel."""
    from scipy import ndimage
    inner=ndimage.binary_erosion(m,iterations=2)
    if not inner.any() or (m&~inner).sum()==0:return frame
    idx=ndimage.distance_transform_edt(~inner,return_distances=False,return_indices=True)
    ring=m&~inner;frame=frame.copy();frame[ring]=frame[idx[0][ring],idx[1][ring]]
    return frame

class SubjectStretch:
    """`regions` = mask directories (mask_NNNNN.png), one per subject, aligned with the depth frames."""
    def __init__(self,depth_path,regions):
        from pipeline import frames
        self.lut=inferno_lut();self.paths=[sorted(Path(r).glob('mask_*.png')) for r in regions]
        lows=[[] for _ in regions];highs=[[] for _ in regions]
        for i,depth in enumerate(frames(depth_path)):
            size=(depth.shape[1],depth.shape[0])
            for k,files in enumerate(self.paths):
                m=_mask(files[i],size) if i<len(files) else None
                if m is None or m.sum()<MIN_PIXELS:lows[k].append(np.nan);highs[k].append(np.nan);continue
                s=to_scalar(depth[m],self.lut);lo,hi=np.percentile(s,[LOW,HIGH]);lows[k].append(lo);highs[k].append(hi)
        self.lo=[_smooth(x) for x in lows];self.hi=[_smooth(x) for x in highs]

    def apply(self,i,depth,final_mask):
        """Return a copy of the depth frame with each subject stretched; `final_mask` is the composite's bool mask."""
        out=depth.copy();size=(depth.shape[1],depth.shape[0]);done=np.zeros(final_mask.shape,dtype=bool)
        for k,files in enumerate(self.paths):
            if i>=len(files) or not np.isfinite(self.lo[k][i]):continue
            m=_mask(files[i],size)&final_mask&~done
            if m.sum()<MIN_PIXELS:continue
            lo,hi=self.lo[k][i],self.hi[k][i]
            if hi-lo<MIN_SPAN:mid=(lo+hi)/2;lo,hi=mid-MIN_SPAN/2,mid+MIN_SPAN/2
            s=to_scalar(depth[m],self.lut);scaled=np.clip((s-lo)/(hi-lo),0,1)*255
            out[m]=self.lut[np.rint(scaled).astype(np.uint8)];done|=m
            out=_fix_edge(out,m)
        return out
