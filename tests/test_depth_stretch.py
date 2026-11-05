import json,os,tempfile,unittest
from pathlib import Path
from unittest import mock
import numpy as np
from PIL import Image
import depth_stretch
from pipeline import writer,put,finish,composite

W,H,N=64,48,6

def lut():return depth_stretch.inferno_lut()

def make_job(root,levels=(100,120)):
    """Depth video: background level 30; one subject (left box) in a narrow slice of the colormap."""
    table=lut();root=Path(root);(root/'masks').mkdir()
    out,stream=writer(root/'depth.mp4',W,H);orig,ostream=writer(root/'original.mp4',W,H)
    ramp=np.linspace(levels[0],levels[1],20).astype(int)
    for i in range(N):
        scalar=np.full((H,W),30,dtype=np.uint8);scalar[10:30,8:28]=ramp[None,:]
        put(out,stream,table[scalar],i);put(orig,ostream,np.full((H,W,3),90,dtype=np.uint8),i)
        mask=np.zeros((H,W),dtype=np.uint8);mask[10:30,8:28]=255
        Image.fromarray(mask).save(root/'masks'/f'mask_{i:05d}.png')
    finish(out,stream);finish(orig,ostream)

class StretchTests(unittest.TestCase):
    def test_scalar_round_trip(self):
        table=lut();idx=np.array([0,40,128,255]);got=depth_stretch.to_scalar(table[idx],table)
        self.assertEqual(got.tolist(),idx.tolist())

    def test_stretch_widens_subject_range_and_keeps_background(self):
        with tempfile.TemporaryDirectory() as t:
            make_job(t);table=lut()
            s=depth_stretch.SubjectStretch(Path(t)/'depth.mp4',[Path(t)/'masks'])
            from pipeline import frames
            depth=next(frames(Path(t)/'depth.mp4'));mask=np.zeros((H,W),dtype=bool);mask[10:30,8:28]=True
            out=s.apply(0,depth,mask)
            before=depth_stretch.to_scalar(depth[mask],table);after=depth_stretch.to_scalar(out[mask],table)
            self.assertLess(np.ptp(before),40);self.assertGreater(np.ptp(after),150)
            self.assertTrue((out[~mask]==depth[~mask]).all())

    def test_small_range_is_not_blown_up_into_noise(self):
        with tempfile.TemporaryDirectory() as t:
            make_job(t,(100,101));table=lut()
            s=depth_stretch.SubjectStretch(Path(t)/'depth.mp4',[Path(t)/'masks'])
            self.assertTrue(np.all(s.hi[0]-s.lo[0]<depth_stretch.MIN_SPAN))

    def test_composite_off_by_default_and_on_with_regions(self):
        with tempfile.TemporaryDirectory() as t:
            make_job(t);root=Path(t);table=lut()
            composite(root/'original.mp4',root/'depth.mp4',root/'masks',root)
            plain=Path(root/'composite-silent.mp4').read_bytes()
            composite(root/'original.mp4',root/'depth.mp4',root/'masks',root,stretch=True)
            self.assertNotEqual(plain,Path(root/'composite-silent.mp4').read_bytes())

    def test_stretch_regions_follows_env(self):
        from pipeline import stretch_regions
        with tempfile.TemporaryDirectory() as t:
            (Path(t)/'instance-selection.json').write_text(json.dumps({'ids':[2,4]}))
            with mock.patch.dict(os.environ,{'DEPTH_SUBJECT_STRETCH':''}):self.assertIsNone(stretch_regions(t))
            with mock.patch.dict(os.environ,{'DEPTH_SUBJECT_STRETCH':'1'}):
                self.assertEqual(stretch_regions(t),[Path(t)/'instances'/'2',Path(t)/'instances'/'4'])
                self.assertIs(stretch_regions(t,per_instance=False),True)

if __name__=='__main__':unittest.main()
