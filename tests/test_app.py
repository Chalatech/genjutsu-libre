import os,shutil,unittest,uuid
from pathlib import Path
from unittest import mock
import numpy as np
from fastapi.testclient import TestClient
import app as app_module
from pipeline import writer,put,finish

ENV={'PREP_BACKEND':'gpu','GEN_BACKEND':'kie','KIE_API_KEY':'test-key-123','GEN_MAX_USD':'3'}

class AppUi(unittest.TestCase):
    def setUp(self):
        patcher=mock.patch.dict(os.environ,ENV);patcher.start();self.addCleanup(patcher.stop)
        self.client=TestClient(app_module.app)
        self.job=uuid.uuid4().hex;self.folder=app_module.RUNS/self.job;self.folder.mkdir()
        self.addCleanup(shutil.rmtree,self.folder,ignore_errors=True)

    def video(self,seconds=2):
        out,s=writer(self.folder/'seedance-input.mp4',320,180)
        for i in range(24*seconds):put(out,s,np.zeros((180,320,3),dtype=np.uint8),i)
        finish(out,s)

    def test_page_has_the_new_controls(self):
        r=self.client.get('/');self.assertEqual(r.status_code,200)
        for marker in ['id="prep-backend"','value="gpu"','id="gen-provider"','id="cost-estimate"','id="instances"','id="apply-instances"']:self.assertIn(marker,r.text)
        self.assertNotIn('through Enhancor',r.text)

    def test_config_lists_registry_capabilities_and_defaults_from_env(self):
        j=self.client.get('/config').json()
        self.assertEqual(j['prep_default'],'gpu');self.assertEqual(j['gen_default'],'kie')
        by={p['name']:p for p in j['providers']}
        self.assertEqual(set(by),{'enhancor','kie','openrouter'})
        self.assertTrue(by['kie']['capabilities']['reference_video']);self.assertTrue(by['kie']['ready'])
        self.assertFalse(by['openrouter']['capabilities']['reference_video'])

    def test_estimate_is_free_and_uses_the_provider_formula(self):
        self.video(2)
        with mock.patch('requests.post') as post:
            e=self.client.get(f'/jobs/{self.job}/estimate?provider=kie').json()
            post.assert_not_called()
        self.assertEqual(e['provider'],'kie');self.assertAlmostEqual(e['seconds'],2,places=1)
        self.assertAlmostEqual(e['draft_usd'],17*(2+2)*0.005,places=2);self.assertFalse(e['over_cap'])

    def test_estimate_errors(self):
        self.assertEqual(self.client.get(f'/jobs/{self.job}/estimate?provider=kie').status_code,409)   # not prepared yet
        self.assertEqual(self.client.get(f'/jobs/{self.job}/estimate?provider=nope').status_code,400)

    def test_seedance_rejects_unknown_provider(self):
        (self.folder/'audio-manifest.json').write_text('{}')
        with mock.patch('mask_review.require_approved'):
            r=self.client.post(f'/jobs/{self.job}/seedance',data={'prompt':'p','provider':'nope'},files={'sirio':('a.png',b'x','image/png')})
        self.assertEqual(r.status_code,400)

    def test_instances_route_without_listing_is_404(self):
        self.assertEqual(self.client.get(f'/jobs/{self.job}/instances').status_code,404)

if __name__=='__main__':unittest.main()
