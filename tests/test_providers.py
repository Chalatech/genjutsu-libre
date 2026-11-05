import json,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock,patch
import seedance_bridge,providers
from providers import Env,Job,Capabilities,get_provider
from providers.base import SubmissionRejected,Unsupported,PollResult
from providers.kie import KieProvider
from providers.openrouter import OpenRouterProvider

def env(**values):
    return Env(setting=lambda k,d='':values.get(k,d),credential=lambda k:values.get(k) or (_ for _ in ()).throw(ValueError('Set '+k)),upload=lambda p:'https://example.org/'+Path(p).name)

GOOD_VIDEO={'width':960,'height':540,'fps':24.0,'duration':6.0,'bytes':5_000_000,'suffix':'.mp4','kind':'video'}
GOOD_IMAGE={'width':1024,'height':1024,'bytes':900_000,'suffix':'.png','kind':'image'}
def job(video=None,refs=None,prompt='Replace the subject. @Audio1'):
    refs=refs if refs is not None else [dict(GOOD_IMAGE)]
    return Job(folder=Path('.'),prompt=prompt,video=Path('seedance-input.mp4'),references=[Path('r%d%s'%(i,r['suffix'])) for i,r in enumerate(refs)],video_info=video or dict(GOOD_VIDEO),reference_infos=refs)

def response(payload,ok=True,status=200,text=''):
    r=Mock();r.ok=ok;r.status_code=status;r.text=text or json.dumps(payload);r.json.return_value=payload;return r

class Contract(unittest.TestCase):
    def test_every_registered_provider_meets_the_contract(self):
        self.assertEqual(set(providers.available())>={'enhancor','kie','openrouter'},True)
        for name in providers.available():
            p=get_provider(name,env(OPENROUTER_MODEL='x'))
            with self.subTest(name):
                self.assertEqual(p.name,name)
                with patch('requests.get',side_effect=__import__('requests').ConnectionError) if name=='openrouter' else patch('os.getcwd'):
                    caps=p.capabilities()
                self.assertIsInstance(caps,Capabilities)
                for method in ['validate','estimate_cost','upload','build_request','submit','poll','download','build_final','preflight','doctor','is_draft_request']:
                    self.assertTrue(callable(getattr(p,method)),method)
    def test_unknown_backend_names_the_choices(self):
        with self.assertRaisesRegex(ValueError,'Available: enhancor, kie, openrouter'):get_provider('nope',env())
    def test_default_backend_is_enhancor(self):
        self.assertEqual(get_provider(None,env()).name,'enhancor')

class KieBodies(unittest.TestCase):
    def test_draft_body(self):
        p=KieProvider(env(KIE_MODEL='bytedance/seedance-2-5'))
        body=p.build_request(job(),['https://x/v.mp4','https://x/i.png'])
        i=body['input']
        self.assertEqual(body['model'],'bytedance/seedance-2-5')
        self.assertEqual((i['reference_video_urls'],i['reference_image_urls']),(['https://x/v.mp4'],['https://x/i.png']))
        self.assertEqual((i['draft'],i['resolution'],i['duration'],i['generate_audio']),(True,'480p',-1,False))
        self.assertIn('@Video1',i['prompt']);self.assertIn('@Image1',i['prompt']);self.assertNotIn('@Audio1',i['prompt'])
    def test_reference_video_for_identity_is_a_second_video(self):
        refs=[dict(GOOD_VIDEO),dict(GOOD_IMAGE)]
        body=KieProvider(env()).build_request(job(refs=refs),['https://x/v.mp4','https://x/ref.mp4','https://x/i.png'])['input']
        self.assertEqual(body['reference_video_urls'],['https://x/v.mp4','https://x/ref.mp4']);self.assertEqual(body['reference_image_urls'],['https://x/i.png']);self.assertIn('@Video2',body['prompt'])
    def test_hd_body(self):
        p=KieProvider(env())
        self.assertEqual(p.build_final('t1',{}),{'model':'bytedance/seedance-2-5-draft','input':{'draft_task_id':'t1','resolution':'1080p','output_format':'mp4'}})
        self.assertEqual(p.build_final('t1',{},'720p')['input']['resolution'],'720p')
        with self.assertRaises(ValueError):p.build_final('t1',{},'480p')
    def test_cost_formula(self):
        p=KieProvider(env(KIE_UNIT_CREDITS='17'))
        self.assertAlmostEqual(p.estimate_cost(job()),17*(6+6)*0.005)   # reference + output seconds
        self.assertAlmostEqual(KieProvider(env(KIE_UNIT_CREDITS='20')).estimate_cost(job()),20*12*0.005)

class KieValidation(unittest.TestCase):
    def check(self,**kw):
        v=dict(GOOD_VIDEO);v.update(kw.pop('video',{}))
        KieProvider(env()).validate(job(video=v,**kw))
    def bad(self,pattern,**kw):
        with self.assertRaisesRegex(ValueError,pattern):self.check(**kw)
    def test_good_inputs_pass(self):self.check()
    def test_pixels_bounds(self):
        self.bad('total',video={'width':640,'height':480})      # 307200 < 409600
        self.check(video={'width':1280,'height':720})           # 921600 px is inside the range
    def test_pixels_too_large(self):self.bad('total',video={'width':1920,'height':1080})
    def test_fps(self):
        self.bad('frame rate',video={'fps':12.0});self.bad('frame rate',video={'fps':120.0})
    def test_duration(self):
        self.bad('duration must be 2-30',video={'duration':1.0});self.bad('duration must be 2-30',video={'duration':31.0})
    def test_video_size_and_format(self):
        self.bad('200 MB',video={'bytes':201*1024*1024});self.bad('format',video={'suffix':'.avi'})
    def test_aspect_and_sides(self):
        self.bad('aspect',video={'width':3000,'height':300});self.bad('sides',video={'width':200,'height':2000})
    def test_total_reference_duration(self):
        self.bad('total reference video duration',video={'duration':20.0},refs=[dict(GOOD_VIDEO,duration=20.0)])
    def test_image_rules(self):
        self.bad('under 30 MB',refs=[dict(GOOD_IMAGE,bytes=31*1024*1024)])
        self.bad('sides',refs=[dict(GOOD_IMAGE,width=200)])
        self.bad('format',refs=[dict(GOOD_IMAGE,suffix='.gif')])
        self.bad('aspect',refs=[dict(GOOD_IMAGE,width=2000,height=500)])
    def test_all_problems_reported_together(self):
        with self.assertRaises(ValueError) as c:self.check(video={'fps':12.0,'duration':1.0})
        self.assertIn('frame rate',str(c.exception));self.assertIn('duration',str(c.exception))

class KieSubmitAndPoll(unittest.TestCase):
    def test_rejected_by_code_even_with_http_200(self):
        with patch('requests.post',return_value=response({'code':402,'msg':'Insufficient credits'})) as post:
            with self.assertRaisesRegex(SubmissionRejected,'402'):KieProvider(env(KIE_API_KEY='k')).submit({'model':'m','input':{}})
        self.assertEqual(post.call_count,1)
    def test_submit_returns_task_id_and_bearer_header(self):
        with patch('requests.post',return_value=response({'code':200,'data':{'taskId':'T9'}})) as post:
            self.assertEqual(KieProvider(env(KIE_API_KEY='k')).submit({'a':1})[0],'T9')
        self.assertEqual(post.call_args.kwargs['headers'],{'Authorization':'Bearer k'});self.assertTrue(post.call_args.args[0].endswith('/api/v1/jobs/createTask'))
    def poll(self,data,code=200):
        with patch('requests.get',return_value=response({'code':code,'msg':'m','data':data})) as get:
            r=KieProvider(env(KIE_API_KEY='k')).poll('T9')
        self.assertEqual(get.call_args.kwargs['params'],{'taskId':'T9'});return r
    def test_waiting_queuing_generating_are_pending(self):
        for s in ['waiting','queuing','generating']:self.assertEqual(self.poll({'state':s}).state,'pending')
    def test_success_extracts_url(self):
        r=self.poll({'state':'success','resultJson':json.dumps({'resultUrls':['https://f/r.mp4']})})
        self.assertEqual((r.state,r.result_url),('done','https://f/r.mp4'))
    def test_success_without_url_is_an_error(self):
        with self.assertRaises(ValueError):self.poll({'state':'success','resultJson':'{}'})
    def test_fail_carries_provider_message(self):
        r=self.poll({'state':'fail','failMsg':'content policy','creditsConsumed':0});self.assertEqual((r.state,r.error),('failed','content policy'))
    def test_error_code_on_status_raises(self):
        with self.assertRaises(RuntimeError):self.poll({},code=500)

class KieFacade(unittest.TestCase):
    """The shared behaviour in seedance_bridge, driven through the kie provider with HTTP mocked."""
    def setUp(self):
        self.d=tempfile.TemporaryDirectory();self.p=Path(self.d.name);(self.p/'audio-manifest.json').write_text('{}')
        self.cfg={'GEN_BACKEND':'kie','KIE_API_KEY':'k','GEN_MAX_USD':'3','KIE_UNIT_CREDITS':'17'}
        self.posts=[]
        def post(url,**kw):
            self.posts.append(url)
            if 'file-stream-upload' in url:return response({'success':True,'code':200,'data':{'downloadUrl':'https://tmp/'+kw['files']['file'][0]}})
            return response({'code':200,'data':{'taskId':'T1'}})
        self.post=post
    def tearDown(self):self.d.cleanup()
    def run_submit(self,cfg=None,infos=None):
        cfg={**self.cfg,**(cfg or {})}
        (self.p/'seedance-input.mp4').write_bytes(b'v');(self.p/'ref.png').write_bytes(b'i')
        probes={'seedance-input.mp4':GOOD_VIDEO,'ref.png':GOOD_IMAGE}
        stack=[patch('mask_review.require_approved'),patch('mask_review.is_mesh',return_value=False),patch('audio_workflow.prepare_seedance_video',return_value=self.p/'seedance-input.mp4'),
               patch('config.setting',side_effect=lambda k,d='':cfg.get(k,d)),patch('seedance_bridge.setting',side_effect=lambda k,d='':cfg.get(k,d)),
               patch('seedance_bridge.credential',side_effect=lambda k:cfg[k]),patch('seedance_bridge.probe',side_effect=lambda path:(infos or probes)[Path(path).name]),patch('requests.post',side_effect=self.post)]
        for s in stack:s.start()
        try:return seedance_bridge.submit(self.p,'Replace the subject',str(self.p/'ref.png'),status=lambda s:None)
        finally:
            for s in reversed(stack):s.stop()
    def test_submit_writes_state_and_never_resends(self):
        self.assertEqual(self.run_submit(),'T1')
        for name in ['seedance-request.json','provider-task.json','seedance-response.json','generation.json']:self.assertTrue((self.p/name).exists(),name)
        self.assertEqual(json.loads((self.p/'provider-task.json').read_text()),{'provider':'kie','task_id':'T1'})
        self.assertEqual(json.loads((self.p/'seedance-response.json').read_text())['requestId'],'T1')
        self.assertEqual(sum('createTask' in u for u in self.posts),1)
        before=len(self.posts)
        with self.assertRaisesRegex(ValueError,'already has a provider response'):self.run_submit()
        self.assertEqual(len(self.posts),before)       # duplicate refused before any request
    def test_provider_task_alone_blocks_duplicates(self):
        (self.p/'provider-task.json').write_text('{"provider":"kie","task_id":"T0"}')
        with self.assertRaisesRegex(ValueError,'already has a provider response'):self.run_submit()
        self.assertEqual(self.posts,[])
    def test_cost_cap_blocks_before_any_upload(self):
        with self.assertRaisesRegex(ValueError,'GEN_MAX_USD'):self.run_submit({'GEN_MAX_USD':'0.5'})
        self.assertEqual(self.posts,[])
    def test_local_validation_blocks_before_any_upload(self):
        with self.assertRaisesRegex(ValueError,'frame rate'):self.run_submit(infos={'seedance-input.mp4':dict(GOOD_VIDEO,fps=12.0),'ref.png':GOOD_IMAGE})
        self.assertEqual(self.posts,[])
    def test_http_failure_on_create_is_not_retried(self):
        def post(url,**kw):
            self.posts.append(url)
            if 'file-stream-upload' in url:return self.__class__.post_ok(url,kw)
            raise OSError('connection reset')
        self.post=post
        with self.assertRaises(OSError):self.run_submit()
        self.assertEqual(sum('createTask' in u for u in self.posts),1)
        self.assertFalse((self.p/'provider-task.json').exists())
    @staticmethod
    def post_ok(url,kw):return response({'success':True,'code':200,'data':{'downloadUrl':'https://tmp/'+kw['files']['file'][0]}})
    def test_rejection_is_saved_and_reported(self):
        def post(url,**kw):
            if 'file-stream-upload' in url:return self.post_ok(url,kw)
            return response({'code':422,'msg':'bad prompt'})
        self.post=post
        with self.assertRaisesRegex(ValueError,'422'):self.run_submit()
        self.assertTrue((self.p/'submission-error.json').exists());self.assertFalse((self.p/'provider-task.json').exists())
    def test_wait_uses_saved_provider_and_survives_waiting(self):
        self.run_submit()
        seq=[response({'code':200,'data':{'state':'waiting'}}),response({'code':200,'data':{'state':'generating'}}),response({'code':200,'data':{'state':'success','resultJson':json.dumps({'resultUrls':['https://f/r.mp4']})}})]
        media=Mock();media.content=b'video'
        with patch('config.setting',side_effect=lambda k,d='':{'GEN_BACKEND':'enhancor'}.get(k,d)),patch('seedance_bridge.credential',return_value='k'),patch('seedance_bridge.setting',side_effect=lambda k,d='':self.cfg.get(k,d)),\
             patch('requests.get',side_effect=seq+[media]) as get,patch('seedance_bridge.time.sleep'),patch('seedance_bridge.finish_preview') as fin:
            seedance_bridge.wait_and_finish(self.p,'T1',lambda s:None)
        self.assertEqual((self.p/'seedance-result.mp4').read_bytes(),b'video');fin.assert_called_once()
        self.assertIn('recordInfo',get.call_args_list[0].args[0])       # kie chosen from provider-task.json although GEN_BACKEND says enhancor
    def test_failed_task_raises_provider_message(self):
        self.run_submit()
        with patch('seedance_bridge.credential',return_value='k'),patch('seedance_bridge.setting',side_effect=lambda k,d='':self.cfg.get(k,d)),\
             patch('requests.get',return_value=response({'code':200,'data':{'state':'fail','failMsg':'timeout in queue'}})):
            with self.assertRaisesRegex(RuntimeError,'timeout in queue'):seedance_bridge.wait_and_finish(self.p,'T1',lambda s:None)
    def test_hd_submit_once_with_draft_id(self):
        self.run_submit()
        for n in ['final-preview.mp4','upload.mp4']:(self.p/n).write_bytes(b'x')
        calls=[]
        def post(url,**kw):calls.append(kw['json']);return response({'code':200,'data':{'taskId':'HD1'}})
        with patch('config.setting',side_effect=lambda k,d='':self.cfg.get(k,d)),patch('seedance_bridge.setting',side_effect=lambda k,d='':self.cfg.get(k,d)),patch('seedance_bridge.credential',return_value='k'),patch('requests.post',side_effect=post):
            self.assertEqual(seedance_bridge.submit_hd(self.p),'HD1')
            with self.assertRaisesRegex(ValueError,'HD already submitted'):seedance_bridge.submit_hd(self.p)
        self.assertEqual(calls,[{'model':'bytedance/seedance-2-5-draft','input':{'draft_task_id':'T1','resolution':'1080p','output_format':'mp4'}}])
        self.assertEqual(json.loads((self.p/'hd/provider-task.json').read_text())['task_id'],'HD1')
    def test_hd_needs_a_completed_draft(self):
        self.run_submit()
        with patch('config.setting',side_effect=lambda k,d='':self.cfg.get(k,d)):
            with self.assertRaisesRegex(ValueError,'completed draft'):seedance_bridge.submit_hd(self.p)

CATALOG={'data':[{'id':'wan/wan-3.0','architecture':{'input_modalities':['text','image'],'output_modalities':['video']}},
                 {'id':'acme/text-only','architecture':{'input_modalities':['text'],'output_modalities':['text']}},
                 {'id':'acme/video-edit','architecture':{'input_modalities':['text','image','video'],'output_modalities':['video']}}]}
class OpenRouter(unittest.TestCase):
    def provider(self,model,catalog=CATALOG,error=None):
        get=patch('requests.get',side_effect=error) if error else patch('requests.get',return_value=response(catalog))
        self.get=get.start();self.addCleanup(get.stop)
        return OpenRouterProvider(env(OPENROUTER_MODEL=model,OPENROUTER_API_KEY='k'))
    def test_image_only_video_model_is_refused_before_anything(self):
        p=self.provider('wan/wan-3.0')
        self.assertFalse(p.capabilities().reference_video);self.assertEqual(p.capabilities().max_reference_images,30)
        with self.assertRaisesRegex(Unsupported,'does not accept a reference video'):p.preflight()
    def test_text_model_is_refused(self):
        with self.assertRaises(Unsupported):self.provider('acme/text-only').preflight()
    def test_unknown_model_and_missing_model(self):
        with self.assertRaisesRegex(Unsupported,'not in the OpenRouter catalog'):self.provider('nope/x').preflight()
        with self.assertRaisesRegex(Unsupported,'Set OPENROUTER_MODEL'):self.provider('').preflight()
    def test_unreachable_catalog_falls_back_to_unsupported(self):
        import requests
        p=self.provider('wan/wan-3.0',error=requests.ConnectionError())
        self.assertFalse(p.capabilities().reference_video)
        with self.assertRaisesRegex(Unsupported,'unreachable'):p.preflight()
    def test_catalog_confirmed_model_passes_preflight_but_submit_is_not_implemented(self):
        p=self.provider('acme/video-edit');self.assertTrue(p.capabilities().reference_video);p.preflight()
        with self.assertRaises(Unsupported):p.submit({})
    def test_catalog_read_once_per_provider(self):
        p=self.provider('wan/wan-3.0');p.capabilities();p.capabilities();self.assertEqual(self.get.call_count,1)
    def test_facade_stops_before_preparing_media(self):
        p=self.provider('wan/wan-3.0')
        with tempfile.TemporaryDirectory() as d,patch('mask_review.require_approved'),patch('config.setting',side_effect=lambda k,d='':{'GEN_BACKEND':'openrouter','OPENROUTER_MODEL':'wan/wan-3.0'}.get(k,d)),\
             patch('seedance_bridge.setting',side_effect=lambda k,d='':{'GEN_BACKEND':'openrouter','OPENROUTER_MODEL':'wan/wan-3.0'}.get(k,d)),patch('audio_workflow.prepare_seedance_video') as prep,patch('requests.post') as post:
            with self.assertRaises(Unsupported):seedance_bridge.submit(d,'p','r.png')
            prep.assert_not_called();post.assert_not_called()

class EnhancorUnchanged(unittest.TestCase):
    def test_enhancor_has_no_cost_estimate_so_the_cap_never_applies(self):
        self.assertIsNone(get_provider('enhancor',env()).estimate_cost(job()))
    def test_doctor_checks_name_the_webhook(self):
        names=[c.name for c in get_provider('enhancor',env()).doctor()];self.assertEqual(names,['ENHANCOR_API_KEY','ENHANCOR_WEBHOOK_URL'])
        self.assertEqual([c.name for c in get_provider('kie',env(KIE_API_KEY='abc')).doctor() if not c.ok],[])
if __name__=='__main__':unittest.main()
