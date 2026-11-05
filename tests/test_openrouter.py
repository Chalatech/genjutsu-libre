import json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock,patch
import requests
import seedance_bridge
from providers import Env,Job,get_provider
from providers.base import SubmissionRejected,Unsupported
from providers.openrouter import OpenRouterProvider

ROOT=Path(__file__).resolve().parent.parent
MODELS={'data':[{'id':'bytedance/seedance-2.5','supported_resolutions':['480p','720p'],'supported_aspect_ratios':['16:9','9:16','1:1'],
    'supported_sizes':['854x480','480x854','640x640','1280x720','720x1280','960x960'],'supported_durations':list(range(4,31)),
    'pricing_skus':{'video_tokens':'0.0000107','video_tokens_with_video_input':'0.0000064'}},
    {'id':'acme/old-video','supported_durations':[4],'supported_resolutions':['720p']}]}
VIDEO={'width':960,'height':540,'fps':24.0,'duration':4.0,'bytes':5_000_000,'suffix':'.mp4','kind':'video'}
IMAGE={'width':1024,'height':1024,'bytes':900_000,'suffix':'.png','kind':'image'}

def env(**v):
    return Env(setting=lambda k,d='':v.get(k,d),credential=lambda k:v.get(k) or (_ for _ in ()).throw(ValueError('Set '+k)))
def resp(payload,ok=True,status=200,content=b''):
    r=Mock();r.ok=ok;r.status_code=status;r.text=json.dumps(payload);r.json.return_value=payload;r.content=content;return r
def job(video=VIDEO,n_images=2,extra_videos=0,folder=Path('.')):
    refs=[dict(IMAGE) for _ in range(n_images)]+[dict(VIDEO,duration=3.0) for _ in range(extra_videos)]
    return Job(folder=folder,prompt='Replace the subject. @Audio1',video=Path('seedance-input.mp4'),video_info=dict(video),
               references=[Path(f'r{i}{r["suffix"]}') for i,r in enumerate(refs)],reference_infos=refs)
def provider(**v):
    v.setdefault('OPENROUTER_API_KEY','k')
    g=patch('requests.get',return_value=resp(MODELS));g.start()
    p=OpenRouterProvider(env(**v));p._tg=g;return p
class Base(unittest.TestCase):
    def provider(self,**v):
        p=provider(**v);self.addCleanup(p._tg.stop);return p

class Capabilities(Base):
    def test_parsed_from_the_video_models_endpoint(self):
        p=self.provider();c=p.capabilities()
        self.assertTrue(c.reference_video);self.assertEqual((c.min_duration,c.max_duration),(4,30))
        self.assertEqual((c.resolutions,c.final_resolutions,c.draft),(('480p',),('720p',),True))
        self.assertTrue(requests.get.call_args.args[0].endswith('/api/v1/videos/models'))
        p.capabilities();self.assertEqual(requests.get.call_count,1)
    def test_unreachable_list_falls_back_to_the_table(self):
        with patch('requests.get',side_effect=requests.ConnectionError):
            p=OpenRouterProvider(env(OPENROUTER_API_KEY='k'));self.assertTrue(p.capabilities().reference_video);self.assertEqual(p.capabilities().max_duration,30)
    def test_listed_model_without_reference_video_support_is_refused(self):
        p=self.provider(OPENROUTER_MODEL='acme/old-video')
        with self.assertRaisesRegex(Unsupported,'does not accept a reference video'):p.preflight()
    def test_wan_from_table_and_unknown_model(self):
        with self.assertRaisesRegex(Unsupported,'does not accept a reference video'):self.provider(OPENROUTER_MODEL='alibaba/wan-3.0').preflight()
        with self.assertRaisesRegex(Unsupported,'not in OpenRouter'):self.provider(OPENROUTER_MODEL='nope/x').preflight()
    def test_default_model_is_seedance_25(self):
        self.assertEqual(self.provider()._model_id(),'bytedance/seedance-2.5')

class Keys(Base):
    def test_key_fallback_name(self):
        p=OpenRouterProvider(env(OPENROUTER_KEY='fallback'));self.assertEqual(p._headers(),{'Authorization':'Bearer fallback'})
        p=OpenRouterProvider(env(OPENROUTER_KEY='b',OPENROUTER_API_KEY='a'));self.assertEqual(p._headers(),{'Authorization':'Bearer a'})
    def test_missing_key(self):
        with self.assertRaisesRegex(ValueError,'OPENROUTER_API_KEY'):OpenRouterProvider(env())._headers()
    def test_doctor_reports_key_and_host(self):
        p=self.provider(OPENROUTER_KEY='x',OPENROUTER_API_KEY='');names={c.name:c for c in p.doctor()}
        self.assertTrue(names['OPENROUTER_API_KEY'].ok);self.assertTrue(names['OPENROUTER_MODEL'].ok);self.assertTrue(names['OPENROUTER_MEDIA_HOST'].ok)
        self.assertFalse({c.name:c for c in self.provider(OPENROUTER_API_KEY='',OPENROUTER_MEDIA_HOST='kie').doctor()}['OPENROUTER_MEDIA_HOST'].ok)

class MediaHost(Base):
    def test_default_is_kie_when_a_kie_key_exists_else_tmpfiles(self):
        self.assertEqual(self.provider(KIE_API_KEY='kk')._host(),'kie');self.assertEqual(self.provider()._host(),'tmpfiles')
    def test_kie_upload_is_reused(self):
        p=self.provider(OPENROUTER_MEDIA_HOST='kie',KIE_API_KEY='kk')
        with patch('providers.openrouter.KieProvider.upload',return_value='https://kie/x.mp4') as up:self.assertEqual(p.upload('a.mp4'),'https://kie/x.mp4');up.assert_called_once_with('a.mp4')
    def test_tmpfiles_has_no_fallbacks(self):
        p=self.provider(OPENROUTER_MEDIA_HOST='tmpfiles',KIE_API_KEY='kk')
        with patch('media_host.upload',return_value='https://tmpfiles.org/dl/1/x.mp4') as up:p.upload('a.mp4')
        up.assert_called_once_with('a.mp4',hosts=['tmpfiles'])
    def test_catbox_is_not_an_option(self):
        with self.assertRaisesRegex(ValueError,'kie or tmpfiles'):self.provider(OPENROUTER_MEDIA_HOST='catbox').upload('a.mp4')
    def test_media_host_hosts_override(self):
        import media_host
        with patch('media_host._upload_once',return_value='https://t/x') as once,patch('media_host.setting',side_effect=lambda k,d='':{'MEDIA_HOST':'tmpfiles','MEDIA_HOST_FALLBACKS':'catbox'}.get(k,d)):
            media_host.upload('a',hosts=['tmpfiles']);once.assert_called_once_with('a','tmpfiles')

class Request(Base):
    URLS=['https://h/v.mp4','https://h/i1.png','https://h/i2.png']
    def test_probe_body(self):
        body=self.provider().build_request(job(),self.URLS)
        self.assertEqual(body['input_references'],[{'type':'video_url','video_url':{'url':'https://h/v.mp4'}},
            {'type':'image_url','image_url':{'url':'https://h/i1.png'}},{'type':'image_url','image_url':{'url':'https://h/i2.png'}}])
        self.assertEqual({k:body[k] for k in ('model','duration','resolution','aspect_ratio','generate_audio')},
                         {'model':'bytedance/seedance-2.5','duration':4,'resolution':'480p','aspect_ratio':'16:9','generate_audio':False})
        for note in ('@Video1','@Image1','@Image2'):self.assertIn(note,body['prompt'])
        self.assertNotIn('@Audio1',body['prompt']);self.assertNotIn('seed',body)
    def test_duration_rounds_and_clamps(self):
        p=self.provider()
        for secs,want in [(14.96,15),(2.0,4),(45.0,30),(4.4,4)]:self.assertEqual(p.build_request(job(dict(VIDEO,duration=secs)),self.URLS)['duration'],want)
    def test_aspect_closest_supported(self):
        p=self.provider()
        self.assertEqual(p.build_request(job(dict(VIDEO,width=540,height=960)),self.URLS)['aspect_ratio'],'9:16')
        self.assertEqual(p.build_request(job(dict(VIDEO,width=700,height=700)),self.URLS)['aspect_ratio'],'1:1')
    def test_resolution_from_env_and_character_video_as_second_video(self):
        p=self.provider(OPENROUTER_RESOLUTION='720p');body=p.build_request(job(n_images=1,extra_videos=1),['https://h/v.mp4','https://h/i.png','https://h/r.mp4'])
        self.assertEqual(body['resolution'],'720p')
        self.assertEqual([r['type'] for r in body['input_references']],['video_url','video_url','image_url'])
        self.assertIn('@Video2',body['prompt'])

class Cost(Base):
    def test_probe_estimate(self):
        # output 854x480 x4 s + input 960x540 x4 s, 24 fps, /1024, $0.0000064 per token
        want=(854*480*24*4/1024+960*540*24*4/1024)*0.0000064
        self.assertAlmostEqual(self.provider().estimate_cost(job()),want);self.assertAlmostEqual(want,0.5570,places=3)
    def test_long_clip_and_reference_video_seconds(self):
        p=self.provider();v=dict(VIDEO,duration=14.96)
        want=(854*480*24*15/1024+960*540*24*14.96/1024)*0.0000064
        self.assertAlmostEqual(p.estimate_cost(job(v)),want)
        self.assertGreater(p.estimate_cost(job(v,n_images=0,extra_videos=1)),p.estimate_cost(job(v,n_images=0)))
    def test_final_estimate_is_larger(self):
        p=self.provider();self.assertGreater(p.estimate_final_cost(4.0),p.estimate_cost(job()))

class Validation(Base):
    def bad(self,pattern,**kw):
        with self.assertRaisesRegex(ValueError,pattern):self.provider().validate(job(**kw))
    def test_good(self):self.provider().validate(job())
    def test_rules(self):
        self.bad('frame rate',video=dict(VIDEO,fps=12.0));self.bad('duration must be 2-30',video=dict(VIDEO,duration=1.0))
        self.bad('total',video=dict(VIDEO,width=1920,height=1080));self.bad('sides',n_images=0,video=dict(VIDEO,width=200,height=2000))
    def test_image_count_is_configurable(self):
        with self.assertRaisesRegex(ValueError,'more than 3 reference images'):self.provider(OPENROUTER_MAX_REF_IMAGES='3').validate(job(n_images=4))
        self.provider(OPENROUTER_MAX_REF_IMAGES='3').validate(job(n_images=3))

class Transport(Base):
    def test_submit_once_with_auth_and_rejection(self):
        p=self.provider()
        with patch('requests.post',return_value=resp({'id':'J1','polling_url':'u','status':'pending'})) as post:
            self.assertEqual(p.submit({'model':'m'})[0],'J1')
        self.assertEqual(post.call_args.kwargs['headers'],{'Authorization':'Bearer k'});self.assertTrue(post.call_args.args[0].endswith('/api/v1/videos'));self.assertEqual(post.call_count,1)
        with patch('requests.post',return_value=resp({},ok=False,status=402)):
            with self.assertRaises(SubmissionRejected):p.submit({})
    def poll(self,payload):
        p=self.provider()
        with patch('requests.get',return_value=resp(payload)) as get:r=p.poll('J1')
        self.assertTrue(get.call_args.args[0].endswith('/api/v1/videos/J1'));return p,r
    def test_states(self):
        for s in ('pending','in_progress','processing'):self.assertEqual(self.poll({'status':s})[1].state,'pending')
        for s in ('cancelled','expired'):self.assertEqual(self.poll({'status':s})[1].state,'failed')
        r=self.poll({'status':'failed','error':'moderation'})[1];self.assertEqual((r.state,r.error),('failed','moderation'))
        r=self.poll({'status':'completed','unsigned_urls':['https://openrouter.ai/api/v1/videos/J1/content?index=0'],'usage':{'cost':0.5}})[1]
        self.assertEqual((r.state,r.result_url),('done','https://openrouter.ai/api/v1/videos/J1/content?index=0'))
        self.assertEqual(self.poll({'status':'completed','urls':['https://signed/x']})[1].result_url,'https://signed/x')
        with self.assertRaises(ValueError):self.poll({'status':'completed'})
    def test_download_sends_auth_and_records_cost(self):
        p,r=self.poll({'status':'completed','generation_id':'gen-1','unsigned_urls':['https://openrouter.ai/api/v1/videos/J1/content?index=0'],'usage':{'cost':0.4321,'is_byok':False}})
        with tempfile.TemporaryDirectory() as d,patch('requests.get',return_value=resp({},content=b'mp4')) as get:
            p.download(r.result_url,Path(d)/'seedance-result.mp4')
            self.assertEqual(get.call_args.kwargs['headers'],{'Authorization':'Bearer k'})
            self.assertEqual((Path(d)/'seedance-result.mp4').read_bytes(),b'mp4')
            self.assertEqual(json.loads((Path(d)/'provider-usage.json').read_text())['cost_usd'],0.4321)
    def test_signed_url_gets_no_key(self):
        p=self.provider()
        with tempfile.TemporaryDirectory() as d,patch('requests.get',return_value=resp({},content=b'x')) as get:
            p.download('https://signed.example/x.mp4',Path(d)/'o.mp4');self.assertEqual(get.call_args.kwargs['headers'],{})
    def test_second_pass_is_a_new_generation_at_higher_resolution(self):
        p=self.provider();draft=p.build_request(job(),Request.URLS)
        self.assertTrue(p.is_draft_request(draft))
        final=p.build_final('J1',draft);self.assertEqual(final['resolution'],'720p');self.assertEqual(final['input_references'],draft['input_references'])
        self.assertFalse(p.is_draft_request(final))
        with self.assertRaises(ValueError):p.build_final('J1',draft,'1080p')

class Facade(unittest.TestCase):
    def setUp(self):
        self.d=tempfile.TemporaryDirectory();self.p=Path(self.d.name);(self.p/'audio-manifest.json').write_text('{}')
        self.cfg={'GEN_BACKEND':'openrouter','OPENROUTER_API_KEY':'k','OPENROUTER_MEDIA_HOST':'tmpfiles','GEN_MAX_USD':'3'};self.posts=[]
        (self.p/'seedance-input.mp4').write_bytes(b'v')
        for i in range(3):(self.p/f'r{i}.png').write_bytes(b'i')
    def tearDown(self):self.d.cleanup()
    def post(self,url,**kw):
        self.posts.append((url,kw.get('json')));return resp({'id':'J1','polling_url':'u','status':'pending'})
    def run_submit(self,extra=None):
        cfg=self.cfg
        probes={'seedance-input.mp4':VIDEO,'r0.png':IMAGE,'r1.png':IMAGE,'r2.png':IMAGE}
        stack=[patch('mask_review.require_approved'),patch('mask_review.is_mesh',return_value=False),patch('audio_workflow.prepare_seedance_video',return_value=self.p/'seedance-input.mp4'),
               patch('config.setting',side_effect=lambda k,d='':cfg.get(k,d)),patch('seedance_bridge.setting',side_effect=lambda k,d='':cfg.get(k,d)),
               patch('seedance_bridge.probe',side_effect=lambda path:probes[Path(path).name]),patch('requests.get',return_value=resp(MODELS)),
               patch('media_host.upload',side_effect=lambda path,hosts=None:'https://tmpfiles.org/dl/1/'+Path(path).name),patch('requests.post',side_effect=self.post)]
        for s in stack:s.start()
        try:
            r=self.p
            return seedance_bridge.submit(r,'Replace the subject',str(r/'r0.png'),str(r/'r1.png') if extra is not None else None,status=lambda s:None,extra_references=extra)
        finally:
            for s in reversed(stack):s.stop()
    def test_three_references_and_duplicate_refusal(self):
        self.assertEqual(self.run_submit([str(self.p/'r2.png')]),'J1')
        url,body=self.posts[0];self.assertTrue(url.endswith('/api/v1/videos'))
        self.assertEqual([r['type'] for r in body['input_references']],['video_url']+['image_url']*3)
        self.assertEqual(json.loads((self.p/'provider-task.json').read_text()),{'provider':'openrouter','task_id':'J1'})
        self.assertEqual(json.loads((self.p/'seedance-request.json').read_text())['model'],'bytedance/seedance-2.5')
        with self.assertRaisesRegex(ValueError,'already has a provider response'):self.run_submit([str(self.p/'r2.png')])
        self.assertEqual(len(self.posts),1)
    def test_too_many_references_refused_before_upload(self):
        self.cfg['OPENROUTER_MAX_REF_IMAGES']='2'
        with self.assertRaisesRegex(ValueError,'at most 2'):self.run_submit([str(self.p/'r2.png')])
        self.assertEqual(self.posts,[])
    def test_cost_cap_blocks_before_upload(self):
        self.cfg['GEN_MAX_USD']='0.1'
        with self.assertRaisesRegex(ValueError,'GEN_MAX_USD'):self.run_submit([str(self.p/'r2.png')])
        self.assertEqual(self.posts,[])

class Cli(unittest.TestCase):
    def test_reference_is_repeatable_and_compatible_with_second_reference(self):
        import ast
        src=(ROOT/'workflow.py').read_text()
        self.assertIn("action='append'",src)
        # drive the real argument parser and submit call with the facade stubbed
        calls={}
        stub=type(sys)('seedance_bridge');stub.submit=lambda *a,**k:calls.update(a=a,k=k) or 'RID';stub.submit_hd=None;stub.wait_and_finish=lambda *a,**k:None
        job='a'*32
        for argv,want in [(['--reference','x','--reference','y','--reference','z'],('x','y',['z'])),
                          (['--reference','x','--second-reference','y'],('x','y',None)),
                          (['--reference','x'],('x',None,None)),
                          (['--reference','x','--reference','z','--second-reference','y'],('x','y',['z']))]:
            with tempfile.TemporaryDirectory() as d:
                (Path(d)/'runs'/job).mkdir(parents=True);(Path(d)/'p.txt').write_text('hi')
                with patch.dict(sys.modules,{'seedance_bridge':stub,'gpu_backend':Mock(enabled=lambda:False)}),patch.object(sys,'argv',['workflow.py','submit',job,'--prompt-file',str(Path(d)/'p.txt'),*argv]),\
                     patch('config.ROOT',Path(d)),patch('config.load_env'):
                    try:exec(compile(src,'workflow.py','exec'),{'__name__':'__main__'})
                    except (SystemExit,FileNotFoundError):pass
                self.assertEqual((calls['a'][2],calls['a'][3],calls['k'].get('extra_references')),want,argv)
if __name__=='__main__':unittest.main()
