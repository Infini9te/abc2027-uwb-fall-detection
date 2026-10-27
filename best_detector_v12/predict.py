"""Frozen V12 inference; each stream is processed independently."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import joblib,numpy as np,torch
from threadpoolctl import threadpool_limits
import uwb_detector as b
import uwb_temporal_v2 as t
import iq_features as iq
HERE=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--split',choices=['val','test'],default='val');p.add_argument('--evaluator-package',type=Path);p.add_argument('--out',type=Path);a=p.parse_args()
 root=a.evaluator_package or a.data;sys.path.insert(0,str(root/'src'));import abc2027.evaluate as ev
 cfg=json.loads((HERE/'selected.json').read_text());out=a.out or HERE/'results';cache=out/'cache';cache.mkdir(parents=True,exist_ok=True)
 manifest=json.loads((HERE/'manifest.json').read_text())
 for rel,sha in manifest.items():assert hashlib.sha256((HERE/rel).read_bytes()).hexdigest()==sha,'Package changed: '+rel
 loaded=[]
 for c in cfg['components']:
  path=HERE/c['path'];ck=None
  if c['kind']=='tree':m=joblib.load(path)
  else:
   ck=torch.load(path,map_location='cpu',weights_only=False);m=t.TemporalModel(ck['features']);m.load_state_dict(ck['state']);m.eval()
  loaded.append((c,m,ck))
 rows=[];durs={};probs={};start=time.time()
 for path in sorted((a.data/'streams'/a.split).glob('stream_*.h5')):
  raw,d=b.extract(path,cache);paired,d2=iq.extract(path,cache);assert d==d2 and len(raw)==len(paired);sid=b.stream_id(path);durs[sid]=d;prob=np.zeros(len(raw))
  for c,m,ck in loaded:
   if c['kind']=='tcn':q=t.align_delayed(t.probabilities(m,raw,ck['mean'],ck['std']),ck.get('label_delay',0))
   else:q=b.predict(m,iq.short_future(paired) if c['features']=='iq_short_future' else raw[:,96:] if c['features']=='motion' else raw)
   prob+=c['weight']*q
  probs[sid]=prob;rr=b.decode(prob,sid,d,cfg['decoder'])
  for r in rr:r['t_alarm']=min(d,r['t_alarm']+cfg['lookahead_seconds'])
  rows.extend(rr);print('Predicted',sid,flush=True)
 assert durs,'No streams found'
 dest=out/(a.split+'_submission.csv');b.write_csv(dest,rows);checked=ev.validate_predictions(dest,durs)
 info=dict(status='PASS official prediction validator',model='V12 short I/Q future-context blend',prediction_count=len(checked),submission=str(dest.resolve()),submission_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),streams=durs,official_test_score=None,independent_streams=True,elapsed_seconds=time.time()-start)
 if a.split=='val':
  with np.load(HERE/'validation_reference.npz') as z:
   for sid in probs:np.testing.assert_allclose(probs[sid],z[sid],rtol=1e-6,atol=1e-7)
  report=b.score_save('replay',checked,durs,ev.load_ground_truth(a.data/'streams_events.csv',True),out,ev);info['validation_ranking']=report['ranking_score'];info['probability_replay_pass']=True
  print('PASS packaged probability replay',flush=True)
 b.dump(out/'validation_report.json',info);print(json.dumps(info,indent=2),flush=True)
if __name__=='__main__':
 torch.set_num_threads(4)
 with threadpool_limits(limits=4):main()

