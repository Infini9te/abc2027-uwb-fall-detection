"""Compare 11 new models and fixed 50:50 V5 blends; retain V5 if gates fail."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import uwb_detector as b
import uwb_search_v4 as v
HERE=Path(__file__).resolve().parent
ROOT=HERE/'research_v7'
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);a=p.parse_args()
 sys.path.insert(0,str(a.data/'src'));import abc2027.evaluate as ev
 dur=ev.stream_durations(a.data/'streams/val');td={s:d for s,d in dur.items() if s in ['val_00','val_01','val_02']};dd={s:d for s,d in dur.items() if s not in td}
 gt=ev.load_ground_truth(a.data/'streams_events.csv',True)
 base=json.loads((HERE/'search_v5/selected.json').read_text());bp=v.read_probs(HERE/'search_v5/selected_probabilities.npz')
 def score(cfg,probs,ds):return ev.evaluate(gt,[r for s in ds for r in v.decode_rows(probs[s],s,dur[s],cfg)],ds)
 best=(b.ranking_key(score(base,bp,td)),base,bp);trials=[]
 for path in sorted(ROOT.glob('*/manifest.json')):
  m=json.loads(path.read_text());component=m['component'];probs=v.read_probs(path.parent/'probabilities.npz')
  for weight in (1.,.5):
   mix={s:weight*probs[s]+(1-weight)*bp[s] for s in dur}
   decoder,history=b.tune({s:mix[s] for s in td},td,gt,ev.evaluate)
   components=[dict(component,weight=weight)]+([dict(c,weight=c['weight']*(1-weight)) for c in base['components']] if weight<1 else [])
   cfg=dict(name=component['name']+('_plus_v5' if weight<1 else ''),decoder=decoder,boundary=None,alarm_delay=0.,components=components)
   report=score(cfg,mix,td);key=b.ranking_key(report)
   trials.append(dict(name=cfg['name'],ranking=report['ranking_score'],config=cfg,decoder_trials=history))
   print(cfg['name'],round(key[0],6),flush=True)
   if key>best[0]:best=(key,cfg,mix)
 _,cfg,mix=best
 rows=[r for s in dur for r in v.decode_rows(mix[s],s,dur[s],cfg)]
 full=b.score_save('candidate_validation',rows,dur,gt,ROOT,ev);diag=b.score_save('candidate_diagnostic',[r for r in rows if r['stream_id'] in dd],dd,gt,ROOT,ev)
 old=score(base,bp,dur);od=score(base,bp,dd)
 promoted=b.ranking_key(full)>b.ranking_key(old) and diag['ranking_score']['f1_mean']>=od['ranking_score']['f1_mean']
 b.dump(ROOT/'trials.json',trials);b.dump(ROOT/'candidate.json',cfg)
 decision=dict(candidate=cfg['name'],promoted=promoted,candidate_ranking=full['ranking_score'],candidate_diagnostic=diag['ranking_score'],v5_ranking=old['ranking_score'],v5_diagnostic=od['ranking_score'],note='22 configurations, 48 decoder trials each, selected on val_00..02. Reused validation and diagnostic are not independent test results.')
 if not promoted:cfg,mix=base,bp
 b.dump(ROOT/'selected.json',cfg);np.savez(ROOT/'selected_probabilities.npz',**mix);b.dump(ROOT/'decision.json',decision)
 print(json.dumps(decision,indent=2))
if __name__=='__main__':main()
