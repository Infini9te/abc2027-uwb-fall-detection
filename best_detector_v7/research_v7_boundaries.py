"""Six fixed boundary refinements, selected on val_00..02 only."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import uwb_detector as b
import uwb_search_v4 as v
HERE=Path(__file__).resolve().parent;ROOT=HERE/'research_v7'
def refine(rows,other,offsets,prob,duration,kind,weight):
 result=[]
 for row in rows:
  r=dict(row);a,z=r['t_start'],r['t_end'];target=None
  if kind=='forest':
   matches=[]
   for q in other:
    inter=max(0,min(z,q['t_end'])-max(a,q['t_start']));union=max(z,q['t_end'])-min(a,q['t_start'])
    if inter/union>=.1:matches.append((inter/union,q))
   if matches:
    q=max(matches,key=lambda x:x[0])[1];target=(q['t_start'],q['t_end'])
  else:
   ids=np.arange(max(0,int(a*10)),min(len(prob),int(z*10)+1));ids=ids[prob[ids]>.2]
   if len(ids):target=(float(np.median(ids*.1+.05-offsets[ids,0])),float(np.median(ids*.1+.05+offsets[ids,1])))
  if target:
   na=max(0,a+weight*np.clip(target[0]-a,-1,1));nz=min(duration,z+weight*np.clip(target[1]-z,-1,1))
   if na<nz:r.update(t_start=float(na),t_end=float(nz))
  result.append(r)
 return result
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.data/'src'));import abc2027.evaluate as ev
 cfg=json.loads((ROOT/'selected.json').read_text());probs=v.read_probs(ROOT/'selected_probabilities.npz');dur=ev.stream_durations(a.data/'streams/val');td={s:d for s,d in dur.items() if s in ['val_00','val_01','val_02']};dd={s:d for s,d in dur.items() if s not in td};gt=ev.load_ground_truth(a.data/'streams_events.csv',True)
 base={s:v.decode_rows(probs[s],s,d,cfg) for s,d in dur.items()};fc=json.loads((HERE/'forest_v6/selected.json').read_text());fp=v.read_probs(HERE/'forest_v6/selected_probabilities.npz');off=v.read_probs(ROOT/'boundary_tcn/boundaries.npz');bp=v.read_probs(ROOT/'boundary_tcn/probabilities.npz')
 def score(rows,ds):return ev.evaluate(gt,[r for s in ds for r in rows[s]],ds)
 best=(b.ranking_key(score(base,td)),None,base);trials=[]
 for kind in ['forest','boundary_tcn']:
  for weight in [.25,.5,.75]:
   rows={s:refine(base[s],v.decode_rows(fp[s],s,d,fc),off[s],bp[s],d,kind,weight) for s,d in dur.items()}
   key=b.ranking_key(score(rows,td));trials.append(dict(kind=kind,weight=weight,tuning_key=key));print(kind,weight,key,flush=True)
   if key>best[0]:best=(key,dict(kind=kind,weight=weight),rows)
 _,ref,rows=best;full=score(rows,dur);diag=score(rows,dd)
 promoted=ref is not None and b.ranking_key(full)>b.ranking_key(score(base,dur)) and diag['ranking_score']['f1_mean']>=score(base,dd)['ranking_score']['f1_mean']
 b.dump(ROOT/'boundary_trials.json',dict(trials=trials,selected=ref,promoted=promoted,full=full,diagnostic=diag))
 if promoted:
  cfg['refiner']=ref
  if ref['kind']=='forest':cfg['refiner']['config']=fc
  else:cfg['refiner']['component']=json.loads((ROOT/'boundary_tcn/manifest.json').read_text())['component']
  b.dump(ROOT/'selected.json',cfg)
 else:rows=base
 flat=[r for s in dur for r in rows[s]]
 b.score_save('final_validation',flat,dur,gt,ROOT,ev);b.score_save('final_diagnostic',[r for r in flat if r['stream_id'] in dd],dd,gt,ROOT,ev)
 print('Boundary refinement promoted:',promoted)
if __name__=='__main__':main()
