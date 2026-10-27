"""ABC2027: causal 10 Hz multi-radar event detector. See RUN_GUIDE.md."""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import h5py
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

VERSION = 'causal-sector-v1'
DT = 0.1
SERIALS = [29171, 29444, 30722]
FIELDS = ['stream_id', 't_start', 't_end', 'score', 't_alarm']

def log(msg):
    print(msg, flush=True)

def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')

def trailing(x, n):
    # Only current/past rows; no symmetric smoothing or stream-wide statistics.
    c = np.concatenate([np.zeros_like(x[:1], dtype=np.float64), np.cumsum(x, axis=0, dtype=np.float64)])
    end = np.arange(1, len(x)+1)
    begin = np.maximum(0, end-n)
    den = (end-begin).reshape((-1,) + (1,)*(x.ndim-1))
    return ((c[end]-c[begin])/den).astype(np.float32)

def lag(x, n):
    return x[np.maximum(0, np.arange(len(x))-n)]

def sector(x):
    return x.reshape(len(x), 3, 16, 23).mean(axis=-1)

def make_features(profiles, variances):
    features = [np.arcsinh(sector(profiles)*1e5).reshape(len(profiles), -1),
                np.log1p(sector(profiles**2)*1e10).reshape(len(profiles), -1)]
    summaries = []
    for energy in [variances] + [(profiles-lag(profiles,n))**2 for n in (1,3,10,30)]:
        bands = sector(energy)
        features.append(np.log1p(bands*1e10).reshape(len(profiles), -1))
        total = energy.mean(axis=2)
        peak = energy.max(axis=2)
        mass = energy.sum(axis=2) + 1e-20
        center = (energy*np.linspace(0,1,368,dtype=np.float32)).sum(axis=2)/mass
        summaries.append(np.concatenate([np.log1p(total*1e10), np.log1p(peak*1e10), center],axis=1))
    summary = np.concatenate(summaries,axis=1)
    features.extend([summary, trailing(summary,3), trailing(summary,10), trailing(summary,30)])
    return np.ascontiguousarray(np.concatenate(features,axis=1),dtype=np.float32)

def stream_id(path):
    return path.parent.name+'_'+path.stem.split('_')[-1]

def extract(path, cache):
    stat = path.stat()
    key = hashlib.sha256((VERSION+str(path.resolve())+str(stat.st_size)+str(stat.st_mtime_ns)).encode()).hexdigest()[:16]
    dest = cache / (stream_id(path)+'_'+key+'.npz')
    if dest.exists():
        with np.load(dest) as z:
            return z['x'], float(z['duration'])
    log('Extracting '+stream_id(path))
    profiles, variances = [], []
    with h5py.File(path,'r') as h:
        ds = h['cir']
        if len(ds.shape)!=3 or ds.shape[0]!=3 or ds.shape[2]!=368:
            raise ValueError(f'Unexpected CIR shape: {ds.shape}')
        serials = list(map(int,h['radar_serials'][:]))
        order = [serials.index(s) for s in SERIALS]
        count = ds.shape[1]
        for a in range(0,count,5000):
            block = np.asarray(ds[:,a:min(a+5000,count),:],dtype=np.float32)[order]
            if not np.isfinite(block).all():
                raise ValueError(f'Non-finite signal in {path} at frame {a}')
            # Repeat the last observed frame for a partial final 0.1 s block.
            pad = (-block.shape[1])%5
            if pad:
                block = np.concatenate([block,np.repeat(block[:,-1:,:],pad,axis=1)],axis=1)
            b = block.reshape(3,-1,5,368)
            profiles.append(b.mean(axis=2).transpose(1,0,2))
            variances.append(b.var(axis=2).transpose(1,0,2))
        if len(h['t'])!=count:
            raise ValueError('Timestamp count mismatch')
        ts=h['t'][:]
        if not np.allclose(ts, np.arange(count)/50,atol=1e-5):
            raise ValueError('Expected official zero-based 50 Hz time grid')
    x=make_features(np.concatenate(profiles),np.concatenate(variances))
    duration=count/50
    np.savez(dest,x=x,duration=duration)
    return x,duration

def labels(n,events):
    t=np.arange(n)*DT+DT/2
    y=np.zeros(n,dtype=np.uint8)
    valid=np.ones(n,dtype=bool)
    near=np.zeros(n,dtype=bool)
    for e in events:
        if e['ignore']:
            valid[(t>=e['t_start']-2)&(t<=e['t_end']+2)]=False
        else:
            y[(t>=e['t_start'])&(t<e['t_end'])]=1
            near|=(t>=e['t_start']-5)&(t<=e['t_end']+5)
    valid[y==1]=True  # evaluable event wins over an overlapping ignore zone
    return y,valid,near

def decode(prob, sid, duration, cfg):
    p=trailing(prob,cfg['smooth'])
    active=np.flatnonzero(p>=cfg['low'])
    if not len(active): return []
    gaps=np.flatnonzero(np.diff(active)>cfg['gap']+1)
    starts=np.r_[active[0],active[gaps+1]]
    ends=np.r_[active[gaps],active[-1]]+1
    rows=[]
    for a,b in zip(starts,ends):
        # The first two consecutive high frames actually trigger the alarm.
        high=p[a:b]>=cfg['high']
        triggers=np.flatnonzero(high[:-1]&high[1:])
        if not len(triggers): continue
        alarm=min(duration,(a+int(triggers[0])+2)*DT)
        start=max(0.,a*DT+cfg['start_shift'])
        end=min(duration,b*DT+cfg['end_shift'])
        if end<=start: continue
        rows.append(dict(stream_id=sid,t_start=float(start),t_end=float(end),
                         score=float(np.max(p[a:b])),t_alarm=float(alarm)))
    return rows

def write_csv(path,rows):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS)
        w.writeheader(); w.writerows(rows)

def predict(model,x):
    return np.concatenate([model.predict_proba(x[i:i+10000])[:,1] for i in range(0,len(x),10000)])

def ranking_key(report):
    r=report['ranking_score']
    error=r['start_error_median_abs']
    return (r['f1_mean'],-(error if error is not None else 1e9),-r['false_alarms_per_hour'])

def tune(probs,durations,gt,evaluate):
    best=None; records=[]
    # Small declared grid, not thousands of unconstrained validation trials.
    for high,ratio,gap,smooth in itertools.product((.25,.4,.55,.7),(.5,.75),(2,5),(1,3)):
        cfg=dict(high=high,low=high*ratio,gap=gap,smooth=smooth,start_shift=0.,end_shift=0.)
        rows=[e for sid,p in probs.items() for e in decode(p,sid,durations[sid],cfg)]
        report=evaluate(gt,rows,durations)
        records.append(dict(config=cfg,**report['ranking_score']))
        if best is None or ranking_key(report)>ranking_key(best[1]): best=(cfg,report)
    # Boundary correction is tuned ONLY on val_00..02, never on val_03..04.
    base=best[0]
    for left,right in itertools.product((-.3,-.15,0.,.15),(-.3,-.15,0.,.15)):
        cfg=dict(base,start_shift=left,end_shift=right)
        rows=[e for sid,p in probs.items() for e in decode(p,sid,durations[sid],cfg)]
        report=evaluate(gt,rows,durations)
        records.append(dict(config=cfg,**report['ranking_score']))
        if ranking_key(report)>ranking_key(best[1]): best=(cfg,report)
    return best[0],records

def score_save(name,rows,durations,gt,out,official):
    path=out/(name+'_predictions.csv'); write_csv(path,rows)
    checked=official.validate_predictions(path,durations)
    report=official.evaluate(gt,checked,durations)
    report['scope_note']='Validation diagnostic; not an estimate of hidden test performance.'
    dump(out/(name+'_report.json'),report)
    p=report['primary']; r=report['ranking_score']
    log(f"{name}: ranking_f1_mean={r['f1_mean']:.4f} F1@0.3={p['f1']:.4f} TP={p['tp']} FP={p['fp']} FN={p['fn']} start_MAE={p['start_error_mean_abs']} end_MAE={p['end_error_mean_abs']} latency={p['latency_median']}")
    # Inspect matches and mistakes without changing inference inputs.
    errors=[]
    for sid in durations:
        result=official._match_stream([e for e in gt if e['stream_id']==sid], [e for e in checked if e['stream_id']==sid], .3)
        used_p=set(); used_g=set()
        for pi,gi,iou in result['pairs']:
            p=result['predictions'][pi]; g=result['scored'][gi]
            used_p.add(pi); used_g.add(gi)
            errors.append(dict(kind='match',stream_id=sid,tiou=iou,start_error=p['t_start']-g['t_start'],end_error=p['t_end']-g['t_end'],duration_error=(p['t_end']-p['t_start'])-(g['t_end']-g['t_start']),prediction=p,truth=g))
        for gi,g in enumerate(result['scored']):
            if gi not in used_g: errors.append(dict(kind='miss',stream_id=sid,truth=g))
        for pi,p in enumerate(result['predictions']):
            if pi not in used_p:
                ignored=any(e['ignore'] and e['stream_id']==sid and min(p['t_end'],e['t_end']+2)>max(p['t_start'],e['t_start']-2) for e in gt)
                errors.append(dict(kind='ignored_prediction' if ignored else 'false_alarm',stream_id=sid,prediction=p))
    dump(out/(name+'_errors.json'),errors)
    return report

def train(args,official):
    out=args.out; cache=out/'cache'; cache.mkdir(parents=True,exist_ok=True)
    gt=official.load_ground_truth(args.data/'streams_events.csv',ignore_flagged=True)
    rng=np.random.default_rng(2027)
    xs=[]; ys=[]; manifest=[]
    train_paths=sorted((args.data/'streams/train').glob('stream_*.h5'))
    val_paths=sorted((args.data/'streams/val').glob('stream_*.h5'))
    if len(train_paths)!=13 or len(val_paths)!=5:
        raise ValueError('Expected 13 train and 5 validation streams in this release')
    for path in train_paths:
        x,duration=extract(path,cache); sid=stream_id(path)
        y,valid,near=labels(len(x),[e for e in gt if e['stream_id']==sid])
        # Keep every fall and surrounding hard negative, plus 25% of distant ADL.
        keep=valid & (near | (rng.random(len(x))<.25))
        xs.append(x[keep]); ys.append(y[keep])
        manifest.append(dict(stream_id=sid,frames=len(x),selected=int(keep.sum()),positives=int(y[keep].sum()),duration=duration))
        log(f'{sid}: selected {keep.sum()} time steps, positives {y[keep].sum()}')
    x=np.concatenate(xs); y=np.concatenate(ys); del xs,ys
    log(f'Training on {len(y)} rows, {x.shape[1]} features; positive frames={y.sum()}')
    model=HistGradientBoostingClassifier(max_iter=args.iterations,learning_rate=.07,max_leaf_nodes=23,
          min_samples_leaf=30,l2_regularization=3.,early_stopping=False,random_state=2027)
    # Avoid a random adjacent-window validation split. The only fit labels are train_*.
    weights=np.where(y==1,3.,1.)
    model.fit(x,y,sample_weight=weights)
    del x,y
    joblib.dump(model,out/'model.joblib')
    probs={}; durations={}
    for path in val_paths:
        x,duration=extract(path,cache); sid=stream_id(path)
        probs[sid]=predict(model,x); durations[sid]=duration
        log('Predicted '+sid)
    np.savez(out/'validation_probabilities.npz',**probs)
    tuning_ids=['val_00','val_01','val_02']; check_ids=['val_03','val_04']
    cfg,trials=tune({s:probs[s] for s in tuning_ids},{s:durations[s] for s in tuning_ids},gt,official.evaluate)
    dump(out/'decoder.json',cfg); dump(out/'tuning_trials.json',trials)
    rows=[e for sid,p in probs.items() for e in decode(p,sid,durations[sid],cfg)]
    score_save('tuning', [e for e in rows if e['stream_id'] in tuning_ids],{s:durations[s] for s in tuning_ids},gt,out,official)
    score_save('holdout', [e for e in rows if e['stream_id'] in check_ids],{s:durations[s] for s in check_ids},gt,out,official)
    score_save('validation',rows,durations,gt,out,official)
    # Also report all annotations so ignored events cannot conceal the scope.
    score_save('validation_all_labels',rows,durations,official.load_ground_truth(args.data/'streams_events.csv'),out,official)
    per_stream={s:official.evaluate(gt,[e for e in rows if e['stream_id']==s],{s:d}) for s,d in durations.items()}
    dump(out/'per_stream_report.json',per_stream)
    sample=args.data/'sample_submission.csv'
    if sample.exists():
        baseline=official.validate_predictions(sample,durations)
        score_save('official_sample_baseline',baseline,durations,gt,out,official)
    import sklearn,scipy
    dump(out/'run_manifest.json',dict(feature_version=VERSION,training=manifest,training_split='train only',
         tuning_streams=tuning_ids,holdout_streams=check_ids,
         holdout_warning='Stream holdout, not an additional subject-disjoint split. Repeated ADL sources may occur across validation streams.',
         features='Causal multi-scale range-sector features. Static CIR context included; fixed mat locations may inflate generalization.',
         alarm='First two consecutive high-probability 0.1 s frames. Signal-time latency only; computation not included. End finalized after the merge gap.',
         forbidden_inputs_used=False,iterations=args.iterations,python=sys.version,
         numpy=np.__version__,sklearn=sklearn.__version__,scipy=scipy.__version__,h5py=h5py.__version__,
         evaluator_sha256=hashlib.sha256((args.data/'src/abc2027/evaluate.py').read_bytes()).hexdigest()))
    log('DONE. Send validation_report.json, holdout_report.json and run_manifest.json.')

def infer(args,official):
    model=joblib.load(args.out/'model.joblib')
    cfg=json.loads((args.out/'decoder.json').read_text())
    cache=args.out/'cache'; cache.mkdir(exist_ok=True)
    paths=sorted((args.data/'streams'/args.split).glob('stream_*.h5'))
    if not paths: raise FileNotFoundError('No streams found for '+args.split)
    rows=[]; durations={}
    for path in paths:
        x,d=extract(path,cache); sid=stream_id(path); durations[sid]=d
        rows.extend(decode(predict(model,x),sid,d,cfg))
    dest=args.out/(args.split+'_submission.csv'); write_csv(dest,rows)
    official.validate_predictions(dest,durations)
    log(f'Validated {len(rows)} predictions: {dest}')

def self_test(official):
    rng=np.random.default_rng(1)
    p=rng.normal(0,1e-4,(100,3,368)).astype(np.float32)
    v=np.full_like(p,1e-10)
    np.testing.assert_allclose(make_features(p,v)[:60],make_features(p[:60],v[:60]))
    cfg=dict(high=.6,low=.3,gap=2,smooth=1,start_shift=0.,end_shift=0.)
    prob=np.zeros(100); prob[20:35]=.9
    rows=decode(prob,'val_00',10.,cfg)
    assert len(rows)==1 and abs(rows[0]['t_start']-2)<1e-9
    assert abs(rows[0]['t_end']-3.5)<1e-9 and abs(rows[0]['t_alarm']-2.2)<1e-9
    for end in (23,30,36,50):
        assert decode(prob[:end],'val_00',end*DT,cfg)[0]['t_alarm']==rows[0]['t_alarm']
    gt=[dict(stream_id='val_00',t_start=2.,t_end=3.5,ignore=False)]
    r=official.evaluate(gt,rows,{'val_00':10.})
    assert abs(r['ranking_score']['f1_mean']-1)<1e-9
    assert decode(np.zeros(10),'val_00',1.,cfg)==[]
    y,valid,near=labels(100,[dict(stream_id='val_00',t_start=2.,t_end=4.,ignore=True)])
    assert not valid[20] and y.sum()==0
    log('Self-test PASS: causal features, alarm timing, interval decoding, ignore mask, official scoring.')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['train','predict','self-test'])
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'run')
    p.add_argument('--split',choices=['val','test'],default='test')
    p.add_argument('--iterations',type=int,default=240)
    p.add_argument('--threads',type=int,default=4)
    args=p.parse_args(); args.data=args.data.resolve(); args.out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(args.data/'src'))
    import abc2027.evaluate as official
    with threadpool_limits(limits=args.threads):
        if args.command=='self-test': self_test(official)
        elif args.command=='train': train(args,official)
        else: infer(args,official)

if __name__=='__main__': main()
