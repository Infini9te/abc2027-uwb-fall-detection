"""Reproducible train-only model search, tuning on val_00..02.

The remaining validation streams are previously seen diagnostics, not a test set.
No segments, source corpora, composition metadata or test labels are used.
"""
import argparse, hashlib, itertools, json, sys, time
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
import uwb_detector as b

HERE=Path(__file__).resolve().parent
ROOT=HERE/'search_v4'
SPECS=[
    dict(name='boost_long',kind='boost',features='base',iterations=600,leaves=23,l2=5,positive=3),
    dict(name='boost_context',kind='boost',features='context',iterations=400,leaves=23,l2=5,positive=3),
    dict(name='extra_trees',kind='extra',features='context',iterations=200,leaves=0,l2=0,positive=3),
    dict(name='motion_only',kind='boost',features='motion',iterations=400,leaves=23,l2=8,positive=3),
    dict(name='hard_negatives',kind='boost',features='base',iterations=400,leaves=31,l2=8,positive=3),
]

def transform(x,mode):
    if mode=='base':return x
    if mode=='motion':return x[:,96:]
    if mode=='context':
        q=np.concatenate([x[:,:96],x[:,-45:]],axis=1)
        return np.ascontiguousarray(np.concatenate([x,b.lag(q,3),b.lag(q,10)],axis=1))
    raise ValueError(mode)

def read_probs(path):
    with np.load(path) as z:return {s:z[s].copy() for s in z.files}

def train(a,ev):
    gt=ev.load_ground_truth(a.data/'streams_events.csv',True)
    gt=[e for e in gt if e['stream_id'].startswith('train_')]
    paths=sorted((a.data/'streams/train').glob('stream_*.h5'))
    assert len(paths)==13
    original=joblib.load(HERE/'run/model.joblib')
    arrays=[]
    rng=np.random.default_rng(2027)
    for path in paths:
        x,d=b.extract(path,HERE/'run/cache');sid=b.stream_id(path)
        y,valid,near=b.labels(len(x),[e for e in gt if e['stream_id']==sid])
        keep=valid&(near|(rng.random(len(x))<.25))
        hard=valid&((b.predict(original,x)>.1)|keep)
        arrays.append((x,y,keep,hard))
    for spec in SPECS:
        dest=ROOT/spec['name'];dest.mkdir(exist_ok=True)
        if (dest/'val_probabilities.npz').exists():
            print('Resume:',spec['name'],'already finished',flush=True);continue
        started=time.perf_counter()
        xx=[];yy=[]
        for x,y,keep,hard in arrays:
            mask=hard if spec['name']=='hard_negatives' else keep
            xx.append(transform(x,spec['features'])[mask]);yy.append(y[mask])
        x=np.concatenate(xx);y=np.concatenate(yy);del xx,yy
        print('Training',spec['name'],x.shape,flush=True)
        if spec['kind']=='extra':
            model=ExtraTreesClassifier(n_estimators=spec['iterations'],min_samples_leaf=3,max_features=.5,n_jobs=4,random_state=2027)
        else:
            model=HistGradientBoostingClassifier(max_iter=spec['iterations'],learning_rate=.05,max_leaf_nodes=spec['leaves'],min_samples_leaf=30,l2_regularization=spec['l2'],early_stopping=False,random_state=2027)
        model.fit(x,y,sample_weight=np.where(y==1,spec['positive'],1.));del x,y
        joblib.dump(model,dest/'model.joblib',compress=3)
        probs={}
        for path in sorted((a.data/'streams/val').glob('stream_*.h5')):
            x,d=b.extract(path,HERE/'run/cache')
            probs[b.stream_id(path)]=b.predict(model,transform(x,spec['features']))
        np.savez(dest/'val_probabilities.npz',**probs)
        b.dump(dest/'manifest.json',dict(spec=spec,elapsed_seconds=time.perf_counter()-started,training='train only',seed=2027,feature_version=b.VERSION))
        print('Completed',spec['name'],'seconds',round(time.perf_counter()-started),flush=True)

def component(name,path,kind='tree',features='base'):
    return dict(name=name,path=str(path),kind=kind,features=features)

def decode_rows(prob,sid,duration,config):
    rows=b.decode(prob,sid,duration,config['decoder'])
    for r in rows:r['t_alarm']=min(duration,r['t_alarm']+config.get('alarm_delay',0.))
    refinement=config.get('boundary')
    if refinement is None:return rows
    smooth=b.trailing(prob,config['decoder']['smooth'])
    for r in rows:
        start,end=r['t_start'],r['t_end']
        if refinement['fraction']:
            a=max(0,int(np.floor(start/.1)));z=min(len(smooth),int(np.ceil(end/.1)))
            if z>a:
                support=np.flatnonzero(smooth[a:z]>=refinement['fraction']*np.max(smooth[a:z]))
                if len(support):start=(a+int(support[0]))*.1;end=(a+int(support[-1])+1)*.1
        start=max(0.,start+refinement['start_shift']);end=min(duration,end+refinement['end_shift'])
        # Preserve every issued alarm; retain coarse boundaries if refinement collapses.
        if end>start:r['t_start']=float(start);r['t_end']=float(end)
    return rows

def select(a,ev):
    dur=ev.stream_durations(a.data/'streams/val')
    td={s:d for s,d in dur.items() if s in ['val_00','val_01','val_02']}
    dd={s:d for s,d in dur.items() if s in ['val_03','val_04']}
    gt=ev.load_ground_truth(a.data/'streams_events.csv',True)
    pool={};configs={};results=[]
    old=read_probs(HERE/'run/validation_probabilities.npz')
    temporal=read_probs(HERE/'temporal_v2/val_probabilities.npz')
    orig=component('original',HERE/'run/model.joblib')
    tcn=component('temporal_v2',HERE/'temporal_v2/model.pt','tcn')
    pool['blend_v3']={s:.7*old[s]+.3*temporal[s] for s in old}
    configs['blend_v3']=[dict(orig,weight=.7),dict(tcn,weight=.3)]
    for spec in SPECS:
        dest=ROOT/spec['name']
        if not (dest/'val_probabilities.npz').exists():continue
        pool[spec['name']]=read_probs(dest/'val_probabilities.npz')
        configs[spec['name']]=[dict(component(spec['name'],dest/'model.joblib',features=spec['features']),weight=1.)]
    for name in ['tcn16_seed2028','tcn16_seed2029','tcn16_delay1']:
        dest=ROOT/name
        if (dest/'val_probabilities.npz').exists():
            pool[name]=read_probs(dest/'val_probabilities.npz')
            configs[name]=[dict(component(name,dest/'model.pt','tcn'),weight=1.,delay=1. if name=='tcn16_delay1' else 0.)]
    def assess(name):
        probs=pool[name]
        for s in dur:
            assert probs[s].shape==old[s].shape and np.isfinite(probs[s]).all()
        cfg,trials=b.tune({s:probs[s] for s in td},td,gt,ev.evaluate)
        rows=[r for s in td for r in b.decode(probs[s],s,dur[s],cfg)]
        report=ev.evaluate(gt,rows,td)
        result=dict(name=name,decoder=cfg,ranking=report['ranking_score'],key=b.ranking_key(report),components=configs[name],alarm_delay=max(c.get('delay',0.) for c in configs[name]))
        results.append(result)
        b.dump(ROOT/(name+'_tuning.json'),dict(result=result,trials=trials))
        print('Tuning:',name,round(report['ranking_score']['f1_mean'],5),flush=True)
        return result
    single=[assess(name) for name in list(pool)]
    # A predeclared short ensemble list: each new model equally mixed with V3,
    # plus the two strongest individual new models. Only tuning scores select.
    for name in list(pool):
        if name=='blend_v3':continue
        combo=name+'_plus_v3'
        pool[combo]={s:.5*pool[name][s]+.5*pool['blend_v3'][s] for s in dur}
        configs[combo]=[dict(c,weight=c['weight']*.5) for c in configs[name]+configs['blend_v3']]
        assess(combo)
    top=sorted([r for r in single if r['name']!='blend_v3'],key=lambda r:r['key'],reverse=True)[:2]
    if len(top)==2:
        names=[r['name'] for r in top];combo='top_two'
        pool[combo]={s:.5*pool[names[0]][s]+.5*pool[names[1]][s] for s in dur}
        configs[combo]=[dict(c,weight=c['weight']*.5) for name in names for c in configs[name]]
        assess(combo)
    winner=max(results,key=lambda r:r['key'])
    b.dump(ROOT/'tuning_leaderboard.json',sorted(results,key=lambda r:r['key'],reverse=True))
    # Retrospective boundary revision after a coarse event closes. Alarm timestamps
    # remain the actual earlier trigger; this adds no alarm filtering or backdating.
    trials=[];best_key=tuple(winner['key']);best_boundary=None
    for fraction,left,right in itertools.product((0.,.35,.6),(-.3,0.,.3),(-.3,0.,.3)):
        boundary=dict(fraction=fraction,start_shift=left,end_shift=right)
        config=dict(winner,boundary=boundary)
        rows=[r for s in td for r in decode_rows(pool[winner['name']][s],s,dur[s],config)]
        report=ev.evaluate(gt,rows,td)
        trials.append(dict(boundary=boundary,ranking=report['ranking_score']))
        if b.ranking_key(report)>best_key:best_key=b.ranking_key(report);best_boundary=boundary
    winner=dict(winner,boundary=best_boundary,refined_tuning_key=best_key)
    b.dump(ROOT/'boundary_trials.json',trials)
    b.dump(ROOT/'selected.json',winner)
    p=pool[winner['name']]
    np.savez(ROOT/'selected_probabilities.npz',**p)
    rows=[r for s in dur for r in decode_rows(p[s],s,dur[s],winner)]
    full=b.score_save('selected_validation',rows,dur,gt,ROOT,ev)
    check=b.score_save('selected_diagnostic',[r for r in rows if r['stream_id'] in dd],dd,gt,ROOT,ev)
    previous=json.loads((HERE/'blend_v3/validation_report.json').read_text())
    previous_d=json.loads((HERE/'blend_v3/diagnostic_report.json').read_text())
    promoted=b.ranking_key(full)>b.ranking_key(previous) and check['ranking_score']['f1_mean']>=previous_d['ranking_score']['f1_mean']
    b.dump(ROOT/'decision.json',dict(selected=winner['name'],promoted=promoted,ranking=full['ranking_score'],diagnostic=check['ranking_score'],previous=previous['ranking_score'],note='Validation reused; diagnostic is a rejection gate, not independent evidence. No guarantee on hidden test.'))
    print('SELECTED',winner['name'],'PROMOTED',promoted,flush=True)

def predict(a,ev):
    import torch
    import uwb_temporal_v2 as t
    torch.set_num_threads(4)
    cfg=json.loads((ROOT/'selected.json').read_text())
    loaded=[]
    for c in cfg['components']:
        model_path=Path(c['path'])
        if not model_path.is_absolute():model_path=HERE/model_path
        if c['kind']=='tree':model=joblib.load(model_path);ck=None
        else:
            ck=torch.load(model_path,map_location='cpu',weights_only=False)
            model=t.TemporalModel(ck['features']);model.load_state_dict(ck['state']);model.eval()
        loaded.append((c,model,ck))
    rows=[];dur={};probs={};started=time.perf_counter()
    paths=sorted((a.data/'streams'/a.split).glob('stream_*.h5'))
    if not paths:raise FileNotFoundError('No streams found')
    for path in paths:
        cache=a.cache if a.cache is not None else HERE/'run/cache';cache.mkdir(parents=True,exist_ok=True)
        x,d=b.extract(path,cache);sid=b.stream_id(path);dur[sid]=d
        p=np.zeros(len(x))
        for c,model,ck in loaded:
            q=b.predict(model,transform(x,c['features'])) if c['kind']=='tree' else t.align_delayed(t.probabilities(model,x,ck['mean'],ck['std']),ck.get('label_delay',0))
            p+=c['weight']*q
        probs[sid]=p;rows.extend(decode_rows(p,sid,d,cfg))
        print('Predicted',sid,flush=True)
    dest=ROOT/(a.split+'_submission.csv');b.write_csv(dest,rows);ev.validate_predictions(dest,dur)
    if a.split=='val':
        reference=read_probs(ROOT/'selected_probabilities.npz')
        for sid in probs:np.testing.assert_allclose(probs[sid],reference[sid],rtol=1e-6,atol=1e-7)
        b.score_save('replay',rows,dur,ev.load_ground_truth(a.data/'streams_events.csv',True),ROOT,ev)
        print('PASS: all model probabilities reproduced.',flush=True)
    print('Output:',dest,'elapsed seconds:',round(time.perf_counter()-started,2),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['train','select','predict']);p.add_argument('--data',type=Path,required=True);p.add_argument('--split',choices=['val','test'],default='val');p.add_argument('--cache',type=Path);a=p.parse_args()
    ROOT.mkdir(exist_ok=True)
    sys.path.insert(0,str(a.data/'src'));import abc2027.evaluate as ev
    if a.command=='train':
        z=np.random.default_rng(0).normal(size=(100,516)).astype(np.float32)
        for mode in ['base','context','motion']:np.testing.assert_array_equal(transform(z,mode)[:50],transform(z[:50],mode))
        b.dump(ROOT/'search_plan.json',dict(specs=SPECS,neural_runs=['16 epochs seed2028','16 epochs seed2029'],selection='val_00..02 only; then diagnostic rejection gate',evaluator_sha256=hashlib.sha256((a.data/'src/abc2027/evaluate.py').read_bytes()).hexdigest()))
    with threadpool_limits(limits=4):globals()[a.command](a,ev)

if __name__=='__main__':main()
