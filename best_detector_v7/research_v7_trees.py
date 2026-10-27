"""New boosting families. Early stopping uses training streams 10..12 only.
Refit all train streams for the selected iteration count; never fit on val labels.
"""
import argparse,json,sys,time
from pathlib import Path
import joblib,numpy as np
from threadpoolctl import threadpool_limits
import uwb_detector as b

HERE=Path(__file__).resolve().parent
ROOT=HERE/'research_v7'
SPECS=[('xgb_motion4','xgb','motion',4),('xgb_motion6','xgb','motion',6),('xgb_base6','xgb','base',6),
       ('lgb_motion15','lgb','motion',15),('lgb_motion31','lgb','motion',31),('lgb_base31','lgb','base',31),
       ('cat_motion6','cat','motion',6),('cat_base6','cat','base',6)]

def features(x,mode):return x[:,96:] if mode=='motion' else x

def load_train(data,ev):
    gt=ev.load_ground_truth(data/'streams_events.csv',True);rng=np.random.default_rng(2027);streams=[]
    for path in sorted((data/'streams/train').glob('stream_*.h5')):
        x,d=b.extract(path,HERE/'run/cache');sid=b.stream_id(path)
        y,valid,near=b.labels(len(x),[e for e in gt if e['stream_id']==sid])
        keep=valid&(near|(rng.random(len(x))<.25))
        streams.append(dict(x=x,y=y,valid=valid,keep=keep,sid=sid,duration=d))
    assert len(streams)==13
    return streams,gt

def make_model(kind,size,rounds,probe):
    if kind=='xgb':
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=rounds,max_depth=size,learning_rate=.05,min_child_weight=10,subsample=.8,colsample_bytree=.8,reg_lambda=8,reg_alpha=.1,tree_method='hist',device='cpu',n_jobs=4,random_state=2027,eval_metric='logloss',early_stopping_rounds=50 if probe else None)
    if kind=='lgb':
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=rounds,num_leaves=size,learning_rate=.05,min_child_samples=40,subsample=.8,subsample_freq=1,colsample_bytree=.8,reg_lambda=8,reg_alpha=.1,verbosity=-1,n_jobs=4,random_state=2027,deterministic=True,force_col_wise=True)
    from catboost import CatBoostClassifier
    return CatBoostClassifier(iterations=rounds,depth=size,learning_rate=.05,l2_leaf_reg=8,random_strength=1,rsm=.8,thread_count=4,random_seed=2027,loss_function='Logloss',allow_writing_files=False,verbose=100)

def fit(model,kind,x,y,probe=None):
    kwargs=dict(sample_weight=np.where(y==1,3.,1.))
    if probe is not None:
        kwargs['eval_set']=[probe] if kind!='cat' else probe
        if kind=='lgb':
            import lightgbm as lgb
            kwargs['callbacks']=[lgb.early_stopping(50,verbose=False),lgb.log_evaluation(100)]
        elif kind=='xgb':kwargs['verbose']=100
        else:kwargs.update(early_stopping_rounds=50,use_best_model=True)
    elif kind=='xgb':kwargs['verbose']=False
    model.fit(x,y,**kwargs)

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);a=p.parse_args()
    ROOT.mkdir(exist_ok=True);sys.path.insert(0,str(a.data/'src'));import abc2027.evaluate as ev
    streams,gt=load_train(a.data,ev)
    b.dump(ROOT/'tree_plan.json',dict(specifications=SPECS,maximum_rounds=600,early_stop_patience=50,probe_train='train_00..09',probe_validation='train_10..12',final_fit='all 13 training streams',note='Internal streams can share source subjects. This is not subject-disjoint cross-validation.'))
    for name,kind,mode,size in SPECS:
        out=ROOT/name;out.mkdir(exist_ok=True)
        if (out/'probabilities.npz').exists():print('Resume:',name,flush=True);continue
        start=time.perf_counter();print('\nPROBE',name,flush=True)
        x=np.concatenate([features(s['x'],mode)[s['keep']] for s in streams[:10]])
        y=np.concatenate([s['y'][s['keep']] for s in streams[:10]])
        vx=np.concatenate([features(s['x'],mode)[s['valid']] for s in streams[10:]])
        vy=np.concatenate([s['y'][s['valid']] for s in streams[10:]])
        model=make_model(kind,size,600,True);fit(model,kind,x,y,(vx,vy))
        rounds=int(model.best_iteration+1) if kind=='xgb' else int(model.best_iteration_) if kind=='lgb' else int(model.get_best_iteration()+1)
        rounds=max(1,rounds)
        probe_rows=[];probe_d={}
        cfg=dict(high=.4,low=.2,gap=5,smooth=1,start_shift=0.,end_shift=0.)
        for s in streams[10:]:
            prob=b.predict(model,features(s['x'],mode));probe_d[s['sid']]=s['duration'];probe_rows.extend(b.decode(prob,s['sid'],s['duration'],cfg))
        probe_report=ev.evaluate(gt,probe_rows,probe_d)
        del x,y,vx,vy,model
        print('REFIT',name,'rounds',rounds,'internal event F1',round(probe_report['ranking_score']['f1_mean'],4),flush=True)
        x=np.concatenate([features(s['x'],mode)[s['keep']] for s in streams]);y=np.concatenate([s['y'][s['keep']] for s in streams])
        model=make_model(kind,size,rounds,False);fit(model,kind,x,y);del x,y
        joblib.dump(model,out/'model.joblib',compress=3)
        probs={}
        for path in sorted((a.data/'streams/val').glob('stream_*.h5')):
            x,d=b.extract(path,HERE/'run/cache');probs[b.stream_id(path)]=b.predict(model,features(x,mode))
        np.savez(out/'probabilities.npz',**probs)
        b.dump(out/'manifest.json',dict(name=name,kind=kind,features=mode,rounds=rounds,elapsed_seconds=time.perf_counter()-start,internal_report=probe_report,component=dict(name=name,path=str(out/'model.joblib'),kind='tree',features=mode,weight=1.)))
        print('DONE',name,'seconds',round(time.perf_counter()-start),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
