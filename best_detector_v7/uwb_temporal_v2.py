"""Independent causal temporal experiment on the original detector's features.
Only train_* labels fit model weights. val_00..02 tune decoding; val_03..04
are diagnostics already seen in earlier work, not an untouched holdout.
"""
import argparse,copy,json,random,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
import uwb_detector as base

BURN=128
LENGTH=256
RECEPTIVE=125

class Block(nn.Module):
    def __init__(self,d):
        super().__init__();self.d=d
        self.c1=nn.Conv1d(48,48,3,dilation=d);self.c2=nn.Conv1d(48,48,3,dilation=d)
        self.n1=nn.LayerNorm(48);self.n2=nn.LayerNorm(48);self.drop=nn.Dropout(.1)
    def forward(self,x):
        z=self.c1(F.pad(x,(2*self.d,0)))
        z=self.drop(F.gelu(self.n1(z.transpose(1,2)).transpose(1,2)))
        z=self.c2(F.pad(z,(2*self.d,0)))
        z=self.drop(F.gelu(self.n2(z.transpose(1,2)).transpose(1,2)))
        return x+z

class TemporalModel(nn.Module):
    def __init__(self,features=516):
        super().__init__();self.project=nn.Conv1d(features,48,1)
        self.blocks=nn.Sequential(*[Block(d) for d in (1,2,4,8,16)])
        self.head=nn.Conv1d(48,1,1)
    def forward(self,x):return self.head(self.blocks(F.gelu(self.project(x)))).squeeze(1)

class Windows(Dataset):
    def __init__(self,streams):
        self.streams=streams;self.index=[];self.weights=[]
        for si,(x,y,valid) in enumerate(streams):
            starts=list(range(0,len(x)-LENGTH+1,64))
            if starts[-1]!=len(x)-LENGTH:starts.append(len(x)-LENGTH)
            for start in starts:
                yy=y[start+BURN:start+LENGTH];vv=valid[start+BURN:start+LENGTH]
                if vv.sum()<32:continue
                self.index.append((si,start));self.weights.append(3. if np.any((yy==1)&vv) else 1.)
    def __len__(self):return len(self.index)
    def __getitem__(self,i):
        si,start=self.index[i];x,y,v=self.streams[si];end=start+LENGTH
        return torch.from_numpy(x[start:end].T.copy()),torch.from_numpy(y[start:end].astype(np.float32)),torch.from_numpy(v[start:end].astype(np.float32))

def probabilities(model,x,mean,std):
    model.eval();z=np.clip((x-mean)/std,-12,12).astype(np.float32);parts=[]
    with torch.no_grad():
        for start in range(0,len(z),2048):
            left=max(0,start-(RECEPTIVE-1));end=min(len(z),start+2048)
            p=torch.sigmoid(model(torch.from_numpy(z[left:end].T.copy()[None])))[0].numpy()
            parts.append(p[start-left:])
    return np.concatenate(parts)

def align_delayed(p,delay):
    if not delay:return p
    if delay>=len(p):return np.zeros_like(p)
    return np.concatenate([p[delay:],np.zeros(delay,dtype=p.dtype)])

def self_test():
    torch.manual_seed(1);m=TemporalModel(8).eval();x=torch.randn(1,8,400)
    with torch.no_grad():
        whole=m(x);prefix=m(x[:,:,:200])
        torch.testing.assert_close(whole[:,:200],prefix,rtol=1e-5,atol=1e-5)
        tail=m(x[:,:,200-(RECEPTIVE-1):])[:,RECEPTIVE-1:]
        torch.testing.assert_close(whole[:,200:],tail,rtol=1e-5,atol=1e-5)
    m.train();loss=m(x[:,:,:256]).square().mean();loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    print('PASS: no future-frame dependence, chunk equivalence, finite training gradients.',flush=True)

def train(a,ev):
    out=a.out;out.mkdir(parents=True,exist_ok=True)
    if (out/'model.pt').exists():raise FileExistsError('Model exists. Use predict or choose a new --out directory to preserve this run.')
    gt=ev.load_ground_truth(a.data/'streams_events.csv',ignore_flagged=True)
    arrays=[];ys=[];vs=[];total=0;sums=None;squares=None
    for path in sorted((a.data/'streams/train').glob('stream_*.h5')):
        x,d=base.extract(path,a.cache);sid=base.stream_id(path)
        y,v,_=base.labels(len(x),[e for e in gt if e['stream_id']==sid])
        if a.label_delay:
            y=np.concatenate([np.zeros(a.label_delay,dtype=y.dtype),y[:-a.label_delay]])
            v=np.concatenate([np.zeros(a.label_delay,dtype=bool),v[:-a.label_delay]])
        arrays.append(x);ys.append(y);vs.append(v)
        # Train-only scaler, excluding ignored zones.
        q=x[v].astype(np.float64);s=q.sum(axis=0);ss=(q*q).sum(axis=0)
        sums=s if sums is None else sums+s;squares=ss if squares is None else squares+ss;total+=len(q)
        print(sid,'frames',len(x),'positives',int(y.sum()),flush=True)
    if len(arrays)!=13:raise ValueError('Expected all 13 training streams')
    mean=(sums/total).astype(np.float32);std=np.sqrt(np.maximum(squares/total-(sums/total)**2,1e-6)).astype(np.float32)
    streams=[(np.clip((x-mean)/std,-12,12).astype(np.float32),y,v) for x,y,v in zip(arrays,ys,vs)];del arrays
    ds=Windows(streams);sampler=WeightedRandomSampler(ds.weights,num_samples=len(ds),replacement=True,generator=torch.Generator().manual_seed(a.seed))
    dl=DataLoader(ds,batch_size=24,sampler=sampler,num_workers=0)
    model=TemporalModel(len(mean));opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.001)
    best=float('inf');state=None;history=[];started=time.perf_counter()
    print('Training windows:',len(ds),'epochs:',a.epochs,'device: CPU',flush=True)
    for epoch in range(a.epochs):
        model.train();losses=[]
        for xb,yb,vb in dl:
            opt.zero_grad(set_to_none=True)
            logits=model(xb)[:,BURN:];target=yb[:,BURN:];mask=vb[:,BURN:]
            loss=F.binary_cross_entropy_with_logits(logits,target,pos_weight=torch.tensor(4.),reduction='none')
            loss=(loss*mask).sum()/mask.sum().clamp_min(1)
            if not torch.isfinite(loss):raise ValueError('Non-finite training loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach()))
        value=float(np.mean(losses));history.append(dict(epoch=epoch+1,loss=value))
        if value<best:best=value;state=copy.deepcopy(model.state_dict())
        print(f'Epoch {epoch+1}/{a.epochs}: train_loss={value:.5f}, elapsed={time.perf_counter()-started:.0f}s',flush=True)
    torch.save(dict(state=state,mean=mean,std=std,features=len(mean),history=history,seed=a.seed,label_delay=a.label_delay),out/'model.pt')
    base.dump(out/'manifest.json',dict(model='causal TCN',label_delay_frames=a.label_delay,feature_version=base.VERSION,receptive_frames=RECEPTIVE,epochs=a.epochs,seed=a.seed,training_labels='train only',normalization='train only',checkpoint_selection='training loss only',torch=torch.__version__,history=history,
        validation_note='Decoder tuned on val_00..02; val_03..04 already seen diagnostics, not independent holdout.',
        latency_note='Signal-time alarm simulation; bulk CPU inference runtime is not included. Not a deployed real-time service.'))
    infer(a,ev,tuning=True)

def infer(a,ev,tuning=False):
    ckpt=torch.load(a.out/'model.pt',map_location='cpu',weights_only=False)
    model=TemporalModel(ckpt['features']);model.load_state_dict(ckpt['state']);model.eval()
    split='val' if tuning else a.split;probs={};dur={}
    for path in sorted((a.data/'streams'/split).glob('stream_*.h5')):
        x,d=base.extract(path,a.cache);sid=base.stream_id(path)
        probs[sid]=align_delayed(probabilities(model,x,ckpt['mean'],ckpt['std']),ckpt.get('label_delay',0));dur[sid]=d
        print('Predicted',sid,flush=True)
    if not probs:raise FileNotFoundError('No streams for '+split)
    np.savez(a.out/(split+'_probabilities.npz'),**probs)
    if tuning:
        gt=ev.load_ground_truth(a.data/'streams_events.csv',True)
        ids=['val_00','val_01','val_02'];cfg,trials=base.tune({s:probs[s] for s in ids},{s:dur[s] for s in ids},gt,ev.evaluate)
        base.dump(a.out/'decoder.json',cfg);base.dump(a.out/'tuning_trials.json',trials)
    else:cfg=json.loads((a.out/'decoder.json').read_text())
    rows=[r for sid,p in probs.items() for r in base.decode(p,sid,dur[sid],cfg)]
    for r in rows:r['t_alarm']=min(dur[r['stream_id']],r['t_alarm']+ckpt.get('label_delay',0)*.1)
    base.write_csv(a.out/(split+'_submission.csv'),rows);ev.validate_predictions(a.out/(split+'_submission.csv'),dur)
    if tuning:
        full=base.score_save('validation',rows,dur,gt,a.out,ev)
        check=base.score_save('diagnostic',[r for r in rows if r['stream_id'] in ['val_03','val_04']],{s:d for s,d in dur.items() if s in ['val_03','val_04']},gt,a.out,ev)
        old=json.loads((a.baseline/'validation_report.json').read_text())
        old_check=json.loads((a.baseline/'holdout_report.json').read_text())
        summary=dict(baseline_f1=old['ranking_score']['f1_mean'],new_f1=full['ranking_score']['f1_mean'],baseline_diagnostic_f1=old_check['ranking_score']['f1_mean'],new_diagnostic_f1=check['ranking_score']['f1_mean'],
                     candidate_improves_both=full['ranking_score']['f1_mean']>old['ranking_score']['f1_mean'] and check['ranking_score']['f1_mean']>=old_check['ranking_score']['f1_mean'],baseline_unchanged=True)
        base.dump(a.out/'comparison.json',summary);print('\n=== SEND THIS OUTPUT ===\n'+json.dumps(summary,indent=2),flush=True)

def main():
    here=Path(__file__).resolve().parent
    p=argparse.ArgumentParser();p.add_argument('command',choices=['self-test','train','predict']);p.add_argument('--data',type=Path);p.add_argument('--out',type=Path,default=here/'temporal_v2');p.add_argument('--cache',type=Path,default=here/'run/cache');p.add_argument('--baseline',type=Path,default=here/'run');p.add_argument('--epochs',type=int,default=8);p.add_argument('--seed',type=int,default=2027);p.add_argument('--threads',type=int,default=4);p.add_argument('--split',choices=['val','test'],default='test');p.add_argument('--label-delay',type=int,default=0,choices=[0,10]);a=p.parse_args()
    torch.set_num_threads(a.threads);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    if a.command=='self-test':self_test();return
    if a.data is None:p.error('--data required')
    a.cache.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(a.data/'src'));import abc2027.evaluate as ev
    if a.command=='train':train(a,ev)
    else:infer(a,ev)

if __name__=='__main__':main()
