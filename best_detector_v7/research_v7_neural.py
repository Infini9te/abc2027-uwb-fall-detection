"""Compact research adaptations, not reproductions of published benchmark models.
Recurrent, causal local attention, and boundary-supervised temporal models.
Epoch selection uses train_10..12; final model refits all thirteen train streams.
"""
import argparse,copy,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
import uwb_detector as b
from uwb_temporal_v2 import Block
from research_v7_trees import load_train,HERE,ROOT

class Model(nn.Module):
    def __init__(self,architecture,features=516):
        super().__init__();self.architecture=architecture
        self.project=nn.Conv1d(features,48,1)
        if architecture=='cnn_lstm':
            self.body=nn.LSTM(48,48,batch_first=True);self.head=nn.Linear(48,1)
        elif architecture=='local_attention':
            self.pre=nn.Conv1d(48,48,3)
            layer=nn.TransformerEncoderLayer(48,4,96,.1,activation='gelu',batch_first=True,norm_first=True)
            self.body=nn.TransformerEncoder(layer,2,enable_nested_tensor=False);self.head=nn.Linear(48,1)
        elif architecture=='boundary_tcn':
            self.body=nn.Sequential(*[Block(d) for d in (1,2,4,8,16)]);self.head=nn.Conv1d(48,3,1)
        else:raise ValueError(architecture)
    def forward(self,x,state=None):
        z=F.gelu(self.project(x))
        if self.architecture=='cnn_lstm':
            z,state=self.body(z.transpose(1,2),state)
            return self.head(z).transpose(1,2),state
        if self.architecture=='local_attention':
            z=F.gelu(self.pre(F.pad(z,(2,0)))).transpose(1,2)
            n=z.shape[1];i=torch.arange(n,device=z.device);delta=i[:,None]-i[None,:]
            z=self.body(z,mask=(delta<0)|(delta>31))
            return self.head(z).transpose(1,2),None
        return self.head(self.body(z)),None

def infer(model,x,mean,std):
    model.eval();z=np.clip((x-mean)/std,-12,12).astype(np.float32);parts=[];state=None
    chunk=256 if model.architecture=='local_attention' else 1024
    history=64 if model.architecture=='local_attention' else 124
    with torch.no_grad():
        for start in range(0,len(z),chunk):
            end=min(len(z),start+chunk)
            left=start if model.architecture=='cnn_lstm' else max(0,start-history)
            raw,state=model(torch.from_numpy(z[left:end].T.copy()[None]),state)
            parts.append(raw[0,:,start-left:].T.numpy())
    raw=np.concatenate(parts);prob=1/(1+np.exp(-np.clip(raw[:,0],-60,60)))
    offsets=np.logaddexp(0,raw[:,1:])*5 if raw.shape[1]>1 else None
    return prob,offsets

def self_test():
    torch.manual_seed(5)
    for architecture in ['cnn_lstm','local_attention','boundary_tcn']:
        model=Model(architecture,8).eval();x=torch.randn(1,8,360)
        with torch.no_grad():
            whole,_=model(x);prefix,state=model(x[:,:,:180]);torch.testing.assert_close(whole[:,:,:180],prefix,atol=3e-5,rtol=3e-5)
            if architecture=='cnn_lstm':tail,_=model(x[:,:,180:],state)
            else:
                history=64 if architecture=='local_attention' else 124
                tail,_=model(x[:,:,180-history:]);tail=tail[:,:,history:]
            torch.testing.assert_close(whole[:,:,180:],tail,atol=3e-5,rtol=3e-5)
        model.train();raw,_=model(x[:,:,:256]);loss=raw.square().mean();loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        print('PASS causality, chunk/state equivalence, gradients:',architecture,flush=True)

class Windows(Dataset):
    def __init__(self,streams,mean,std):
        self.streams=[];self.index=[];self.weights=[]
        for si,s in enumerate(streams):
            x=np.clip((s['x']-mean)/std,-12,12).astype(np.float32)
            self.streams.append((x,s['target'],s['valid']))
            for start in range(0,len(x)-255,64):
                v=s['valid'][start+128:start+256];y=s['y'][start+128:start+256]
                if v.sum()<32:continue
                self.index.append((si,start));self.weights.append(3. if np.any((y==1)&v) else 1.)
    def __len__(self):return len(self.index)
    def __getitem__(self,i):
        si,start=self.index[i];x,y,v=self.streams[si];end=start+256
        return torch.from_numpy(x[start:end].T.copy()),torch.from_numpy(y[start:end].T.copy()),torch.from_numpy(v[start:end].astype(np.float32))

def scaling(streams):
    sums=np.zeros(516,dtype=np.float64);squares=np.zeros(516,dtype=np.float64);count=0
    for s in streams:
        x=s['x'][s['valid']].astype(np.float64);sums+=x.sum(0);squares+=(x*x).sum(0);count+=len(x)
    mean=sums/count;std=np.sqrt(np.maximum(squares/count-mean**2,1e-6))
    return mean.astype(np.float32),std.astype(np.float32)

def fit_phase(architecture,streams,epochs,seed,ev,gt,probe=None):
    torch.manual_seed(seed);np.random.seed(seed)
    mean,std=scaling(streams);ds=Windows(streams,mean,std)
    sampler=WeightedRandomSampler(ds.weights,len(ds),replacement=True,generator=torch.Generator().manual_seed(seed))
    dl=DataLoader(ds,batch_size=24,sampler=sampler,num_workers=0)
    model=Model(architecture);opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.001)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=12,eta_min=.00002)
    history=[];best=None;started=time.perf_counter()
    cfg=dict(high=.4,low=.2,gap=5,smooth=1,start_shift=0.,end_shift=0.)
    for epoch in range(1,epochs+1):
        model.train();losses=[]
        for x,target,valid in dl:
            # Mild feature noise regularizes all candidates; no validation-derived augmentation.
            x=x+torch.randn_like(x)*.02
            opt.zero_grad(set_to_none=True);raw,_=model(x);raw=raw[:,:,128:];target=target[:,:,128:];valid=valid[:,128:]
            loss=F.binary_cross_entropy_with_logits(raw[:,0],target[:,0],pos_weight=torch.tensor(4.),reduction='none')
            loss=(loss*valid).sum()/valid.sum().clamp_min(1)
            if architecture=='boundary_tcn':
                positive=valid*target[:,0]
                boundary=F.smooth_l1_loss(F.softplus(raw[:,1:]),target[:,1:],reduction='none').mean(1)
                loss=loss+.5*(boundary*positive).sum()/positive.sum().clamp_min(1)
                logp=F.logsigmoid(raw[:,0]);smooth=(logp[:,1:]-logp[:,:-1].detach()).square().clamp(max=16)
                mask=valid[:,1:]*valid[:,:-1]
                loss=loss+.02*(smooth*mask).sum()/mask.sum().clamp_min(1)
            assert torch.isfinite(loss)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach()))
        sched.step();record=dict(epoch=epoch,loss=float(np.mean(losses)))
        if probe is not None and epoch%2==0:
            rows=[];dur={}
            for s in probe:
                prob,_=infer(model,s['x'],mean,std);dur[s['sid']]=s['duration'];rows.extend(b.decode(prob,s['sid'],s['duration'],cfg))
            report=ev.evaluate(gt,rows,dur);record['internal_ranking']=report['ranking_score']
            if best is None or b.ranking_key(report)>best[0]:best=(b.ranking_key(report),epoch)
        history.append(record)
        print(architecture,'probe' if probe else 'refit','epoch',epoch,'loss',round(record['loss'],5),'internal F1',record.get('internal_ranking',{}).get('f1_mean'),'elapsed',round(time.perf_counter()-started),flush=True)
    return model,mean,std,history,best

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path);p.add_argument('--self-test',action='store_true');p.add_argument('--architecture',choices=['cnn_lstm','local_attention','boundary_tcn','all'],default='all');a=p.parse_args();torch.set_num_threads(3)
    if a.self_test:self_test();return
    if a.data is None:p.error('--data required')
    ROOT.mkdir(exist_ok=True);sys.path.insert(0,str(a.data/'src'));import abc2027.evaluate as ev
    streams,gt=load_train(a.data,ev)
    for s in streams:
        target=np.zeros((len(s['x']),3),dtype=np.float32);target[:,0]=s['y'];times=np.arange(len(target))*.1+.05
        for e in gt:
            if e['stream_id']!=s['sid'] or e['ignore']:continue
            mask=(times>=e['t_start'])&(times<e['t_end'])
            target[mask,1]=(times[mask]-e['t_start'])/5;target[mask,2]=(e['t_end']-times[mask])/5
        s['target']=target
    names=['cnn_lstm','local_attention','boundary_tcn'] if a.architecture=='all' else [a.architecture]
    for architecture in names:
        out=ROOT/architecture;out.mkdir(exist_ok=True)
        if (out/'probabilities.npz').exists():print('Resume',architecture,flush=True);continue
        model,mean,std,probe_history,best=fit_phase(architecture,streams[:10],12,2031,ev,gt,probe=streams[10:])
        epochs=best[1];del model
        print('SELECTED EPOCHS',architecture,epochs,flush=True)
        model,mean,std,history,_=fit_phase(architecture,streams,epochs,2031,ev,gt)
        torch.save(dict(architecture=architecture,state=model.state_dict(),mean=mean,std=std,features=516,epochs=epochs,seed=2031),out/'model.pt')
        probs={};bounds={}
        for path in sorted((a.data/'streams/val').glob('stream_*.h5')):
            x,d=b.extract(path,HERE/'run/cache');sid=b.stream_id(path);probs[sid],offsets=infer(model,x,mean,std)
            if offsets is not None:bounds[sid]=offsets
        np.savez(out/'probabilities.npz',**probs)
        if bounds:np.savez(out/'boundaries.npz',**bounds)
        b.dump(out/'manifest.json',dict(architecture=architecture,selected_epochs=epochs,probe_history=probe_history,refit_history=history,selection='train_10..12 event F1, fixed decoder; refit train_00..12',component=dict(name=architecture,path=str(out/'model.pt'),kind='research_nn',features='base',weight=1.)))
        print('DONE',architecture,flush=True)

if __name__=='__main__':main()
