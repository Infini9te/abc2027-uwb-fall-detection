import numpy as np
import uwb_detector as b

def pair_features(p,v):
    i,q=p[:,:,:184],p[:,:,184:]
    energy=i*i+q*q+v[:,:,:184]+v[:,:,184:]
    def bands(e):return e.reshape(len(e),3,8,23).mean(-1)
    def describe(e):
        mass=e.sum(-1)+1e-20
        center=(e*np.linspace(0,1,184,dtype=np.float32)).sum(-1)/mass
        spread=(e*(np.linspace(0,1,184,dtype=np.float32)[None,None,:]-center[:,:,None])**2).sum(-1)/mass
        return np.concatenate([np.log1p(bands(e)*1e10).reshape(len(e),-1),np.log1p(e.mean(-1)*1e10),np.log1p(e.max(-1)*1e10),center,spread],axis=1)
    # Columns 0:36 are static amplitude/energy; 36:144 motion; 144:252 short smoothing.
    short=[describe(v[:,:,:184]+v[:,:,184:])]
    long=[]
    for n in (1,3,10,30):
        e=(i-b.lag(i,n))**2+(q-b.lag(q,n))**2
        (short if n<=3 else long).append(describe(e))
    s=np.concatenate(short,axis=1)
    return np.ascontiguousarray(np.concatenate([describe(energy),s,b.trailing(s,3),*long,b.trailing(s,10),b.trailing(s,30)],axis=1),dtype=np.float32)

def short_future(x):
    summary=x[:,36:144]
    indexes=np.arange(len(x))
    return np.ascontiguousarray(np.concatenate([x[:,36:252],summary[np.minimum(len(x)-1,indexes+3)],summary[np.minimum(len(x)-1,indexes+5)]],axis=1))
def extract(path,cache):
    oldfun,oldver=b.make_features,b.VERSION
    try:
        b.make_features=pair_features;b.VERSION='paired-iq-v11-1'
        return b.extract(path,cache)
    finally:b.make_features,b.VERSION=oldfun,oldver
