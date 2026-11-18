import numpy as np
from scipy.special import expit, log_expit
STAGES=['T','O','P','eR','eL','eCp','eCm']
FEATS=['1','u','w','v','m']
TRUE={('T','1'):-0.5,('T','u'):0.5,('O','1'):0.0,('O','w'):0.4,('P','1'):1.0,('P','v'):0.5,
      ('eR','1'):-4.0,('eR','m'):0.5,('eL','1'):-1.5,('eCp','1'):-4.0,('eCm','1'):-3.0}
def simulate(N,rng,corr=0.0,true=TRUE):
    C=np.full((4,4),corr); np.fill_diagonal(C,1)
    Z=rng.multivariate_normal(np.zeros(4),C,size=N)
    X=np.column_stack([np.ones(N),Z])
    V=np.maximum(1,np.round(np.exp(np.log(3000)+1.5*rng.standard_normal(N)))).astype(int)
    th={s:np.zeros(5) for s in STAGES}
    for (s,f),b in true.items(): th[s][FEATS.index(f)]=b
    rho=rhos(X,th)
    cnt={k:rng.binomial(V,rho[k]) for k in rho}
    return X,V,cnt
def rhos(X,th):
    q={s:expit(X@th[s]) for s in STAGES}
    base=q['T']*q['O']
    return {'R':base*q['P']*q['eR'],'L':base*q['P']*q['eL'],'C':base*(q['P']*q['eCp']+(1-q['P'])*q['eCm'])}
class Spec:
    def __init__(self,mask,outcomes=('R','L','C')):
        self.mask=mask; self.outcomes=outcomes  # mask: dict stage->list of feats
        self.idx=[(s,f) for s in STAGES for f in mask.get(s,[])]
    def unpack(self,p):
        th={s:np.zeros(5) for s in STAGES}
        for v,(s,f) in zip(p,self.idx): th[s][FEATS.index(f)]=v
        return th
    def pack(self,true=TRUE): return np.array([true.get(k,0.0) for k in self.idx])
    def nll(self,p,X,V,cnt):
        r=rhos(X,self.unpack(p)); ll=0
        for o in self.outcomes:
            rr=np.clip(r[o],1e-12,1-1e-12); ll+=np.sum(cnt[o]*np.log(rr)+(V-cnt[o])*np.log1p(-rr))
        return -ll
    def fisher(self,p,X,V,icc=0.0,h=1e-5):
        r0=rhos(X,self.unpack(p)); J={o:[] for o in self.outcomes}
        for j in range(len(p)):
            pp=p.copy(); pp[j]+=h; pm=p.copy(); pm[j]-=h
            rp=rhos(X,self.unpack(pp)); rm=rhos(X,self.unpack(pm))
            for o in self.outcomes: J[o].append((rp[o]-rm[o])/(2*h))
        I=np.zeros((len(p),len(p)))
        for o in self.outcomes:
            Jo=np.array(J[o]).T; w=V/(r0[o]*(1-r0[o])*(1+(V-1)*icc))
            I+=(Jo*w[:,None]).T@Jo
        return I
FULL={'T':['1','u'],'O':['1','w'],'P':['1','v'],'eR':['1','m'],'eL':['1'],'eCp':['1'],'eCm':['1']}
