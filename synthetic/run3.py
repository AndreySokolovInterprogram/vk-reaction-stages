import numpy as np,sys,os,copy,time; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from model import *
from scipy.optimize import minimize
def m(base=FULL,drop=(),**add):
    d=copy.deepcopy(base)
    for s,fs in add.items():
        for f in fs:
            if f not in d[s]: d[s].append(f)
    for s in drop: d[s]=[]
    return d
def fit(spec,X,V,cnt,p0=None):
    p0=spec.pack() if p0 is None else p0
    r=minimize(spec.nll,p0+0.05,args=(X,V,cnt),method='L-BFGS-B',options={'maxiter':3000,'maxfun':200000})
    return r.x,r
def simulate_het(N,rng,sd_eps=1.0,where='T'):
    X,V,_=simulate(N,rng)
    th={s:np.zeros(5) for s in STAGES}
    for (s,f),b in TRUE.items(): th[s][FEATS.index(f)]=b
    q={s:expit(X@th[s]) for s in STAGES}
    eps=sd_eps*rng.standard_normal(N)
    q[where]=expit(X@th[where]+eps)
    base=q['T']*q['O']
    r={'R':base*q['P']*q['eR'],'L':base*q['P']*q['eL'],'C':base*(q['P']*q['eCp']+(1-q['P'])*q['eCm'])}
    return X,V,{k:rng.binomial(V,r[k]) for k in r}
def show(spec,p,keys=None):
    return '  '.join(f'{s}.{f}={v:+.3f}' for (s,f),v in zip(spec.idx,p) if keys is None or f in keys)
rng=np.random.default_rng(7)
# S3 fixed: R only
X,V,cnt=simulate(10000,rng)
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'run1.py')).read().split('rng=')[0])
allf=['u','w','v','m']
t=time.time()
# S4: misspecification eCm -> fit without negative-comment path
specS4=Spec(m(drop=['eCm'],eR=['v'],eL=['v']))
p,_=fit(specS4,X,V,cnt); print('S4 fit w/o eCm, v free in P,eR,eL:',show(specS4,p,['v','1']),f'({time.time()-t:.0f}s)')
spec4b=Spec(m(eR=['v'],eL=['v'])); p,_=fit(spec4b,X,V,cnt); print('S4 ref with eCm:',show(spec4b,p,['v','1']))
# S5: post-level heterogeneity in T (sd 1), fit free attribution of u and full of others
for sd in (0.5,1.0):
    Xh,Vh,ch=simulate_het(10000,np.random.default_rng(11),sd)
    sp=Spec(m(O=['u'],P=['u'],eR=['u'],eL=['u']))
    p,r=fit(sp,Xh,Vh,ch); print(f'S5 het sd={sd}, u free in T,O,P,eR,eL:',show(sp,p,['u']),'| conv',r.success)
    sp2=Spec(m(base=FULL,drop=['O'],T=['w'],P=['u'],eR=['u'],eL=['u']))
    p,r=fit(sp2,Xh,Vh,ch); print(f'S5 het sd={sd}, T+O merged, u free in TO,P,eR,eL:',show(sp2,p,['u','w']))
