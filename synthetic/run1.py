import numpy as np,sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from model import *
rng=np.random.default_rng(20261113)
X,V,cnt=simulate(10000,rng)
def report(name,spec,icc_list=(0,0.002,0.01)):
    p=spec.pack(); print(f'\n=== {name} (k={len(p)}) ===')
    for icc in icc_list:
        I=spec.fisher(p,X,V,icc); ev,evec=np.linalg.eigh(I)
        try: se=np.sqrt(np.diag(np.linalg.inv(I)))
        except Exception: se=np.full(len(p),np.inf)
        print(f' icc={icc}: cond={ev[-1]/max(ev[0],1e-300):.2e}; min eig={ev[0]:.3g}')
        print('   '+'  '.join(f'{s}.{f}={v:+.2f}±{e:.3f}' for (s,f),v,e in zip(spec.idx,p,se)))
        if icc==0:
            v=evec[:,0]; big=np.argsort(-abs(v))[:4]
            print('   weakest direction:',', '.join(f'{spec.idx[i][0]}.{spec.idx[i][1]}:{v[i]:+.2f}' for i in big))
report('S1 full model, R+L+C',Spec(FULL))
