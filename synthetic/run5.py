import numpy as np,sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'run3.py')).read().split('rng=np.random.default_rng(7)')[0])
X,V,cnt=simulate(10000,np.random.default_rng(11))
sp2=Spec(m(base=FULL,drop=['O'],T=['w'],P=['u'],eR=['u'],eL=['u']))
p,r=fit(sp2,X,V,cnt); print('merged T+O, no heterogeneity:',show(sp2,p,['u','w']))
# replicate check S1 with overdispersion via shared post-level eps sd=1 in T, fit full exclusions spec
est=[]
for k in range(10):
    Xh,Vh,ch=simulate_het(10000,np.random.default_rng(100+k),1.0)
    sp=Spec(FULL); p,_=fit(sp,Xh,Vh,ch); est.append(p)
est=np.array(est); sp=Spec(FULL)
print('S1 under heterogeneity sd=1 (10 reps): mean (true) sd')
for (s,f),mu,sd,tv in zip(sp.idx,est.mean(0),est.std(0),sp.pack()): print(f'  {s}.{f}: {mu:+.3f} ({tv:+.2f}) sd {sd:.3f}')
