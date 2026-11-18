import numpy as np,sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'run3.py')).read().split('rng=np.random.default_rng(7)')[0])
# functional form: true T link = cloglog, O link = probit; fit logit with u,w free in T and O
from scipy.stats import norm
rng=np.random.default_rng(5)
X,V,_=simulate(10000,rng)
th={s:np.zeros(5) for s in STAGES}
for (s,f),b in TRUE.items(): th[s][FEATS.index(f)]=b
q={s:expit(X@th[s]) for s in STAGES}
q['T']=1-np.exp(-np.exp(X@th['T'])); q['O']=norm.cdf(X@th['O']*0.6)
base=q['T']*q['O']; r={'R':base*q['P']*q['eR'],'L':base*q['P']*q['eL'],'C':base*(q['P']*q['eCp']+(1-q['P'])*q['eCm'])}
cnt={k:rng.binomial(V,r[k]) for k in r}
sp=Spec(m(T=['w'],O=['u'],P=['u','w'],eR=['u','w'],eL=['u','w']))
p,res=fit(sp,X,V,cnt); print('S6 wrong link (T cloglog, O probit), u,w free everywhere:',show(sp,p,['u','w']),'conv',res.success)
# R only, full exclusions (only stages that matter)
X,V,cnt=simulate(10000,np.random.default_rng(3))
spR=Spec({'T':['1','u'],'O':['1','w'],'P':['1','v'],'eR':['1','m']},('R',))
I=spR.fisher(spR.pack(),X,V,0.0); ev=np.linalg.eigvalsh(I); print('S3a R only, exclusions: cond',f'{ev[-1]/ev[0]:.2e}', 'SE',np.round(np.sqrt(np.diag(np.linalg.inv(I))),3))
spR2=Spec({'T':['1','u','v'],'O':['1','w'],'P':['1','v'],'eR':['1','m','v']},('R',))
I=spR2.fisher(spR2.pack(),X,V,0.0); ev,evec=np.linalg.eigh(I); print('S3b R only, v free in T,P,eR: cond',f'{ev[-1]/ev[0]:.2e}','SE',dict(zip([f"{a}.{b}" for a,b in spR2.idx],np.round(np.sqrt(np.diag(np.linalg.inv(I))),3))))
p,_=fit(spR2,X,V,cnt); print('   fit:',show(spR2,p,['v']), ' sum=',round(sum(v for (s,f),v in zip(spR2.idx,p) if f=='v'),3))
