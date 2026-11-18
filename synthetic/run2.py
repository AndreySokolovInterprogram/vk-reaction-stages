import numpy as np,sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from model import *
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'run1.py')).read().split("report('S1")[0])
import copy
def m(**add):
    d=copy.deepcopy(FULL)
    for s,fs in add.items():
        for f in fs:
            if f not in d[s]: d[s].append(f)
    return d
report('S2a: u allowed in T and O',Spec(m(O=['u'])),(0,0.01))
report('S2b: v allowed in P, eR, eL',Spec(m(eR=['v'],eL=['v'])),(0,0.01))
report('S2c: m allowed in P, eR, eL',Spec(m(P=['m'],eL=['m'])),(0,0.01))
allf=['u','w','v','m']
report('S2d: all features in T,O,P,eR,eL',Spec(m(T=allf,O=allf,P=allf,eR=allf,eL=allf)),(0,0.01))
report('S2e: all features in T,P,eR,eL (T and O merged)',Spec(m(T=allf+['w'],P=allf,eR=allf,eL=allf)|{'O':[]}),(0,0.01))
report('S3a: R only, full exclusions',Spec(FULL,('R',)),(0,0.01))
report('S3b: R only, v allowed in P and eR',Spec(m(eR=['v']),('R',)),(0,0.01))
report('S3c: R+L, v allowed in P,eR,eL',Spec(m(eR=['v'],eL=['v']),('R','L')),(0,0.01))
