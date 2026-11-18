"""S1 and S2d at the real overdispersion level (Sec. V-E of the paper).

Reconstruction: the original script of this calculation was not preserved; this file rebuilds it from its
description. Overdispersion of each counter is set by an intraclass correlation icc_o = alpha_o * median(rho_o),
where alpha_o is the NB2 dispersion of the main model (step 3.5 of the main analysis, text/image/video).
Output: mean, min and max over 20 simulation seeds of the SEs of the feature effects (S1: motive, attention,
consumption, evaluation) and of the cross-stage terms that separate attention from consumption (S2d).

alpha is read from ANALYSIS_OUT/step3_5_negbin_alpha.csv (model "main: text/image/video") if available,
otherwise the values of the reported run are used (R 1.609, L 1.398, C 3.077).
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import *

ALPHA = {'R': 1.609, 'L': 1.398, 'C': 3.077}
f = os.path.join(os.environ.get('ANALYSIS_OUT', ''), 'step3_5_negbin_alpha.csv')
if os.path.exists(f):
    import pandas as pd
    a = pd.read_csv(f); a = a[a.model == 'main: text/image/video'].set_index('counter').alpha
    ALPHA = {'R': float(a['reposts']), 'L': float(a['likes']), 'C': float(a['comments'])}
print('alpha NB2:', {k: round(v, 3) for k, v in ALPHA.items()})


def fisher_alpha(spec, p, X, V, alpha, h=1e-5):
    """Fisher information as in Spec.fisher, but with a counter-specific icc_o = alpha_o * median(rho_o)."""
    r0 = rhos(X, spec.unpack(p)); J = {o: [] for o in spec.outcomes}
    for j in range(len(p)):
        pp = p.copy(); pp[j] += h; pm = p.copy(); pm[j] -= h
        rp = rhos(X, spec.unpack(pp)); rm = rhos(X, spec.unpack(pm))
        for o in spec.outcomes: J[o].append((rp[o] - rm[o]) / (2 * h))
    I = np.zeros((len(p), len(p)))
    for o in spec.outcomes:
        Jo = np.array(J[o]).T; r = r0[o]; icc = alpha[o] * np.median(r)
        w = V / (r * (1 - r) * (1 + (V - 1) * icc))
        I += (Jo * w[:, None]).T @ Jo
    return I


S1 = Spec(FULL)
ALLF = ['u', 'w', 'v', 'm']
S2d = Spec({s: (FULL[s] + [f for f in ALLF if f not in FULL[s]]) if s in ('T', 'O', 'P', 'eR', 'eL') else FULL[s]
            for s in FULL})   # all features allowed in T, O, P, eR, eL
SHOW1 = [('eR', 'm', 'motive'), ('T', 'u', 'attention'), ('O', 'w', 'consumption'), ('P', 'v', 'evaluation')]
res1, res2 = [], []
for k in range(20):
    X, V, _ = simulate(10000, np.random.default_rng(1000 + k))
    se = np.sqrt(np.diag(np.linalg.inv(fisher_alpha(S1, S1.pack(), X, V, ALPHA))))
    res1.append([se[S1.idx.index((s, f))] for s, f, _ in SHOW1])
    se2 = np.sqrt(np.diag(np.linalg.inv(fisher_alpha(S2d, S2d.pack(), X, V, ALPHA))))
    res2.append([se2[S2d.idx.index(('T', 'w'))], se2[S2d.idx.index(('O', 'u'))]])
res1, res2 = np.array(res1), np.array(res2)
print('S1, real overdispersion, N=10000, 20 seeds: SE of feature effects (mean [min, max])')
for j, (s, f, nm) in enumerate(SHOW1):
    print(f'  {nm:12s} {s}.{f}: {res1[:, j].mean():.3f} [{res1[:, j].min():.3f}, {res1[:, j].max():.3f}]')
print('S2d (all features free in T, O, P, eR, eL): SE of cross-stage terms T.w and O.u (attention vs consumption):')
for j, nm in enumerate(['T.w', 'O.u']):
    print(f'  {nm}: {res2[:, j].mean():.3f} [{res2[:, j].min():.3f}, {res2[:, j].max():.3f}]')
