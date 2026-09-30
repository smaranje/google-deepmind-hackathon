import sys, json, numpy as np, torch
torch.set_num_threads(1); sys.path.insert(0, 'gru'); sys.path.insert(0, 'research')
import train_gru as T; from common import load
s = sys.argv[1]; PER = [float(x) for x in sys.argv[2].split(',')]; fold = int(sys.argv[3])
sig = np.array(json.load(open('research/sigma_cal.json'))[s]); old = load(s); sc = [json.load(open(f'research/{s}_sc{i}.json')) for i in (1, 2)]
b = old[0]['brief']; names = b['observables']; bounds = b['interventions']; ctl = list(bounds)
typ, pool, i = [('random', old, 1), ('random', old, 2), ('scenario', sc, 0), ('scenario', sc, 1)][fold]
r = pool[i]['runs'][0]; Y = np.array([[o[k] for k in names] for o in r['observations']])
tr = [d['runs'][0] for j, d in enumerate(old) if not (pool is old and j == i)] + [d['runs'][0] for j, d in enumerate(sc) if not (pool is sc and j == i)]
Pb = T.predict([T.train(tr, names, ctl, bounds, sig, H=32, seed=k) for k in range(2)], r, names, ctl, bounds)
_orig = T.prep
def prep_clock(runs, n_, c_, b_, mu, sd):
    X0, U, Yt, M = _orig(runs, n_, c_, b_, mu, sd); t = torch.arange(1, U.shape[1] + 1, dtype=torch.float32)[None, :, None]
    feats = [f(2 * np.pi * t / P) for P in PER for f in (torch.sin, torch.cos)]
    return X0, torch.cat([U] + [x.expand(U.shape[0], -1, -1) for x in feats], -1), Yt, M
T.prep = prep_clock; _N = T.Net; T.Net = lambda k, o, H: _N(k + 2 * len(PER), o, H)
Pc = T.predict([T.train(tr, names, ctl, bounds, sig, H=32, seed=k) for k in range(2)], r, names, ctl, bounds)
f = lambda P, a=0: float(np.mean(1 / (1 + np.abs(P[a:] - Y[a:]) / sig)))
print(json.dumps({'s': s, 'fold': fold, 'type': typ, 'base': [f(Pb), f(Pb, 50)], 'clock': [f(Pc), f(Pc, 50)]}))
