import sys, json, numpy as np, importlib.util as u
sys.path.insert(0, 'research'); sys.path.insert(0, 'gru')
from common import load, starter, direct
import train_gru as T
sp = u.spec_from_file_location('f2', 'direct2/fit_direct2.py'); f2 = u.module_from_spec(sp); sp.loader.exec_module(f2)
sp = u.spec_from_file_location('d2p', 'direct2/predict.py'); D2 = u.module_from_spec(sp); sp.loader.exec_module(D2)
X = dict(cross=(4, 32)); R = dict(cross=(4, 32), rff=300, rff_scale=0.3)
REG = {'power_grid': ('d1', 3, 1), 'epidemic': ('starter', 0, 1), 'market': ('d2', dict(lam=30, **R), 1), 'traffic': ('starter', 0, 1),
       'wildlife': ('d2', dict(lam=10, **X), 3), 'reservoir': ('d2', dict(lam=30, **R), 3), 'ad_auction': ('d2', dict(lam=10, **R), 3),
       'social_contagion': ('d1', 3, 1), 'hospital_queue': ('starter', 0, 1)}
s = sys.argv[1]; kind, arg, up = REG[s]
sig = np.array(json.load(open('research/sigma_cal.json'))[s])
old = load(s); sc = [json.load(open(f'research/{s}_sc{i}.json')) for i in (1, 2)]
b = old[0]['brief']; names = b['observables']; bounds = b['interventions']; ctl = list(bounds)
W = [0, 0.25, 0.5, 0.75, 1.0]; E = {w: {'random': [], 'scenario': []} for w in W}
for typ, pool, i in [('random', old, 1), ('random', old, 2), ('scenario', sc, 0), ('scenario', sc, 1)]:
    r = pool[i]['runs'][0]; Y = np.array([[o[k] for k in names] for o in r['observations']])
    oo = [d for j, d in enumerate(old) if not (pool is old and j == i)]; os_ = [d for j, d in enumerate(sc) if not (pool is sc and j == i)]
    G = T.predict([T.train([d['runs'][0] for d in oo + os_], names, ctl, bounds, sig, H=32, seed=k) for k in range(2)], r, names, ctl, bounds)
    if kind == 'starter': Rp = starter(oo + os_ * up, s)(r)
    elif kind == 'd1': Rp = direct(oo + os_ * up, arg)(r)
    else:
        m = f2.fit(oo + os_ * up, **arg); D2._MODEL = m; Rp = D2.predict(r['initial'], r['actions'], {})
    Rp = np.array([[p[k] for k in names] for p in Rp])
    for w in W: E[w][typ].append(np.abs(w * G + (1 - w) * Rp - Y))
print(json.dumps({'s': s, **{f'gru{w}': [round(float(np.mean(1 / (1 + np.vstack(E[w][t]) / sig))), 4) for t in ('random', 'scenario')] for w in W}}))
