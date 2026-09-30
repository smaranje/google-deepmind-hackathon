import sys, json, numpy as np
sys.path.insert(0, 'research'); sys.path.insert(0, 'gru')
from common import load
import train_gru as T
s = sys.argv[1]; sig = np.array(json.load(open('research/sigma_cal.json'))[s])
old = load(s); sc = [json.load(open(f'research/{s}_sc{i}.json')) for i in (1, 2)]
b = old[0]['brief']; names = b['observables']; bounds = b['interventions']; ctl = list(bounds)
B = [(0, 20), (20, 60), (60, 150), (150, 300), (300, 460)]; acc = {x: [] for x in B}; per_obs = []
for pool, i in [(old, 1), (old, 2), (sc, 0), (sc, 1)]:
    r = pool[i]['runs'][0]; Y = np.array([[o[k] for k in names] for o in r['observations']])
    tr = [d['runs'][0] for j, d in enumerate(old) if not (pool is old and j == i)] + [d['runs'][0] for j, d in enumerate(sc) if not (pool is sc and j == i)]
    P = T.predict([T.train(tr, names, ctl, bounds, sig, H=32, seed=0)], r, names, ctl, bounds)
    S = 1 / (1 + np.abs(P - Y) / sig); per_obs.append(S.mean(0))
    for lo, hi in B:
        if len(S) > lo: acc[(lo, hi)].append(S[lo:hi].mean())
print(json.dumps({'s': s, 'by_time': {f'{lo}-{hi}': round(float(np.mean(v)), 3) for (lo, hi), v in acc.items() if v},
                  'by_obs': dict(zip(names, np.round(np.mean(per_obs, 0), 3).tolist()))}))
