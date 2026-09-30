"""GRU whose inputs include the fitted hospital physics model's forecasts (physics-informed GRU)."""
import sys, json, numpy as np, torch
sys.path.insert(0, 'diffsim'); sys.path.insert(0, 'gru'); sys.path.insert(0, 'research')
import hospital as H, train_gru as T
from common import load
fold = int(sys.argv[1]); s = 'hospital_queue'; sig = np.array(json.load(open('research/sigma_cal.json'))[s])
old = load(s); sc = [json.load(open(f'research/{s}_sc{i}.json')) for i in (1, 2)]; names = old[0]['brief']['observables']
typ, pool, i = [('random', old, 1), ('random', old, 2), ('scenario', sc, 0), ('scenario', sc, 1)][fold]
r = pool[i]['runs'][0]; tr = [d['runs'][0] for j, d in enumerate(old) if not (pool is old and j == i)] + [d['runs'][0] for j, d in enumerate(sc) if not (pool is sc and j == i)]
phys = H.fit(tr, names, sig, (1., 1., 0.))
def simulate(run):
    X0, U, _, _ = H.batch([run], names)
    with torch.no_grad(): return phys(X0, U)[0].numpy()[:len(run['actions'])]
PS = {id(x): simulate(x) for x in tr + [r]}
allP = np.vstack(list(PS.values())); pm, ps = allP.mean(0), allP.std(0) + 1e-6
_orig = T.prep
def prep2(runs, n_, ctl, bounds, mu, sd):
    X0, U, Y, M = _orig(runs, n_, ctl, bounds, mu, sd); U2 = torch.zeros(U.shape[0], U.shape[1], U.shape[2] + 3)
    U2[:, :, :U.shape[2]] = U
    for k, x in enumerate(runs):
        p = (PS[id(x)] - pm) / ps; U2[k, :len(p), U.shape[2]:] = torch.tensor(p, dtype=torch.float32)
    return X0, U2, Y, M
T.prep = prep2; _N = T.Net; T.Net = lambda k, o, Hd: _N(k + 3, o, Hd)
bounds = old[0]['brief']['interventions']; ctl = list(bounds)
nets = [T.train(tr, names, ctl, bounds, sig, H=64, seed=k) for k in range(2)]
P = T.predict(nets, r, names, ctl, bounds); Y = np.array([[o[k] for k in names] for o in r['observations']])
S = 1 / (1 + np.abs(P - Y) / sig)
print(json.dumps({'fold': fold, 'type': typ, 'score': round(float(S.mean()), 4), 'phys_only': round(float((1 / (1 + np.abs(PS[id(r)] - Y) / sig)).mean()), 4)}))
