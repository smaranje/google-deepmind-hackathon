import sys, json, numpy as np, torch
sys.path.insert(0, 'diffsim'); sys.path.insert(0, 'gru'); sys.path.insert(0, 'research')
import social as S, train_gru as T; from common import load
fold = int(sys.argv[1]); s = 'social_contagion'; sig = np.array(json.load(open('research/sigma_cal.json'))[s])
old = load(s); sc = [json.load(open(f'research/{s}_sc{i}.json')) for i in (1, 2)]; b = old[0]['brief']; names = b['observables']; bounds = b['interventions']; ctl = list(bounds)
typ, pool, i = [('random', old, 1), ('random', old, 2), ('scenario', sc, 0), ('scenario', sc, 1)][fold]
r = pool[i]['runs'][0]; tr = [d['runs'][0] for j, d in enumerate(old) if not (pool is old and j == i)] + [d['runs'][0] for j, d in enumerate(sc) if not (pool is sc and j == i)]
m = S.fit(tr, names, sig, (1., 1., 0.), False, epochs=500); X0, U, _, _ = S.batch([r], names)
with torch.no_grad(): Pp = m(X0, U)[0].numpy()
Pg = T.predict([T.train(tr, names, ctl, bounds, sig, H=32, seed=k) for k in range(3)], r, names, ctl, bounds)
Y = np.array([[o[k] for k in names] for o in r['observations']])
print(json.dumps({'fold': fold, 'type': typ, **{f'w{w}': round(float(np.mean(1 / (1 + np.abs(w * Pp + (1 - w) * Pg - Y) / sig))), 4) for w in (0, 0.25, 0.5, 0.6, 0.75, 0.9, 1.0)}}))
