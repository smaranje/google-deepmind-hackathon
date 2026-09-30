import sys, json, numpy as np, torch
sys.path.insert(0, 'diffsim'); sys.path.insert(0, 'research')
from social import fit, batch; from common import load
seed = int(sys.argv[1]); s = 'social_contagion'; sig = np.array(json.load(open('research/sigma_cal.json'))[s])
ds = load(s) + [json.load(open(f'research/{s}_sc{i}.json')) for i in (1, 2)]; names = ds[0]['brief']['observables']
m = fit([d['runs'][0] for d in ds], names, sig, (1., 1., 0.), False, epochs=700, seed=seed)
json.dump({k: v.detach().numpy().tolist() for k, v in m.named_parameters()}, open(f'diffsim/params_{seed}.json', 'w'))
r = ds[-1]['runs'][0]; X0, U, Y, M = batch([r], names)
with torch.no_grad(): P = m(X0, U)[0].numpy()
np.save(f'diffsim/check_{seed}.npy', P); print('seed', seed, 'in-sample score', float(np.mean(1 / (1 + np.abs(P - Y[0].numpy()) / sig))))
