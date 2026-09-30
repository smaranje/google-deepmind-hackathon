import sys, json, numpy as np, torch
sys.path.insert(0, 'gru'); sys.path.insert(0, 'research')
from train_gru import train, predict as tpredict
from common import load
s = sys.argv[1]; seeds = int(sys.argv[2]); HH = int(sys.argv[3]) if len(sys.argv) > 3 else 32; OUT = sys.argv[4] if len(sys.argv) > 4 else "models_gru"; EPS = int(sys.argv[5]) if len(sys.argv) > 5 else 1500
sig = np.array(json.load(open('research/sigma_cal.json'))[s])
datas = load(s) + [json.load(open(f'research/{s}_sc{i}.json')) for i in (1, 2)]
brief = datas[0]['brief']; names = brief['observables']; bounds = brief['interventions']; ctl = list(bounds)
runs = [d['runs'][0] for d in datas]
Y = np.array([[o[k] for k in names] for r in runs for o in r['observations']]); span = Y.max(0) - Y.min(0)
nets = [train(runs, names, ctl, bounds, sig, H=HH, seed=k, epochs=EPS) for k in range(seeds)]
T = lambda x: x.detach().numpy().tolist()
m = {'family': s, 'observables': names, 'controls': ctl, 'bounds': bounds,
     'lo': (Y.min(0) - 0.25 * span).tolist(), 'hi': (Y.max(0) + 0.25 * span).tolist(), 'fallback': Y.mean(0).tolist(),
     'nets': [{'mu': mu.tolist(), 'sd': sd.tolist(), 'w_h0': T(n.h0.weight), 'b_h0': T(n.h0.bias),
               'w_ih': T(n.gru.weight_ih_l0), 'b_ih': T(n.gru.bias_ih_l0), 'w_hh': T(n.gru.weight_hh_l0), 'b_hh': T(n.gru.bias_hh_l0),
               'w_out': T(n.out.weight), 'b_out': T(n.out.bias)} for n, mu, sd in nets]}
import os; os.makedirs(f'{OUT}/{s}', exist_ok=True)
json.dump(m, open(f'{OUT}/{s}/model.json', 'w'))
import shutil; shutil.copy('gru/predict.py', f'{OUT}/{s}/predict.py')
# check numpy export matches torch
import importlib.util as u
sp_ = u.spec_from_file_location('gp' + s, f'{OUT}/{s}/predict.py'); G = u.module_from_spec(sp_); sp_.loader.exec_module(G)
r = runs[-1]; Pn = np.array([[p[k] for k in names] for p in G.predict(r['initial'], r['actions'], {})])
Pt = np.clip(tpredict(nets, r, names, ctl, bounds), m['lo'], m['hi'])
print(s, 'max |numpy - torch| =', float(np.abs(Pn - Pt).max()))
