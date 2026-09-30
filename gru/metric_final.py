"""Final metric-aligned GRU models (trained on all 5 runs) -> NumPy predictor folder."""
import sys, json, os, shutil, numpy as np
sys.path.insert(0, 'gru'); sys.path.insert(0, 'research')
exec(open('gru/metric_cv.py').read().split("if __name__")[0])
s, OUT = sys.argv[1], sys.argv[2]; CFG = [(int(h), int(k), int(e)) for h, k, e in (x.split(':') for x in sys.argv[3].split(','))]
sig = np.array(json.load(open('research/sigma_cal.json'))[s])
if s == 'power_grid': ds = [json.load(open(p)) for p in ['power-grid-research.json', 'research/power_grid_holdout.json', 'research/power_grid_s2.json']]
else: ds = load(s)
ds += [json.load(open(f'research/{s}_sc{i}.json')) for i in (1, 2)]
b = ds[0]['brief']; names = b['observables']; bounds = b['interventions']; ctl = list(bounds); runs = [d['runs'][0] for d in ds]
Yall = np.array([[o[k] for k in names] for r in runs for o in r['observations']]); span = Yall.max(0) - Yall.min(0)
L = lambda x: x.detach().numpy().tolist(); nets = []
for H, k, ep in CFG:
    n, mu, sd = train_metric(runs, names, ctl, bounds, sig, H=H, seed=k, epochs=ep)
    nets.append({'mu': mu.tolist(), 'sd': sd.tolist(), 'w_h0': L(n.h0.weight), 'b_h0': L(n.h0.bias), 'w_ih': L(n.gru.weight_ih_l0), 'b_ih': L(n.gru.bias_ih_l0),
                 'w_hh': L(n.gru.weight_hh_l0), 'b_hh': L(n.gru.bias_hh_l0), 'w_out': L(n.out.weight), 'b_out': L(n.out.bias)})
m = {'family': s, 'observables': names, 'controls': ctl, 'bounds': bounds, 'lo': (Yall.min(0) - 0.25 * span).tolist(), 'hi': (Yall.max(0) + 0.25 * span).tolist(), 'fallback': Yall.mean(0).tolist(), 'nets': nets}
os.makedirs(f'{OUT}/{s}', exist_ok=True); json.dump(m, open(f'{OUT}/{s}/model.json', 'w')); shutil.copy('gru/predict.py', f'{OUT}/{s}/predict.py'); print(s, 'written', len(nets), 'nets')
