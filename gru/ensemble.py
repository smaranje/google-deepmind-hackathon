import sys, json, os, shutil, numpy as np
sys.path.insert(0, 'gru'); sys.path.insert(0, 'research')
from train_gru import train, predict as tpredict
from common import load
s, fold = sys.argv[1], int(sys.argv[2])
CFG = [(32, k, 1500) for k in range(4)] + [(48, k, 2000) for k in range(3)] + [(64, k, 3000) for k in range(3)]
sig = np.array(json.load(open('research/sigma_cal.json'))[s])
old = load(s); sc = [json.load(open(f'research/{s}_sc{i}.json')) for i in (1, 2)]
brief = old[0]['brief']; names = brief['observables']; bounds = brief['interventions']; ctl = list(bounds)
if fold >= 0:
    typ, pool, i = [('random', old, 1), ('random', old, 2), ('scenario', sc, 0), ('scenario', sc, 1)][fold]
    r = pool[i]['runs'][0]
    tr = [d['runs'][0] for j, d in enumerate(old) if not (pool is old and j == i)] + [d['runs'][0] for j, d in enumerate(sc) if not (pool is sc and j == i)]
    Y = np.array([[o[k] for k in names] for o in r['observations']])
    preds = {}
    for H, k, ep in CFG:
        preds[f'H{H}_s{k}'] = tpredict([train(tr, names, ctl, bounds, sig, H=H, seed=k, epochs=ep)], r, names, ctl, bounds)
    sc_ = lambda P: float(np.mean(1 / (1 + np.abs(P - Y) / sig)))
    out = {k: sc_(v) for k, v in preds.items()}
    out['ENS_H32x4'] = sc_(np.mean([preds[f'H32_s{k}'] for k in range(4)], 0))
    out['ENS_all10'] = sc_(np.mean(list(preds.values()), 0))
    print(json.dumps({'fold': fold, 'type': typ, **out}))
else:
    runs = [d['runs'][0] for d in old + sc]
    Yall = np.array([[o[k] for k in names] for r in runs for o in r['observations']]); span = Yall.max(0) - Yall.min(0)
    T = lambda x: x.detach().numpy().tolist(); nets = []
    for H, k, ep in CFG:
        n, mu, sd = train(runs, names, ctl, bounds, sig, H=H, seed=k, epochs=ep)
        nets.append({'mu': mu.tolist(), 'sd': sd.tolist(), 'w_h0': T(n.h0.weight), 'b_h0': T(n.h0.bias), 'w_ih': T(n.gru.weight_ih_l0), 'b_ih': T(n.gru.bias_ih_l0),
                     'w_hh': T(n.gru.weight_hh_l0), 'b_hh': T(n.gru.bias_hh_l0), 'w_out': T(n.out.weight), 'b_out': T(n.out.bias)})
    m = {'family': s, 'observables': names, 'controls': ctl, 'bounds': bounds, 'lo': (Yall.min(0) - 0.25 * span).tolist(),
         'hi': (Yall.max(0) + 0.25 * span).tolist(), 'fallback': Yall.mean(0).tolist(), 'nets': nets}
    os.makedirs(f'models_ens/{s}', exist_ok=True); json.dump(m, open(f'models_ens/{s}/model.json', 'w')); shutil.copy('gru/predict.py', f'models_ens/{s}/predict.py')
    print('final ensemble written')
