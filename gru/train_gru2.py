"""GRU variant whose inputs include EMAs of the actions (half-lives 16/128/1024)."""
import sys, json, time, os, shutil, numpy as np, torch
sys.path.insert(0, 'gru'); sys.path.insert(0, 'research')
import train_gru as T
from common import load
HL = [16, 128, 1024]
_orig = T.prep
def ema_aug(U):
    cols = [U]
    for h in HL:
        a = 1 - 0.5 ** (1 / h); e = np.zeros(U.shape[-1], np.float32); out = np.empty_like(U)
        for t in range(U.shape[0]): e = e + a * (U[t] - e); out[t] = e
        cols.append(out)
    return np.concatenate(cols, -1)
def prep2(runs, names, ctl, bounds, mu, sd):
    X0, U, Y, M = _orig(runs, names, ctl, bounds, mu, sd)
    Un = U.numpy(); Ua = np.stack([ema_aug(Un[i]) for i in range(len(Un))])
    return X0, torch.tensor(Ua), Y, M
T.prep = prep2
class Net2(T.Net): pass
_origNet = T.Net
def train2(runs, names, ctl, bounds, sig, **kw):
    T.Net = lambda k, o, H: _origNet(k * (1 + len(HL)), o, H)
    try: return T.train(runs, names, ctl, bounds, sig, **kw)
    finally: T.Net = _origNet
s, mode = sys.argv[1], sys.argv[2]
sig = np.array(json.load(open('research/sigma_cal.json'))[s])
old = load(s); sc = [json.load(open(f'research/{s}_sc{i}.json')) for i in (1, 2)]
brief = old[0]['brief']; names = brief['observables']; bounds = brief['interventions']; ctl = list(bounds)
if mode == 'cv':
    E = {'random': [], 'scenario': []}
    for typ, pool, i in [('random', old, 1), ('random', old, 2), ('scenario', sc, 0), ('scenario', sc, 1)]:
        r = pool[i]['runs'][0]
        tr = [d['runs'][0] for j, d in enumerate(old) if not (pool is old and j == i)] + [d['runs'][0] for j, d in enumerate(sc) if not (pool is sc and j == i)]
        nets = [train2(tr, names, ctl, bounds, sig, H=32, seed=k) for k in range(2)]
        P = T.predict(nets, r, names, ctl, bounds); Y = np.array([[o[k] for k in names] for o in r['observations']])
        E[typ].append(np.abs(P - Y))
    print(json.dumps({'s': s, **{t: round(float(np.mean(1 / (1 + np.vstack(v) / sig))), 4) for t, v in E.items()}}))
else:
    runs = [d['runs'][0] for d in old + sc]
    Yall = np.array([[o[k] for k in names] for r in runs for o in r['observations']]); span = Yall.max(0) - Yall.min(0)
    L = lambda x: x.detach().numpy().tolist(); nets = []
    for k in range(3):
        n, mu, sd = train2(runs, names, ctl, bounds, sig, H=32, seed=k)
        nets.append({'mu': mu.tolist(), 'sd': sd.tolist(), 'w_h0': L(n.h0.weight), 'b_h0': L(n.h0.bias), 'w_ih': L(n.gru.weight_ih_l0), 'b_ih': L(n.gru.bias_ih_l0),
                     'w_hh': L(n.gru.weight_hh_l0), 'b_hh': L(n.gru.bias_hh_l0), 'w_out': L(n.out.weight), 'b_out': L(n.out.bias)})
    m = {'family': s, 'observables': names, 'controls': ctl, 'bounds': bounds, 'ema_in': HL, 'lo': (Yall.min(0) - 0.25 * span).tolist(),
         'hi': (Yall.max(0) + 0.25 * span).tolist(), 'fallback': Yall.mean(0).tolist(), 'nets': nets}
    os.makedirs(f'models_gru_ema/{s}', exist_ok=True); json.dump(m, open(f'models_gru_ema/{s}/model.json', 'w')); shutil.copy('gru/predict_ema.py', f'models_gru_ema/{s}/predict.py')
    print('written')
