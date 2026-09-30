"""Compare baseline GRU training vs metric-aligned training (L1 warm-up, then fine-tune on the
actual score 1/(1+|e|/sigma), with the first 50 ticks down-weighted) on identical folds."""
import sys, json, time, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1); sys.path.insert(0, 'gru'); sys.path.insert(0, 'research')
import train_gru as T
from common import load

def train_metric(runs, names, ctl, bounds, sig, H=32, epochs=1500, seed=0, ft_frac=0.4, early_w=0.25, lr=3e-3, wd=1e-4):
    Yall = np.array([[o[k] for k in names] for r in runs for o in r['observations']]); mu, sd = Yall.mean(0), Yall.std(0) + 1e-9
    X0, U, Y, M = T.prep(runs, names, ctl, bounds, mu, sd)
    w = torch.tensor(sd / sig, dtype=torch.float32)
    tw = torch.ones(1, U.shape[1], 1); tw[:, :50] = early_w
    torch.manual_seed(seed); net = T.Net(len(ctl), len(names), H); opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs); n_ft = int(epochs * ft_frac)
    for ep in range(epochs):
        opt.zero_grad(); P = net(X0, U); E = torch.abs(P - Y) * w  # error in sigma units
        if ep < epochs - n_ft: L = nn.functional.smooth_l1_loss(P * w, Y * w, reduction='none', beta=0.05)
        else: L = 1 - 1 / (1 + E)
        loss = (L * M * tw).sum() / ((M * tw).sum() * len(names))
        loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
    return net, mu, sd

if __name__ == '__main__':
    s, H = sys.argv[1], int(sys.argv[2]); sig = np.array(json.load(open('research/sigma_cal.json'))[s])
    if s == 'power_grid':
        import json as J; old = [J.load(open(p)) for p in ['power-grid-research.json', 'research/power_grid_holdout.json', 'research/power_grid_s2.json']]
    else: old = load(s)
    sc = [json.load(open(f'research/{s}_sc{i}.json')) for i in (1, 2)]
    b = old[0]['brief']; names = b['observables']; bounds = b['interventions']; ctl = list(bounds)
    out = {'base': {'random': [], 'scenario': []}, 'metric': {'random': [], 'scenario': []}}; late = {'base': {'random': [], 'scenario': []}, 'metric': {'random': [], 'scenario': []}}
    t0 = time.time()
    for typ, pool, i in [('random', old, 1), ('random', old, 2), ('scenario', sc, 0), ('scenario', sc, 1)]:
        r = pool[i]['runs'][0]; Y = np.array([[o[k] for k in names] for o in r['observations']])
        tr = [d['runs'][0] for j, d in enumerate(old) if not (pool is old and j == i)] + [d['runs'][0] for j, d in enumerate(sc) if not (pool is sc and j == i)]
        Pb = T.predict([T.train(tr, names, ctl, bounds, sig, H=H, seed=k) for k in range(2)], r, names, ctl, bounds)
        Pm = T.predict([train_metric(tr, names, ctl, bounds, sig, H=H, seed=k) for k in range(2)], r, names, ctl, bounds)
        for tag, P in [('base', Pb), ('metric', Pm)]:
            S = 1 / (1 + np.abs(P - Y) / sig); out[tag][typ].append(S); late[tag][typ].append(S[50:])
    res = {tag: {t: [round(float(np.mean(np.vstack(out[tag][t]))), 4), round(float(np.mean(np.vstack(late[tag][t]))), 4)] for t in ('random', 'scenario')} for tag in out}
    print(json.dumps({'s': s, 'H': H, 'full_and_late': res, 'sec': round(time.time() - t0)}))
