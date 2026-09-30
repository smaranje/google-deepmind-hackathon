"""GRU trained with extra weight on ticks deep inside constant-action holds (steady-state behaviour),
scored on (a) all ticks and (b) ticks >= 30 steps into a hold (proxy for the long-hold 'id' band)."""
import sys, json, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1); sys.path.insert(0, 'gru'); sys.path.insert(0, 'research')
import train_gru as T; from common import load
def hold_age(actions):
    age = np.zeros(len(actions)); prev = None
    for i, a in enumerate(actions):
        age[i] = age[i - 1] + 1 if (prev is not None and a == prev) else 0; prev = a
    return age
def train_hold(runs, names, ctl, bounds, sig, H=32, epochs=1500, seed=0, alpha=3.0):
    Yall = np.array([[o[k] for k in names] for r in runs for o in r['observations']]); mu, sd = Yall.mean(0), Yall.std(0) + 1e-9
    X0, U, Y, M = T.prep(runs, names, ctl, bounds, mu, sd); w = torch.tensor(sd / sig, dtype=torch.float32)
    HW = torch.ones_like(M)
    for i, r in enumerate(runs):
        ag = hold_age(r['actions']); HW[i, :len(ag), 0] = torch.tensor(1 + alpha * (ag >= 30), dtype=torch.float32)
    torch.manual_seed(seed); net = T.Net(len(ctl), len(names), H); opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        opt.zero_grad(); P = net(X0, U)
        loss = (nn.functional.smooth_l1_loss(P * w, Y * w, reduction='none', beta=0.05) * M * HW).sum() / ((M * HW).sum() * len(names))
        loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step()
    return net, mu, sd
if __name__ == '__main__':
    s, H, fold = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); sig = np.array(json.load(open('research/sigma_cal.json'))[s])
    old = [json.load(open(p)) for p in ['power-grid-research.json', 'research/power_grid_holdout.json', 'research/power_grid_s2.json']] if s == 'power_grid' else load(s)
    sc = [json.load(open(f'research/{s}_sc{i}.json')) for i in (1, 2)]; b = old[0]['brief']; names = b['observables']; bounds = b['interventions']; ctl = list(bounds)
    typ, pool, i = [('random', old, 1), ('random', old, 2), ('scenario', sc, 0), ('scenario', sc, 1)][fold]
    r = pool[i]['runs'][0]; Y = np.array([[o[k] for k in names] for o in r['observations']]); ag = hold_age(r['actions'])
    tr = [d['runs'][0] for j, d in enumerate(old) if not (pool is old and j == i)] + [d['runs'][0] for j, d in enumerate(sc) if not (pool is sc and j == i)]
    Pb = T.predict([T.train(tr, names, ctl, bounds, sig, H=H, seed=k) for k in range(2)], r, names, ctl, bounds)
    Ph = T.predict([train_hold(tr, names, ctl, bounds, sig, H=H, seed=k) for k in range(2)], r, names, ctl, bounds)
    f = lambda P, msk=None: float(np.mean((1 / (1 + np.abs(P - Y) / sig))[msk] if msk is not None else 1 / (1 + np.abs(P - Y) / sig)))
    deep = ag >= 30
    print(json.dumps({'s': s, 'fold': fold, 'type': typ, 'n_deep': int(deep.sum()), 'base': [f(Pb), f(Pb, deep)], 'hold': [f(Ph), f(Ph, deep)]}))
