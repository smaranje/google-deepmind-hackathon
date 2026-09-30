import sys, json, time, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
sys.path.insert(0, 'research'); from common import load

class Net(nn.Module):
    def __init__(s, k, o, H):
        super().__init__(); s.h0 = nn.Linear(o, H); s.gru = nn.GRU(k, H, batch_first=True); s.out = nn.Linear(H, o)
    def forward(s, x0, u):
        h, _ = s.gru(u, torch.tanh(s.h0(x0)).unsqueeze(0)); return s.out(h)

def prep(runs, names, ctl, bounds, mu, sd):
    lo = np.array([bounds[c][0] for c in ctl]); sp = np.array([bounds[c][1] - bounds[c][0] for c in ctl])
    X0 = np.array([[(r['initial'][n] - m) / s for n, m, s in zip(names, mu, sd)] for r in runs], np.float32)
    L = max(len(r['actions']) for r in runs)
    U = np.zeros((len(runs), L, len(ctl)), np.float32); Y = np.zeros((len(runs), L, len(names)), np.float32); M = np.zeros((len(runs), L, 1), np.float32)
    for i, r in enumerate(runs):
        n = len(r['actions'])
        U[i, :n] = (np.array([[a[c] for c in ctl] for a in r['actions']]) - lo) / sp
        Y[i, :n] = (np.array([[o[k] for k in names] for o in r['observations']]) - mu) / sd; M[i, :n] = 1
    return map(torch.tensor, (X0, U, Y, M))

def train(runs, names, ctl, bounds, sig, H=32, epochs=1500, seed=0, lr=3e-3, wd=1e-4):
    Yall = np.array([[o[k] for k in names] for r in runs for o in r['observations']]); mu, sd = Yall.mean(0), Yall.std(0) + 1e-9
    X0, U, Y, M = prep(runs, names, ctl, bounds, mu, sd)
    w = torch.tensor(sd / sig, dtype=torch.float32)  # error in sigma units
    torch.manual_seed(seed); net = Net(len(ctl), len(names), H); opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        opt.zero_grad(); P = net(X0, U)
        loss = ((nn.functional.smooth_l1_loss(P, Y, reduction='none', beta=0.05) * w) * M).sum() / (M.sum() * len(names))
        loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
    return net, mu, sd

def predict(nets, r, names, ctl, bounds):
    outs = []
    for net, mu, sd in nets:
        X0, U, _, _ = prep([r], names, ctl, bounds, mu, sd)
        with torch.no_grad(): outs.append(net(X0, U)[0].numpy() * sd + mu)
    return np.mean(outs, 0)

if __name__ == '__main__':
    s = sys.argv[1]; H = int(sys.argv[2]) if len(sys.argv) > 2 else 32; seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 2; EP = int(sys.argv[4]) if len(sys.argv) > 4 else 1500
    sig = np.array(json.load(open('research/sigma_cal.json'))[s])
    old = load(s); sc = [json.load(open(f'research/{s}_sc{i}.json')) for i in (1, 2)]
    brief = old[0]['brief']; names = brief['observables']; bounds = brief['interventions']; ctl = list(bounds)
    E = {'random': [], 'scenario': []}; t = time.time()
    for typ, pool, i in [('random', old, 1), ('random', old, 2), ('scenario', sc, 0), ('scenario', sc, 1)]:
        r = pool[i]['runs'][0]
        tr = [d['runs'][0] for j, d in enumerate(old) if not (pool is old and j == i)] + [d['runs'][0] for j, d in enumerate(sc) if not (pool is sc and j == i)]
        nets = [train(tr, names, ctl, bounds, sig, H=H, seed=k, epochs=EP) for k in range(seeds)]
        P = predict(nets, r, names, ctl, bounds); Y = np.array([[o[k] for k in names] for o in r['observations']])
        E[typ].append(np.abs(P - Y))
    print(json.dumps({"s": s, "H": H, "ep": EP, **{t_: round(float(np.mean(1 / (1 + np.vstack(v) / sig))), 4) for t_, v in E.items()}, 'sec': round(time.time() - t)}))
