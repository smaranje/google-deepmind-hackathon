"""Differentiable social-contagion simulator (+ optional joint GRU correction), fitted on free-running rollouts."""
import sys, json, time, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1); sys.path.insert(0, 'research'); from common import load
F = nn.functional

class Sim(nn.Module):
    def __init__(s, gates, hybrid):
        super().__init__(); s.g = gates; s.hybrid = hybrid
        P = lambda *v: nn.Parameter(torch.tensor(v, dtype=torch.float32))
        s.logP = P(5.8, 5.6); s.a0 = P(-4.0, -4.0); s.aL = P(2.0, 2.0); s.aI = P(1.0); s.aC = P(1.0); s.aW = P(2.0); s.aX = P(2.0)
        s.rmax = P(-2.0); s.w0 = P(3.0); s.w1 = P(0.5); s.kq = P(-3.0); s.e0 = P(0.0); s.e1 = P(2.0)
        s.ka = P(-4.0); s.c0 = P(0.0); s.c1 = P(2.0); s.c2 = P(1.0); s.tau = P(3.0); s.al = P(-2.0, -2.0, -2.0); s.cinit = P(0.0)
        if hybrid:
            s.gru = nn.GRUCell(3 + 2, 16); s.out = nn.Linear(16, 2); nn.init.zeros_(s.out.weight); nn.init.zeros_(s.out.bias)
    def forward(s, x0, U):  # x0: (B,2) adopters; U: (B,T,3) raw controls
        B, T, _ = U.shape; A = x0.clone(); Q = torch.zeros_like(A); D = torch.zeros_like(A)
        Pop = torch.exp(s.logP).expand(B, 2); C = torch.sigmoid(s.cinit).expand(B, 2) * 1.0; E = torch.zeros(B, 2); R = torch.zeros(B, 1)
        m1, m2, m3 = s.g; al = torch.sigmoid(s.al); h = torch.zeros(B, 16) if s.hybrid else None; outs = []
        for t in range(T):
            sd, inc, br = U[:, t, 0:1] / 10, U[:, t, 1:2] / 2, U[:, t, 2:3]
            L = sd * (1 - br); X = sd * br
            N = F.softplus(Pop - A - Q - D)
            other = torch.flip(A / Pop, [1])
            logit = s.a0 + s.aL * L + s.aI * inc + m1 * s.aC * C + s.aW * A / Pop + s.aX * X * other * (1 + m3 * R)
            I = N * torch.sigmoid(logit) * torch.sigmoid(s.rmax)
            W = F.softplus(s.w0 * 10 - s.w1 * A.sum(1, keepdim=True) / 10)
            O = Q * (1 - torch.exp(-W / (Q.sum(1, keepdim=True) + 1.0)))
            gap = m2 * E - inc
            Qd = Q * torch.sigmoid(s.kq) * torch.sigmoid(s.e0 + s.e1 * gap)
            Ch = A * torch.sigmoid(s.ka) * torch.sigmoid(s.c0 + s.c1 * gap - s.c2 * m1 * C)
            Rt = D / (1 + F.softplus(s.tau) * 10)
            A = F.relu(A + O - Ch); Q = F.relu(Q + I - O - Qd); D = F.relu(D + Qd + Ch - Rt)
            C = C + al[0] * (O / (O + Qd + 1.0) - C); E = E + al[1] * (inc - E); R = R + al[2] * (X - R)
            y = A
            if s.hybrid:
                h = s.gru(torch.cat([U[:, t] / torch.tensor([10., 2., 1.]), A / 100], 1), h); y = A + 20 * s.out(h)
            outs.append(y)
        return torch.stack(outs, 1)

def batch(runs, names):
    L = max(len(r['actions']) for r in runs)
    X0 = torch.tensor([[r['initial'][n] for n in names] for r in runs], dtype=torch.float32)
    U = torch.zeros(len(runs), L, 3); Y = torch.zeros(len(runs), L, 2); M = torch.zeros(len(runs), L, 1)
    for i, r in enumerate(runs):
        n = len(r['actions']); U[i, :n] = torch.tensor([[a['seeding'], a['incentive'], a['bridge_outreach']] for a in r['actions']])
        Y[i, :n] = torch.tensor([[o[k] for k in names] for o in r['observations']]); M[i, :n] = 1
    return X0, U, Y, M

def fit(runs, names, sig, gates, hybrid, epochs=500, seed=0):
    torch.manual_seed(seed); m = Sim(gates, hybrid); X0, U, Y, M = batch(runs, names); w = 1 / torch.tensor(sig, dtype=torch.float32)
    opt = torch.optim.Adam(m.parameters(), lr=3e-2); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        opt.zero_grad(); P = m(X0, U); loss = (F.smooth_l1_loss(P * w, Y * w, reduction='none', beta=0.1) * M).sum() / (M.sum() * 2)
        if not torch.isfinite(loss): break
        loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
    return m

if __name__ == '__main__':
    fold, tag = int(sys.argv[1]), sys.argv[2]
    gates = {'12': (1., 1., 0.), '13': (1., 0., 1.), '23': (0., 1., 1.), 'all': (1., 1., 1.)}[tag.split('_')[0]]; hybrid = tag.endswith('_hyb')
    s = 'social_contagion'; sig = np.array(json.load(open('research/sigma_cal.json'))[s])
    old = load(s); sc = [json.load(open(f'research/{s}_sc{i}.json')) for i in (1, 2)]; names = old[0]['brief']['observables']
    typ, pool, i = [('random', old, 1), ('random', old, 2), ('scenario', sc, 0), ('scenario', sc, 1)][fold]
    r = pool[i]['runs'][0]; tr = [d['runs'][0] for j, d in enumerate(old) if not (pool is old and j == i)] + [d['runs'][0] for j, d in enumerate(sc) if not (pool is sc and j == i)]
    t0 = time.time(); m = fit(tr, names, sig, gates, hybrid)
    X0, U, Y, M = batch([r], names)
    with torch.no_grad(): P = m(X0, U)[0].numpy()
    Yn = Y[0].numpy(); ok = np.isfinite(P).all()
    sc_ = float(np.mean(1 / (1 + np.abs(P - Yn) / sig))) if ok else 0.0
    print(json.dumps({'fold': fold, 'type': typ, 'tag': tag, 'score': round(sc_, 4), 'sec': round(time.time() - t0)}))
