"""Differentiable hospital-queue simulator fitted on free-running rollouts."""
import sys, json, time, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1); sys.path.insert(0, 'research'); from common import load
F = nn.functional
def smin(x, y, k=0.5): return y - F.softplus(y - x, beta=1 / k) if False else -k * torch.logaddexp(-x / k, -y / k)

class Sim(nn.Module):
    def __init__(s, g):
        super().__init__(); s.g = g
        P = lambda *v: nn.Parameter(torch.tensor(v, dtype=torch.float32))
        s.lam0 = P(1.5); s.lamE = P(0.0); s.urg = P(0.0); s.ra = P(-1.0); s.rt = P(-1.5); s.ot = P(0.0); s.fu = P(-1.0)
        s.Ca = P(3.0); s.Cb = P(3.5); s.kl = P(-4.0); s.Qmax = P(5.5); s.kf = P(-3.0); s.fe = P(0.0)
        s.kh = P(-2.0); s.he = P(0.0); s.kr = P(-3.0); s.rf = P(-1.0); s.fr = P(0.0)
        s.frac = P(0.0, -1.0, -1.0); s.wsc = P(0.0); s.wema = P(-2.0); s.dsc = P(0.0)
    def forward(s, x0, U):
        B, T, _ = U.shape; m1, m2, m3 = s.g; q0 = x0[:, 1:2]
        fr = torch.softmax(torch.cat([s.frac, torch.zeros(1)]), 0)
        Wq, Sa, Wt, St = q0 * fr[0], q0 * fr[1], q0 * fr[2], q0 * fr[3]
        Fat = torch.zeros(B, 1); Sema = U[:, 0, 0:1].clone(); R = torch.zeros(B, 1); adm = torch.ones(B, 1) * 5
        Ca, Cb = torch.exp(s.Ca), torch.exp(s.Cb); outs = []
        for t in range(T):
            st, el, dg, up, ot, fu = [U[:, t, i:i + 1] for i in range(6)]
            eff = st * (1 + F.softplus(s.ot) * ot) * (1 - m1 * torch.sigmoid(s.fe) * Fat) - m2 * torch.sigmoid(s.he) * torch.abs(st - Sema) - F.softplus(s.fu) * fu * st * 0.3
            eff = F.softplus(eff)
            arr = F.softplus(s.lam0) * (1 + 0.3 * torch.tanh(s.urg) * up) + F.softplus(s.lamE) * el / 10 + m3 * R * torch.sigmoid(s.kr)
            R = R - m3 * R * torch.sigmoid(s.kr)
            Wq = Wq + arr
            admit = smin(Wq, F.relu(Ca - Sa - Wt)); Wq = Wq - admit; Sa = Sa + admit
            done_a = smin(Sa, eff * dg * F.softplus(s.ra)); Sa = Sa - done_a; Wt = Wt + done_a
            move = smin(Wt, F.relu(Cb - St)); Wt = Wt - move; St = St + move
            dis = smin(St, eff * (1 - dg) * F.softplus(s.rt)); St = St - dis
            ret = dis * torch.sigmoid(s.rf) * (1 - torch.sigmoid(s.fr) * fu); R = R + m3 * ret
            leave = Wq * torch.sigmoid(s.kl); over = F.softplus(Wq - torch.exp(s.Qmax)); Wq = F.relu(Wq - leave - over)
            Fat = Fat + torch.sigmoid(s.kf) * (ot - Fat); Sema = Sema + torch.sigmoid(s.kh) * (st - Sema)
            adm = adm + torch.sigmoid(s.wema) * (admit - adm)
            wait = torch.exp(s.wsc) * Wq / (adm + 0.5)
            outs.append(torch.cat([wait, Wq + Sa + Wt + St, dis * torch.exp(s.dsc)], 1))
        return torch.stack(outs, 1)

CTL = ['staffing', 'elective_scheduling', 'diagnostic_allocation', 'urgent_priority', 'overtime', 'followup_capacity']
def batch(runs, names):
    L = max(len(r['actions']) for r in runs); X0 = torch.tensor([[r['initial'][n] for n in names] for r in runs], dtype=torch.float32)
    U = torch.zeros(len(runs), L, 6); Y = torch.zeros(len(runs), L, 3); M = torch.zeros(len(runs), L, 1)
    for i, r in enumerate(runs):
        n = len(r['actions']); U[i, :n] = torch.tensor([[a[c] for c in CTL] for a in r['actions']])
        if n < L: U[i, n:] = U[i, n - 1]
        Y[i, :n] = torch.tensor([[o[k] for k in names] for o in r['observations']]); M[i, :n] = 1
    return X0, U, Y, M

def fit(runs, names, sig, g, epochs=500):
    torch.manual_seed(0); m = Sim(g); X0, U, Y, M = batch(runs, names); w = 1 / torch.tensor(sig, dtype=torch.float32)
    opt = torch.optim.Adam(m.parameters(), lr=3e-2); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs); best = (1e9, None)
    for ep in range(epochs):
        opt.zero_grad(); P = m(X0, U); loss = (F.smooth_l1_loss(P * w, Y * w, reduction='none', beta=0.1) * M).sum() / (M.sum() * 3)
        if not torch.isfinite(loss): break
        if loss.item() < best[0]: best = (loss.item(), {k: v.detach().clone() for k, v in m.state_dict().items()})
        loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
    if best[1]: m.load_state_dict(best[1])
    return m

if __name__ == '__main__':
    fold, tag = int(sys.argv[1]), sys.argv[2]; g = {'12': (1., 1., 0.), '13': (1., 0., 1.), '23': (0., 1., 1.)}[tag]
    s = 'hospital_queue'; sig = np.array(json.load(open('research/sigma_cal.json'))[s])
    old = load(s); sc = [json.load(open(f'research/{s}_sc{i}.json')) for i in (1, 2)]; names = old[0]['brief']['observables']
    if fold < 0:
        m = fit([d['runs'][0] for d in old + sc], names, sig, g, epochs=700)
        json.dump({k: v.detach().numpy().tolist() for k, v in m.named_parameters()}, open(f'diffsim/hosp_params_{tag}.json', 'w'))
        r = sc[1]['runs'][0]; X0, U, Y, M = batch([r], names)
        with torch.no_grad(): np.save(f'diffsim/hosp_check_{tag}.npy', m(X0, U)[0].numpy())
        print('final written'); sys.exit()
    typ, pool, i = [('random', old, 1), ('random', old, 2), ('scenario', sc, 0), ('scenario', sc, 1)][fold]
    r = pool[i]['runs'][0]; tr = [d['runs'][0] for j, d in enumerate(old) if not (pool is old and j == i)] + [d['runs'][0] for j, d in enumerate(sc) if not (pool is sc and j == i)]
    t0 = time.time(); m = fit(tr, names, sig, g); X0, U, Y, M = batch([r], names)
    with torch.no_grad(): P = m(X0, U)[0].numpy()
    ok = np.isfinite(P).all(); S = 1 / (1 + np.abs(P - Y[0].numpy()) / sig)
    print(json.dumps({'fold': fold, 'type': typ, 'tag': tag, 'score': round(float(S.mean()), 4) if ok else 0.0, 'by_obs': np.round(S.mean(0), 3).tolist(), 'sec': round(time.time() - t0)}))
