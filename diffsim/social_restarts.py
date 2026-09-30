import sys, json, numpy as np, torch
sys.path.insert(0, 'diffsim'); sys.path.insert(0, 'research')
import social as S; from common import load
fold, seed = int(sys.argv[1]), int(sys.argv[2])
_orig_init = S.Sim.__init__
def init2(self, g, h):
    _orig_init(self, g, h)
    if seed > 0:
        gen = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for p in self.parameters(): p.add_(0.4 * torch.randn(p.shape, generator=gen))
S.Sim.__init__ = init2
s = 'social_contagion'; sig = np.array(json.load(open('research/sigma_cal.json'))[s])
old = load(s); sc = [json.load(open(f'research/{s}_sc{i}.json')) for i in (1, 2)]; names = old[0]['brief']['observables']
def losses_and_pred(train, test):
    m = S.fit(train, names, sig, (1., 1., 0.), False, epochs=500)
    X0, U, Y, M = S.batch(train, names); w = 1 / torch.tensor(sig, dtype=torch.float32)
    with torch.no_grad():
        P = m(X0, U); L = float((torch.abs(P * w - Y * w) * M).sum() / (M.sum() * 2))
        X0t, Ut, _, _ = S.batch([test], names); Pt = m(X0t, Ut)[0].numpy()
    return L, Pt
typ, pool, i = [('random', old, 1), ('random', old, 2), ('scenario', sc, 0), ('scenario', sc, 1)][fold]
r = pool[i]['runs'][0]; tr = [d['runs'][0] for j, d in enumerate(old) if not (pool is old and j == i)] + [d['runs'][0] for j, d in enumerate(sc) if not (pool is sc and j == i)]
L, P = losses_and_pred(tr, r)
np.save(f'diffsim/rs_f{fold}_s{seed}.npy', P); print(json.dumps({'fold': fold, 'seed': seed, 'type': typ, 'train_loss': L}))
