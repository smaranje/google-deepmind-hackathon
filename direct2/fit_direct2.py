import sys
import numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import importlib.util as _u; _sp=_u.spec_from_file_location("d2p", __file__.rsplit("/",1)[0]+"/predict.py"); _m=_u.module_from_spec(_sp); _sp.loader.exec_module(_m); features=_m.features

def fit(datasets, lam=3.0, ema=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512), decay=(2, 8, 32, 128),
        cross=(), rff=0, rff_scale=0.5, seed=0):
    brief = datasets[0]['brief']; names = list(brief['observables']); bounds = brief['interventions']
    Y = np.array([[o[n] for n in names] for d in datasets for r in d['runs'] for o in r['observations']])
    m = {'family': datasets[0]['family'], 'observables': names, 'controls': list(bounds), 'bounds': bounds,
         'center': Y.mean(0).tolist(), 'scale': np.maximum(Y.std(0), 1e-9).tolist(),
         'ema_halflives': list(ema), 'decay_halflives': list(decay), 'cross_halflives': list(cross)}
    span = Y.max(0) - Y.min(0)
    m['lo'] = (Y.min(0) - 0.25 * span).tolist(); m['hi'] = (Y.max(0) + 0.25 * span).tolist()
    runs = [r for d in datasets for r in d['runs']]
    X = np.vstack([features(r['initial'], r['actions'], m) for r in runs])[:, :-1]
    if rff:
        mu, sd = X.mean(0), X.std(0) + 1e-6
        rng = np.random.default_rng(seed)
        m.update(feat_mu=mu.tolist(), feat_sd=sd.tolist(),
                 rff_w=(rng.normal(size=(X.shape[1], rff)) * rff_scale / np.sqrt(X.shape[1])).tolist(),
                 rff_b=rng.uniform(0, 2 * np.pi, rff).tolist())
    X = np.vstack([features(r['initial'], r['actions'], m) for r in runs])
    T = np.vstack([(np.array([[o[n] for n in names] for o in r['observations']]) - Y.mean(0)) / m['scale'] for r in runs])
    P = lam * np.eye(X.shape[1]); P[-1, -1] = 0
    m['w'] = np.linalg.solve(X.T @ X + P, X.T @ T).tolist()
    return m
