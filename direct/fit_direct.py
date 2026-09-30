import json, sys
import numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from predict import features

def fit(datasets, lam=1.0, ema=(1, 2, 4, 8, 16, 32, 64, 128), decay=(2, 8, 32, 128)):
    brief = datasets[0]['brief']; names = list(brief['observables']); bounds = brief['interventions']
    Y = np.array([[o[n] for n in names] for d in datasets for r in d['runs'] for o in r['observations']])
    m = {'family': datasets[0]['family'], 'observables': names, 'controls': list(bounds), 'bounds': bounds,
         'center': Y.mean(0).tolist(), 'scale': np.maximum(Y.std(0), 1e-9).tolist(),
         'ema_halflives': list(ema), 'decay_halflives': list(decay)}
    span = Y.max(0) - Y.min(0)
    m['lo'] = (Y.min(0) - 0.25 * span).tolist(); m['hi'] = (Y.max(0) + 0.25 * span).tolist()
    X, T = [], []
    for d in datasets:
        for r in d['runs']:
            X.append(features(r['initial'], r['actions'], m))
            T.append((np.array([[o[n] for n in names] for o in r['observations']]) - Y.mean(0)) / m['scale'])
    X, T = np.vstack(X), np.vstack(T)
    P = lam * np.eye(X.shape[1]); P[-1, -1] = 0
    m['w'] = np.linalg.solve(X.T @ X + P, X.T @ T).tolist()
    return m
