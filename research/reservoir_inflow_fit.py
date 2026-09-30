"""Reservoir inflow is action-independent: fit it as a sum of sinusoids on early ticks, test on later ticks.

Run from the kit root with the five reservoir research runs in research/.
Fits on ticks <= 256 (pooled over runs) and scores on ticks 256-450, then refits on all data
with the best K and writes research/reservoir_inflow_fit.json (consumed by predictors/reservoir_inflow).
"""
import json
import numpy as np
from scipy.optimize import least_squares

RUNS = [json.load(open(f'research/reservoir{x}.json'))['runs'][0] for x in ['', '_s2', '_s3', '_sc1', '_sc2']]
SIGMA = json.load(open('research/sigma_cal.json'))['reservoir'][1]  # calibrated sigma of `inflow` (not published)

# Cross-run check that motivates the model: inflow at the same tick is nearly identical across runs.
L = min(len(r['observations']) for r in RUNS)
I = np.array([[o['inflow'] for o in r['observations'][:L]] for r in RUNS])
print('mean off-diagonal correlation across runs: %.3f' % ((np.corrcoef(I).sum() - len(RUNS)) / (len(RUNS) * (len(RUNS) - 1))))

T, Y = [], []
for r in RUNS:
    y = np.array([o['inflow'] for o in r['observations']])
    T.append(np.arange(1, len(y) + 1)); Y.append(y)
T, Y = np.concatenate(T), np.concatenate(Y)


def model(p, t, K):
    out = np.full(t.shape, p[0])
    for k in range(K):
        P, a, b = p[1 + 3 * k:4 + 3 * k]
        out += a * np.sin(2 * np.pi * t / P) + b * np.cos(2 * np.pi * t / P)
    return out


best = None
for K, init in [(1, [64]), (2, [64, 32]), (3, [64, 85, 51]), (3, [64, 32, 21.3]), (4, [64, 32, 85, 51])]:
    p0 = [11.4] + sum([[P, 1, 0] for P in init], [])
    tr, te = T <= 256, T > 256
    f = least_squares(lambda p: model(p, T[tr], K) - Y[tr], p0)
    e_te = np.abs(model(f.x, T[te], K) - Y[te])
    s_te = float(np.mean(1 / (1 + e_te / SIGMA)))
    print(f'K={K} periods={np.round(f.x[1::3], 2)} held-out (ticks>256) inflow score {s_te:.3f}')
    if best is None or s_te > best[0]:
        best = (s_te, K, f.x)

_, K, p = best
f = least_squares(lambda q: model(q, T, K) - Y, p)
print('best K', K, 'refit periods', np.round(f.x[1::3], 3))
json.dump({'K': K, 'p': f.x.tolist()}, open('research/reservoir_inflow_fit.json', 'w'))
