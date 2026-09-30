"""Direct history-feature forecaster with interactions and random nonlinear features."""
import json
from pathlib import Path
import numpy as np

_MODEL = None


def features(initial, interventions, model):
    names, controls, bounds = model['observables'], model['controls'], model['bounds']
    center, scale = np.asarray(model['center']), np.asarray(model['scale'])
    lo = np.asarray([bounds[c][0] for c in controls]); span = np.asarray([bounds[c][1] - bounds[c][0] for c in controls])
    u = (np.asarray([[a[c] for c in controls] for a in interventions], float) - lo) / span
    x0 = (np.asarray([initial[n] for n in names], float) - center) / scale
    n, k = u.shape
    t = np.arange(1, n + 1)[:, None]
    emas = {}
    cols = [u]
    for h in model['ema_halflives']:
        alpha = 1 - 0.5 ** (1 / h); e = np.zeros(k); out = np.empty_like(u)
        for i in range(n):
            e = e + alpha * (u[i] - e); out[i] = e
        emas[h] = out; cols.append(out)
    for h in model['decay_halflives']:
        d = 0.5 ** (t / h)
        cols += [d, d * x0[None, :]]
    iu = np.triu_indices(k)
    cols.append((u[:, :, None] * u[:, None, :])[:, iu[0], iu[1]])
    for h in model.get('cross_halflives', []):
        cols.append((u[:, :, None] * emas[h][:, None, :]).reshape(n, -1))
        cols.append((emas[h][:, :, None] * emas[h][:, None, :])[:, iu[0], iu[1]])
    base = np.hstack(cols)
    if model.get('rff_w'):
        z = (base - np.asarray(model['feat_mu'])) / np.asarray(model['feat_sd'])
        base = np.hstack([base, np.cos(z @ np.asarray(model['rff_w']) + np.asarray(model['rff_b']))])
    return np.hstack([base, np.ones((n, 1))])


def predict(initial, interventions, context):
    global _MODEL
    if _MODEL is None:
        _MODEL = json.loads(Path(__file__).with_name('model.json').read_text())
    m = _MODEL
    if not interventions:
        return []
    y = features(initial, interventions, m) @ np.asarray(m['w'])
    phys = np.asarray(m['center']) + np.asarray(m['scale']) * y
    phys = np.clip(phys, np.asarray(m['lo']), np.asarray(m['hi']))
    phys = np.where(np.isfinite(phys), phys, np.asarray(m['center']))
    return [dict(zip(m['observables'], row)) for row in phys.tolist()]
