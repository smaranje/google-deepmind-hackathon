"""GRU state-space forecaster (trained offline, runs in NumPy only)."""
import json
from pathlib import Path
import numpy as np

_M = None


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def predict(initial, interventions, context):
    global _M
    if _M is None:
        _M = json.loads(Path(__file__).with_name('model.json').read_text())
    m = _M
    if not interventions:
        return []
    names, ctl, b = m['observables'], m['controls'], m['bounds']
    lo = np.array([b[c][0] for c in ctl]); sp = np.array([b[c][1] - b[c][0] for c in ctl])
    U = (np.array([[a[c] for c in ctl] for a in interventions], float) - lo) / sp
    outs = []
    for net in m['nets']:
        mu, sd = np.array(net['mu']), np.array(net['sd'])
        x0 = (np.array([initial[n] for n in names], float) - mu) / sd
        Wih, bih = np.array(net['w_ih']), np.array(net['b_ih'])
        Whh, bhh = np.array(net['w_hh']), np.array(net['b_hh'])
        H = Whh.shape[1]
        h = np.tanh(np.array(net['w_h0']) @ x0 + np.array(net['b_h0']))
        Xi = U @ Wih.T + bih
        hs = np.empty((len(U), H))
        for t in range(len(U)):
            gh = Whh @ h + bhh; xi = Xi[t]
            r = _sig(xi[:H] + gh[:H]); z = _sig(xi[H:2 * H] + gh[H:2 * H])
            n = np.tanh(xi[2 * H:] + r * gh[2 * H:])
            h = (1 - z) * n + z * h; hs[t] = h
        outs.append((hs @ np.array(net['w_out']).T + np.array(net['b_out'])) * sd + mu)
    P = np.clip(np.mean(outs, 0), np.array(m['lo']), np.array(m['hi']))
    P = np.where(np.isfinite(P), P, np.array(m['fallback']))
    return [dict(zip(names, row)) for row in P.tolist()]
