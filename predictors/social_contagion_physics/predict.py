"""Social contagion: fitted mechanistic simulator (credibility + incentive-expectation memory).

Per community: adopters, interest queue, cooldown pool and eligible population.
Onboarding is limited by a workforce shared with existing members; waiting and
unmet incentive expectations drive disappointment; the cooldown pool returns later.
"""
import json
from pathlib import Path
import numpy as np

_P = None


def _sp(x):
    return np.logaddexp(0.0, x)


def _sg(x):
    return 1.0 / (1.0 + np.exp(-x))


def predict(initial, interventions, context):
    global _P
    if _P is None:
        _P = {k: np.asarray(v, float) for k, v in json.loads(Path(__file__).with_name('params.json').read_text()).items()}
    p = _P
    A = np.array([initial['adopters_a'], initial['adopters_b']], float)
    Q = np.zeros(2); D = np.zeros(2); Pop = np.exp(p['logP'])
    C = np.full(2, _sg(p['cinit'][0])); E = np.zeros(2); al = _sg(p['al'])
    rmax, kq, ka = _sg(p['rmax'][0]), _sg(p['kq'][0]), _sg(p['ka'][0]); tau = 1 + _sp(p['tau'][0]) * 10
    out = []
    for a in interventions:
        sd, inc, br = a['seeding'] / 10, a['incentive'] / 2, a['bridge_outreach']
        L, X = sd * (1 - br), sd * br
        N = _sp(Pop - A - Q - D)
        other = (A / Pop)[::-1]
        logit = p['a0'] + p['aL'] * L + p['aI'][0] * inc + p['aC'][0] * C + p['aW'][0] * A / Pop + p['aX'][0] * X * other
        I = N * _sg(logit) * rmax
        W = _sp(p['w0'][0] * 10 - p['w1'][0] * A.sum() / 10)
        O = Q * (1 - np.exp(-W / (Q.sum() + 1.0)))
        gap = E - inc
        Qd = Q * kq * _sg(p['e0'][0] + p['e1'][0] * gap)
        Ch = A * ka * _sg(p['c0'][0] + p['c1'][0] * gap - p['c2'][0] * C)
        Rt = D / tau
        A = np.maximum(A + O - Ch, 0); Q = np.maximum(Q + I - O - Qd, 0); D = np.maximum(D + Qd + Ch - Rt, 0)
        C = C + al[0] * (O / (O + Qd + 1.0) - C); E = E + al[1] * (inc - E)
        out.append({'adopters_a': float(A[0]), 'adopters_b': float(A[1])})
    return out
