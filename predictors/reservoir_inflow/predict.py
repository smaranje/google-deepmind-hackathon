"""Reservoir: GRU ensemble forecast, with inflow replaced by its fitted seasonal cycle.

Inflow is identical across research runs at the same tick (correlation 0.995) and
follows one sinusoid (period ~67.8 ticks), independent of the controls.
"""
import importlib.util
import json
from pathlib import Path
import numpy as np

_HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location('reservoir_base', _HERE / 'base' / 'predict.py')
_base = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_base)
_FIT = json.loads((_HERE / 'inflow.json').read_text())


def _inflow(n):
    p, t = _FIT['p'], np.arange(1, n + 1, dtype=float)
    out = np.full(n, p[0])
    for k in range(_FIT['K']):
        P, a, b = p[1 + 3 * k:4 + 3 * k]
        out += a * np.sin(2 * np.pi * t / P) + b * np.cos(2 * np.pi * t / P)
    return out


def predict(initial, interventions, context):
    out = _base.predict(initial, interventions, context)
    for row, v in zip(out, _inflow(len(out)).tolist()):
        row['inflow'] = v
    return out
