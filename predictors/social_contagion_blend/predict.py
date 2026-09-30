"""Social contagion: 0.6 x mechanistic simulator + 0.4 x GRU (held-out best blend)."""
import importlib.util
from pathlib import Path

_HERE = Path(__file__).parent
W_PHYS = 0.6


def _load(name):
    spec = importlib.util.spec_from_file_location(f'social_{name}', _HERE / name / 'predict.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


_phys, _gru = _load('phys'), _load('gru')


def predict(initial, interventions, context):
    p = _phys.predict(initial, interventions, context)
    g = _gru.predict(initial, interventions, context)
    return [{k: W_PHYS * a[k] + (1 - W_PHYS) * b[k] for k in a} for a, b in zip(p, g)]
