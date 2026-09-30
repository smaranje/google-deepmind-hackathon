"""Scoring-shaped probe: holds of varying length, subsets of controls, recovery gaps."""
import json, os, random, sys
from pathlib import Path
from client import Client

def main(family, steps, out, seed):
    out = Path(out)
    assert not out.exists(), out
    with Client(os.environ['GROUNDTRUTH_GATEWAY_URL'], os.environ['GROUNDTRUTH_KEY']) as c:
        brief = c.brief(family); bounds = brief['interventions']; names = list(bounds)
        reset = c.reset(family)
        run = {'initial': reset['observation'], 'actions': [], 'observations': []}
        data = {'family': family, 'brief': brief, 'runs': [run]}
        rng = random.Random(seed); low = {k: bounds[k][0] for k in names}
        while len(run['actions']) < steps:
            hold = rng.choice([4, 8, 16, 32, 64])
            kind = rng.random()
            if kind < 0.25:
                action = dict(low)                                  # recovery / idle
            else:
                active = rng.sample(names, rng.randint(1, len(names)))  # composition
                action = {k: (rng.uniform(*bounds[k]) if k in active else bounds[k][0]) for k in names}
            for _ in range(min(hold, steps - len(run['actions']))):
                obs = c.step(reset['run_id'], action)['observation']
                run['actions'].append(dict(action)); run['observations'].append(obs)
            out.write_text(json.dumps(data) + '\n')
    print('saved', len(run['actions']), 'to', out)

if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]))
