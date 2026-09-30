"""Scoring-matched probe: actions between the reference recovery and pulse actions."""
import json, os, random, re, sys
from pathlib import Path
from client import Client

BRIEFS = Path(__file__).with_name('briefs.md').read_text()

def refs(family):
    sec = BRIEFS.split(f'## {family}\n', 1)[1].split('\n## ', 1)[0]
    rec = json.loads(re.search(r'Reference recovery action: `(\{.*?\})`', sec).group(1))
    pul = json.loads(re.search(r'Reference pulse action: `(\{.*?\})`', sec).group(1))
    return rec, pul

def schedule(family, steps, rng):
    rec, pul = refs(family); names = list(rec)
    def mix(active, lo=0.7, hi=1.0):
        return {k: rec[k] + (rng.uniform(lo, hi) if k in active else 0.0) * (pul[k] - rec[k]) for k in names}
    out = []
    def hold(a, n): out.extend([dict(a)] * n)
    hold(rec, rng.choice([5, 20, 40]))
    while len(out) < steps:
        kind = rng.choice(['sustained', 'order', 'recovery', 'composition'])
        if kind == 'sustained':
            hold(mix(names if rng.random() < 0.6 else rng.sample(names, rng.randint(1, len(names)))), rng.randint(60, 180))
            hold(rec, rng.randint(30, 120))
        elif kind == 'order':
            a = mix(rng.sample(names, rng.randint(1, len(names)))); b = mix(rng.sample(names, rng.randint(1, len(names))))
            d1, d2 = rng.randint(10, 50), rng.randint(10, 50)
            for x, y in ([(a, b), (b, a)] if rng.random() < 0.5 else [(b, a), (a, b)]):
                hold(x, d1); hold(y, d2); hold(rec, rng.randint(20, 60))
        elif kind == 'recovery':
            p = mix(names if rng.random() < 0.5 else rng.sample(names, rng.randint(1, len(names))))
            dur = rng.randint(8, 40)
            for _ in range(rng.randint(2, 4)):
                hold(p, dur); hold(rec, rng.choice([5, 15, 30, 60, 120]))
        else:
            for k in rng.sample(names, len(names)):
                hold(mix([k]), rng.randint(15, 50)); hold(rec, rng.randint(5, 40))
            hold(mix(names), rng.randint(20, 60)); hold(rec, rng.randint(20, 60))
    return out[:steps]

def main(family, steps, out, seed):
    out = Path(out); assert not out.exists(), out
    rng = random.Random(seed); sched = schedule(family, steps, rng)
    with Client(os.environ['GROUNDTRUTH_GATEWAY_URL'], os.environ['GROUNDTRUTH_KEY']) as c:
        brief = c.brief(family)
        b = brief['interventions']
        sched = [{k: min(max(v, b[k][0]), b[k][1]) for k, v in a.items()} for a in sched]
        reset = c.reset(family)
        run = {'initial': reset['observation'], 'actions': [], 'observations': []}
        data = {'family': family, 'brief': brief, 'runs': [run]}
        for i, a in enumerate(sched):
            obs = c.step(reset['run_id'], a)['observation']
            run['actions'].append(a); run['observations'].append(obs)
            if i % 50 == 49: out.write_text(json.dumps(data) + '\n')
        out.write_text(json.dumps(data) + '\n')
    print('saved', len(run['actions']), out)

if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]))
