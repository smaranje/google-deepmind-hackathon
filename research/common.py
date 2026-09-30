import json, sys, numpy as np, importlib.util
sys.path.insert(0,'direct'); import predict as D; from fit_direct import fit as dfit
sys.path.insert(0,'.'); import fit as F
_s=importlib.util.spec_from_file_location('lin','example_submission/predict.py'); L=importlib.util.module_from_spec(_s); _s.loader.exec_module(L)
SYS="power_grid epidemic market traffic supply_chain wildlife reservoir ad_auction social_contagion hospital_queue".split()
REAL={'ad_auction':.5485,'epidemic':.3526,'hospital_queue':.5406,'market':.5081,'power_grid':.5803,'reservoir':.5664,'social_contagion':.4526,'supply_chain':.3512,'traffic':.5800,'wildlife':.4803}
def files(s):
    if s=='power_grid': return ['power-grid-research.json','research/power_grid_holdout.json','research/power_grid_s2.json']
    return [f'research/{s}.json',f'research/{s}_s2.json',f'research/{s}_s3.json']
def load(s): return [json.load(open(p)) for p in files(s)]
def Y(r,names): return np.array([[o[k] for k in names] for o in r['observations']])
def P(pred,names): return np.array([[p[k] for k in names] for p in pred])
def starter(tr,s): L._MODEL=F.fit({'family':s,'brief':tr[0]['brief'],'runs':[x['runs'][0] for x in tr]}); return lambda r: L.predict(r['initial'],r['actions'],{'family':s})
def direct(tr,lam,**kw): D._MODEL=dfit(tr,lam,**kw); m=D._MODEL; return lambda r,m=m: (setattr(D,'_MODEL',m), D.predict(r['initial'],r['actions'],{}))[1]
def cv_errors(runs,s,maker):
    names=runs[0]['brief']['observables']; out=[]
    for i in range(len(runs)):
        tr=[d for j,d in enumerate(runs) if j!=i]; r=runs[i]['runs'][0]
        out.append(np.abs(P(maker(tr,s)(r),names)-Y(r,names)))
    return np.vstack(out)
def score(E,sig): return float((1/(1+E/sig)).mean())
def calib(E,std,target):
    lo,hi=1e-4,10.0
    for _ in range(60):
        c=(lo*hi)**.5
        if score(E,c*std)<target: lo=c
        else: hi=c
    return c
