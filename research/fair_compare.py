import sys,json,numpy as np; sys.path.insert(0,'research'); from common import *
import importlib.util as u
sp=u.spec_from_file_location('f2','direct2/fit_direct2.py'); f2=u.module_from_spec(sp); sp.loader.exec_module(f2)
sp=u.spec_from_file_location('d2p','direct2/predict.py'); D2=u.module_from_spec(sp); sp.loader.exec_module(D2)
def d2(tr,**kw): m=f2.fit(tr,**kw); return lambda r,m=m: (setattr(D2,'_MODEL',m), D2.predict(r['initial'],r['actions'],{}))[1]
X=dict(cross=(4,32)); R=dict(cross=(4,32),rff=300,rff_scale=0.3)
DEP={'power_grid':('d1',3),'epidemic':('d2',dict(lam=30)),'market':('d1',3),'traffic':('starter',0),'supply_chain':('d2',dict(lam=10,**R)),
     'wildlife':('d1',30),'reservoir':('d1',10),'ad_auction':('starter',0),'social_contagion':('d1',3),'hospital_queue':('starter',0)}
V5={'power_grid':('d2',dict(lam=30,**X),3),'epidemic':('starter',0,1),'market':('d2',dict(lam=30,**R),1),'traffic':('d2',dict(lam=30,**X),3),
    'supply_chain':('d2',dict(lam=10,**R),1),'wildlife':('d2',dict(lam=10,**X),3),'reservoir':('d2',dict(lam=30,**R),3),
    'ad_auction':('d2',dict(lam=10,**R),3),'social_contagion':('d2',dict(lam=10,**R),1),'hospital_queue':('starter',0,1)}
def mk(kind,arg,s):
    if kind=='starter': return lambda tr: starter(tr,s)
    if kind=='d1': return lambda tr: direct(tr,arg)
    return lambda tr: d2(tr,**arg)
s=sys.argv[1]; sig=np.array(json.load(open('research/sigma_cal.json'))[s])
old=load(s); sc=[json.load(open(f'research/{s}_sc{i}.json')) for i in (1,2)]
names=old[0]['brief']['observables']
folds=[('random',old,1),('random',old,2),('scenario',sc,0),('scenario',sc,1)]
res={'dep':{'random':[],'scenario':[]},'v5':{'random':[],'scenario':[]}}
for typ,pool,i in folds:
    r=pool[i]['runs'][0]
    others_old=[d for j,d in enumerate(old) if not (pool is old and j==i)]
    others_sc=[d for j,d in enumerate(sc) if not (pool is sc and j==i)]
    k,a=DEP[s]; pd=mk(k,a,s)(others_old+others_sc)(r)
    k,a,up=V5[s]; pv=mk(k,a,s)(others_old+others_sc*up)(r)
    res['dep'][typ].append(np.abs(P(pd,names)-Y(r,names))); res['v5'][typ].append(np.abs(P(pv,names)-Y(r,names)))
out={m:{t:score(np.vstack(v),sig) for t,v in d.items()} for m,d in res.items()}
print(json.dumps({'s':s,**out}))
