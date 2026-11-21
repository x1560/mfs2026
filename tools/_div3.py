# -*- coding: utf-8 -*-
"""验证: 正样本之间是否比负样本之间更相似 -> div 随扰动比上升 -> 与标签信号反向"""
import sys, os
sys.path.insert(0,'C:/Users/Administrator/Desktop/study/论文复现/code')
os.chdir('C:/Users/Administrator/Desktop/study/论文复现/code')
import numpy as np
from collections import defaultdict
import eval as E, metrics as M
import itertools

user_seq, labels, label_time, item_rating = E.load_serenlens('./data/SerenLens/SerenLens_Movies.csv')
fdf = E.pd.read_pickle('factors_SEREN_Movies.pkl')
fdf['nov_q']=M.quantile_normalize(fdf['novelty'].values)
fdf['unpop_q']=M.quantile_normalize(fdf['unpopularity'].values)
fdf['rare']=0.5*(fdf['nov_q']+fdf['unpop_q'])
sim=E.SimEngine(E.load_embeddings('emb_SEREN_Movies.pkl'))
_,_ = E.load_item_meta('cate_SEREN_Movies.pkl','pop_SEREN_Movies.pkl')
ui={u:set(it for it,ts,r in s) for u,s in user_seq.items()}
pu,pr=[],[]
for uid,g in fdf[fdf['label']==1].groupby('uid'):
    pu.append(uid); pr.append([g[c].mean() for c in E.PROFILE_COLS])
pr=np.array(pr); pr=pr/(np.linalg.norm(pr,axis=1,keepdims=True)+1e-9)
sm=pr@pr.T; ss={}
for i,uid in enumerate(pu):
    row=sm[i].copy(); row[i]=-1.0
    t=np.argpartition(row,-E.PROFILE_TOPK)[-E.PROFILE_TOPK:]
    t=t[np.argsort(row[t])[::-1]]
    ss[uid]=[(pu[j],float(row[j])) for j in t if row[j]>=0]
ibu=defaultdict(list)
for _,r in fdf.iterrows():
    ibu[r['uid']].append((r['item'],r['rel'],r['dif'],r['nov_q'],r['unpop_q'],r['qua'],r['label']))
valid=[u for u in ss if u in ibu and u in ui]
rng=np.random.RandomState(0)
pp_sims=[]; nn_sims=[]; pn_sims=[]
divs_by_lvl=defaultdict(list)
levels=np.linspace(0,1,11); k=10
for _ in range(300):
    u=valid[rng.randint(len(valid))]
    ppo,npo=E.build_sim_user_pools(u,ss,ibu,ui,k,pool_dedup='none',pos_ratio_thr=0.5)
    if ppo is None or len(ppo)<2: continue
    P=[t[0] for t in ppo[:2]]; Ng=[t[0] for t in npo[:8]]
    for a,b in itertools.combinations(P,2): pp_sims.append(sim.sim(a,b))
    for a,b in itertools.combinations(Ng,2): nn_sims.append(sim.sim(a,b))
    for a in P:
        for b in Ng: pn_sims.append(sim.sim(a,b))
    for lvl in levels:
        nr=int(round(lvl*k)); lst=(P[:k-nr]+Ng[:nr])[:k]
        d=1.0
        for i in range(1,len(lst)):
            d*=M.div_discount([sim.sim(lst[i],lst[j]) for j in range(i)])
        divs_by_lvl[lvl].append(d)
print('正-正 相似度: mean={:+.4f}'.format(np.mean(pp_sims)))
print('负-负 相似度: mean={:+.4f}'.format(np.mean(nn_sims)))
print('正-负 相似度: mean={:+.4f}'.format(np.mean(pn_sims)))
print()
print('扰动比  平均 div(各位置乘积)')
for lvl in levels:
    v=divs_by_lvl[lvl]
    print('  {:3.0f}%   {:.4f}'.format(lvl*100, np.mean(v)))
