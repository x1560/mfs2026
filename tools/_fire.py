# -*- coding: utf-8 -*-
"""threshold-pool 在 Amazon 共现空间里的触发率"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, 'C:/Users/Administrator/Desktop/study/论文复现/code')
os.chdir('C:/Users/Administrator/Desktop/study/论文复现/code')
import numpy as np
from collections import defaultdict
import eval as E, metrics as M, amazon_sim
K = 10; SEED = 42
user_seq, labels, label_time, item_rating = E.load_serenlens('./data/SerenLens/SerenLens_Movies.csv')
fdf = E.pd.read_pickle('factors_SEREN_Movies.pkl')
fdf['nov_q'] = M.quantile_normalize(fdf['novelty'].values); fdf['unpop_q'] = M.quantile_normalize(fdf['unpopularity'].values)
fdf['rare'] = 0.5*(fdf['nov_q']+fdf['unpop_q'])
emb = E.load_embeddings('emb_SEREN_Movies.pkl'); sim_emb = E.SimEngine(emb).sim
cooc = amazon_sim.build('Movies', min_user_items=30, item_filter=set(emb.keys()), topn=100)
pos_df = fdf[fdf['label']==1]; pu, pr = [], []
for uid, g in pos_df.groupby('uid'):
    pu.append(uid); pr.append([g[c].mean() for c in E.PROFILE_COLS])
pr = np.array(pr); pr = pr/np.linalg.norm(pr,axis=1,keepdims=True)
sm = pr@pr.T; ss={}
for i,uid in enumerate(pu):
    row=sm[i].copy(); row[i]=-1.0
    t=np.argpartition(row,-E.PROFILE_TOPK)[-E.PROFILE_TOPK:]; t=t[np.argsort(row[t])[::-1]]
    ss[uid]=[(pu[j],float(row[j])) for j in t if row[j]>=0]
ibu=defaultdict(list)
for _,r in fdf.iterrows():
    ibu[r['uid']].append((r['item'],r['rel'],r['dif'],r['nov_q'],r['unpop_q'],r['qua'],r['label'],r['uid']))
ui={u:set(it for it,ts,rr in seq) for u,seq in user_seq.items()}
valid=[u for u in ss if u in ibu and u in ui]
rng=np.random.RandomState(SEED); lists=[]
for _ in range(400):
    t=rng.choice(valid)
    p=E.build_sim_user_pools(t,ss,ibu,ui,K,pool_dedup='none',pos_ratio_thr=0.5)
    if p is None or p[0] is None: continue
    pp,nn=p
    lists.append([x[0] for x in (pp[:2]+nn[:8])])
for tag, simf in [('Amazon 共现', cooc.sim), ('现嵌入', sim_emb)]:
    fire=0; tot=0; mrefs=[]; ms=[]
    for its in lists:
        pv=[simf(its[a],its[b]) for a in range(len(its)) for b in range(a+1,len(its))]
        mr=float(np.mean(pv)) if pv else 0.0
        mrefs.append(mr)
        for p in range(1,len(its)):
            m=float(np.mean([simf(its[p],its[j]) for j in range(p)]))
            ms.append(m); tot+=1
            fire += 1 if m>mr else 0
    print('  {:12s} 列表平均相似度(参照)={:.4f} | 位置平均 m={:.4f} | div 触发率={:.1%}'.format(
        tag, float(np.mean(mrefs)), float(np.mean(ms)), fire/tot))
