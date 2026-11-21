import sys, os
sys.path.insert(0,'C:/Users/Administrator/Desktop/study/论文复现/code'); os.chdir('C:/Users/Administrator/Desktop/study/论文复现/code')
import numpy as np
from collections import defaultdict
import eval as E, metrics as M
fdf = E.pd.read_pickle('factors_SEREN_S2018.pkl')
fdf['nov_q']=M.quantile_normalize(fdf['novelty'].values); fdf['unpop_q']=M.quantile_normalize(fdf['unpopularity'].values)
fdf['rare']=0.5*(fdf['nov_q']+fdf['unpop_q'])
pos = fdf[fdf['label']==1]
pu, pr = [], []
for uid, g in pos.groupby('uid'):
    pu.append(uid); pr.append([g[c].mean() for c in E.PROFILE_COLS])
pr = np.array(pr); pr = pr/np.linalg.norm(pr,axis=1,keepdims=True)
S = pr@pr.T; ss={}
for i,uid in enumerate(pu):
    row=S[i].copy(); row[i]=-1.0
    t=np.argpartition(row,-E.PROFILE_TOPK)[-E.PROFILE_TOPK:]; t=t[np.argsort(row[t])[::-1]]
    ss[uid]=[(pu[j],float(row[j])) for j in t if row[j]>=0]
ibu=defaultdict(list)
for _,r in fdf.iterrows():
    ibu[r['uid']].append((r['item'],r['rel'],r['dif'],r['nov_q'],r['unpop_q'],r['qua'],r['label']))
user_seq,_,_,_ = E.load_serenlens('SerenLens_S2018.csv')
ui={u:set(it for it,ts,rr in seq) for u,seq in user_seq.items()}
valid=[u for u in ss if u in ibu and u in ui]
rng=np.random.RandomState(42); ok=0; npos=[]
for _ in range(300):
    u=valid[rng.randint(len(valid))]
    p=E.build_sim_user_pools(u,ss,ibu,ui,10,pool_dedup='none',pos_ratio_thr=0.5)
    if p is not None and p[0] is not None:
        ok+=1; npos.append(len(p[0]))
print('  S2018(新口径): 有画像用户 {} | 300 次抽样成功建池 {} 次 | 平均 pos_pool={:.2f}'.format(
    len(valid), ok, float(np.mean(npos)) if npos else 0))
