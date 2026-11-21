# -*- coding: utf-8 -*-
"""把 main.tex 的 RQ4 两张大表换成一张图 + 两段文字。"""
import io
import os

P = r"C:\Users\Administrator\Desktop\M_DCG\M_DCG\pdf\main.tex"

NEW = r"""\enlargethispage{\baselineskip}%
Figure~\ref{fig:rq4} traces the within-user Hit@2 as the quality factor disappears. The hit rate is non-monotonic on all three datasets, decreasing first and then recovering at the heaviest masking. The recovery has a simple explanation. Once almost every item's quality is masked, the score falls back to the remaining factors, which stay discriminative on their own, so the metric becomes coarser without collapsing. Even so, Hit@2 falls by roughly a third before the recovery on Books and Movies, and by less than a tenth on S2018, and under full masking it stays $3$ to $8$ times the random rate. The quality factor thus contributes to, but does not underwrite, the metric's ranking power.

Figure~\ref{fig:rq4} also places the four item-level factors and an optimally weighted linear combination of them on the same axes. The single factors are flat by construction, since none of them reads $\Qua$: their Hit@2 stays at $0.039$--$0.088$ on Books and Movies and at $0.540$--$0.722$ on S2018 throughout, against $0.507$, $0.474$ and $0.861$ for M-DCG at zero masking. The linear combination, whose weights are fitted on the labeled interactions of the other folds, is a much stronger competitor on Books and Movies, yet it reaches only $0.152$ and $0.181$ there, so M-DCG sits roughly $2.5$ to $3.3$ times higher, and its own curve is nearly flat as well ($0.109$--$0.152$ on Books, $0.136$--$0.190$ on Movies). Only on S2018 does the linear form come close, at $0.781$ against $0.861$. The masking therefore costs the linear form as little as it costs M-DCG; what separates the two is the level at which they start, not how fast they fall.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{rq4_masking.png}
\caption{Within-user Hit@2 as the quality factor is progressively masked. Thick blue: M-DCG; dashed orange: the optimal linear combination of the five factors (weights fitted out of fold); thin grey: each single item-level factor, which does not read $\Qua$ and is therefore flat in the mask.}
\label{fig:rq4}
\end{figure*}
"""

with io.open(P, "r", encoding="utf-8") as f:
    lines = f.readlines()

# 定位: 从 \enlargethispage 那行 到 第二张表的 \end{table}
start = None
for i, ln in enumerate(lines):
    if ln.strip() == r"\enlargethispage{\baselineskip}%" and i > 600:
        start = i
        break
if start is None:
    raise SystemExit("start marker not found")

end = None
seen_end = 0
for j in range(start, len(lines)):
    if lines[j].strip() == r"\end{table}":
        seen_end += 1
        if seen_end == 2:
            end = j
            break
if end is None:
    raise SystemExit("end marker not found")

print("replacing lines", start + 1, "..", end + 1)
old_block = "".join(lines[start:end + 1])
assert "tab:rq5" in old_block and "tab:rq5_ndcg" in old_block, "unexpected block"

out = lines[:start] + [NEW] + lines[end + 1:]
with io.open(P, "w", encoding="utf-8") as f:
    f.writelines(out)
print("done; file lines", len(out))
