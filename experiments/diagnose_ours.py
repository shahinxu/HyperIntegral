"""Diagnose an Ours run: does the tensor move, and where does the truth sit?

Answers three questions that come up when the measured AUC is near chance:

  1. is W frozen, as it was under the old in-graph clamp (bug 2)?
  2. how many candidates end pinned at the clipping floor, where they tie?
  3. do the true hyperedges score above the negatives at all?

Read-only with respect to the response tables; it writes a scratch run under
/tmp. Needs the scene preset in the environment, exactly as `paper_sweep`
sets it:

    ECOLOGY_HYPERGRAPH_PRESET=n9 python -m experiments.diagnose_ours

Without that variable the ecological scene falls back to n=60 and the run
either measures the wrong system or exhausts GPU memory.
"""

import numpy as np, sys, subprocess
def pick_gpu(need_gb=12):
    out=subprocess.run(["nvidia-smi","--query-gpu=index,memory.used,memory.total",
                        "--format=csv,noheader,nounits"],capture_output=True,text=True).stdout
    best,free=0,-1
    for ln in out.strip().splitlines():
        i,u,t=[int(x) for x in ln.split(", ")]
        if (t-u)/1024>free: best,free=i,(t-u)/1024
    print(f"[diag] 选用 GPU{best}，空闲 {free:.0f} GB")
    return best
from hypergraph.scene_registry import get_scene_model
from models.integral.train_nn import train
from models.integral.train import (build_orderwise_support, compute_auc_scores,
                                   macro_auc, infer_n_nodes_from_edge_config)
from models.integral.train_nn import infer_max_order_from_edge_config

HG, spec = get_scene_model("ecological")
d = HG.get_default_params()
cfg = HG.get_hyperedge_config(d["n_nodes"], d["max_order"])
N = infer_n_nodes_from_edge_config(cfg) or d["n_nodes"]
mo = infer_max_order_from_edge_config(cfg) or d["max_order"]
ec = HG.get_hyperedge_config(N, mo)
sup = build_orderwise_support(N, mo)
W_INIT = -1.5
W, _ = train(HG, spec.label, N, mo, ec, sup,
             n_samples=300, noise=0.02, n_epochs=8000, lr=1e-3, seed=11,
             stage1_epochs=1500, physics_ramp_epochs=1000, eval_every=0,
             results_root="/tmp/claude-0/diagrun", run_tag="diag", gpu_id=pick_gpu(), chunk_size=2000)
w = W.reshape(-1); moved = np.abs(w - W_INIT)
print("\n================ W 诊断 ================")
print(f"候选总数          {w.size}")
print(f"|W-init| 中位数    {np.median(moved):.4f}")
print(f"几乎没动 (<0.05)   {int((moved<0.05).sum())}  ({100*(moved<0.05).mean():.1f}%)")
print(f"明显移动 (>0.5)    {int((moved>0.5).sum())}")
print(f"W 范围            [{w.min():.3f}, {w.max():.3f}]")
print(f"W>0 的个数         {int((w>0).sum())}")
CLIP=2.0
at_floor=int(np.isclose(w,-CLIP,atol=1e-6).sum())
print(f"\n--- 裁剪边界 ---")
print(f"恰好压在 -{CLIP} 上  {at_floor}  ({100*at_floor/w.size:.1f}%)  <- 这些候选分数完全相同，排序时是平局")
uniq=len(np.unique(np.round(w,6)))
print(f"不同取值个数        {uniq} / {w.size}")
s, _ = compute_auc_scores(W, ec, N, mo, sup)
print("AUC:", {k: round(v,4) for k,v in s.items() if v is not None}, "macro=", round(macro_auc(s),4))

# 真实超边落在哪里：被裁剪压平的里面有多少是真边？
from models.integral.train import compute_auc_scores as _c
import itertools
ORDER_KEYS=["edges","triangles","quads","quints"]
truth={k:set(tuple(sorted(e)) for e in ec.get(k,[])) for k in ORDER_KEYS}
off=0
print("\n--- 真边 vs 裁剪边界 ---")
for k in ORDER_KEYS:
    cands=sup.get(k,[])
    if not cands: continue
    n=len(cands)
    seg=w[off:off+n]; off+=n
    lab=np.array([1 if tuple(sorted(e)) in truth[k] else 0 for e in cands])
    floor=np.isclose(seg,-CLIP,atol=1e-6)
    print(f"{k}: 候选 {n}  真边 {int(lab.sum())}  "
          f"压在边界 {int(floor.sum())}  其中真边 {int((floor&(lab==1)).sum())}  "
          f"真边分数中位数 {np.median(seg[lab==1]):+.3f}  负样本中位数 {np.median(seg[lab==0]):+.3f}")
true_idx = set()
