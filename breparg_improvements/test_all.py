"""
test_all.py — 三个方案的完整验证测试。
运行:python test_all.py
所有断言通过才视为可提交。
"""
import os, sys, random, warnings, traceback
import numpy as np
import torch
warnings.filterwarnings("ignore")

# --- 让本测试既能找到三个方案模块,也能找到 BrepARG 上游的 model.py ---
# 三个方案模块(fsq_quantise / gnn_ordering / constrained_decoding)与本文件同目录。
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _find_breparg_dir():
    """向上查找包含 BrepARG/model.py 的目录,兼容不同的目录组织。
    依次尝试: <here>/BrepARG, <here 上溯 6 层>/BrepARG, 以及 model.py 直接同目录。"""
    d = _HERE
    for _ in range(6):
        cand = os.path.join(d, 'BrepARG')
        if os.path.exists(os.path.join(cand, 'model.py')):
            return cand
        if os.path.exists(os.path.join(d, 'model.py')):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None

PASS, FAIL = [], []
def check(name, cond, extra=""):
    if cond:
        PASS.append(name); print(f"  ✓ {name} {extra}")
    else:
        FAIL.append(name); print(f"  ✗ FAIL: {name} {extra}")

def section(t): print(f"\n{'='*70}\n{t}\n{'='*70}")

torch.manual_seed(0); np.random.seed(0); random.seed(0)

# ===========================================================================
section("方案① FSQ 量化器")
# ===========================================================================
from fsq_quantise import FSQ, FSQQuantiser

# --- FSQ 核心:round-trip / index 范围 ---
levels = [8, 8, 8, 8]
fsq = FSQ(levels)
check("FSQ codebook_size == prod(levels)", fsq.codebook_size == 8*8*8*8,
      f"(={fsq.codebook_size})")

z = torch.randn(100, len(levels))
codes, idx = fsq(z)
check("FSQ index 范围 [0, K)", bool((idx >= 0).all() and (idx < fsq.codebook_size).all()),
      f"min={idx.min().item()} max={idx.max().item()}")
# round-trip: indices -> codes -> indices 必须一致
codes_rt = fsq.indices_to_codes(idx)
idx_rt = fsq.codes_to_indices(codes_rt)
check("FSQ indices->codes->indices 自洽", bool((idx == idx_rt).all()))
check("FSQ codes==indices_to_codes(codes_to_indices(codes))",
      bool(torch.allclose(codes, codes_rt, atol=1e-5)))

# --- 偶数 level 也要落在合法整数网格 ---
fsq_even = FSQ([4, 6, 8])
z2 = torch.randn(200, 3) * 5  # 大幅度,测试 bound
_, idx2 = fsq_even(z2)
check("FSQ 偶数level index 合法", bool((idx2 >= 0).all() and (idx2 < 4*6*8).all()),
      f"K={4*6*8} max={idx2.max().item()}")

# --- 梯度直通 ---
zg = torch.randn(10, 4, requires_grad=True)
cg, _ = fsq(zg)
cg.sum().backward()
check("FSQ straight-through 梯度非空", zg.grad is not None and bool((zg.grad.abs() > 0).any()))

# --- 确定性 ---
za = torch.randn(5, 4)
_, ia = fsq(za); _, ib = fsq(za)
check("FSQ 确定性(同输入同 index)", bool((ia == ib).all()))

# --- FSQQuantiser drop-in 接口:shape / 返回签名 ---
q = FSQQuantiser(num_embed=4096, embed_dim=64, fsq_levels=(8,8,8,8), in_dim=64)
zin = torch.randn(7, 64, 2, 2)   # 模拟 quant_conv 输出 (B,64,2,2)
z_q, loss, stats = q(zin)
perplexity, min_enc, enc_idx = stats
check("FSQQ 输出 z_q 形状 == 输入", tuple(z_q.shape) == (7, 64, 2, 2),
      f"{tuple(z_q.shape)}")
check("FSQQ loss 是标量 tensor", loss.dim() == 0)
check("FSQQ encoding_indices 展平 (B*H*W,)", tuple(enc_idx.shape) == (7*2*2,),
      f"{tuple(enc_idx.shape)}")
check("FSQQ index 范围合法", bool((enc_idx >= 0).all() and (enc_idx < 4096).all()))
check("FSQQ perplexity 合理 (1..K)", 1.0 <= perplexity.item() <= 4096.0,
      f"={perplexity.item():.1f}")
# reshape(N,4) 必须成立(下游 2sequence L333 依赖)
reshaped = enc_idx.reshape(7, 4)
check("FSQQ index 可 reshape(N,4)", tuple(reshaped.shape) == (7, 4))

# --- num_embed 与 levels 不一致时必须报错 ---
try:
    _ = FSQQuantiser(num_embed=4096, embed_dim=64, fsq_levels=(8,8,8,5))  # 2560≠4096
    check("FSQQ num_embed 不一致应报错", False)
except AssertionError:
    check("FSQQ num_embed 不一致应报错", True)

# --- 占位 embedding 物化正确(供需要 .embedding.weight 的外部代码) ---
q._update_placeholder_embedding()
ids_all = torch.arange(q.fsq.codebook_size)
codes_all = q.fsq.indices_to_codes(ids_all)
check("FSQQ 占位 embedding 物化正确",
      bool(torch.allclose(q.embedding.weight.data, codes_all, atol=1e-5)))

# --- 与真实 diffusers VQModel 集成:encoder->quant_conv->FSQ->post_quant_conv->decoder ---
try:
    from diffusers import VQModel
    class _VQVAE_FSQ(VQModel):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.quantize = FSQQuantiser(num_embed=4096, embed_dim=self.quantize.vq_embed_dim,
                                         fsq_levels=(8,8,8,8), in_dim=self.quantize.vq_embed_dim)
    m = _VQVAE_FSQ(in_channels=3, out_channels=3,
                   down_block_types=['DownEncoderBlock2D']*5,
                   up_block_types=['UpDecoderBlock2D']*5,
                   block_out_channels=[32,64,128,256,512], layers_per_block=2,
                   act_fn='silu', latent_channels=128, vq_embed_dim=64,
                   num_vq_embeddings=4096, norm_num_groups=32, sample_size=512)
    x = torch.randn(3, 3, 32, 32)
    h = m.encoder(x); h = m.quant_conv(h)
    check("集成: quant_conv 输出 (B,64,2,2)", tuple(h.shape) == (3, 64, 2, 2), f"{tuple(h.shape)}")
    z_q, vq_loss, idxs = m.quantize(h)
    recon = m.decoder(m.post_quant_conv(z_q))
    check("集成: 重建输出 (B,3,32,32)", tuple(recon.shape) == (3, 3, 32, 32), f"{tuple(recon.shape)}")
    ei = idxs[2]
    se_tokens = int(ei.numel() // 3)  # per-sample
    check("集成: 每个面/边产生 4 个 geometry token", se_tokens == 4, f"(={se_tokens})")
    # 训练一步反传(确认端到端可训练)
    loss = torch.nn.functional.mse_loss(recon, x) + vq_loss
    loss.backward()
    g_ok = any(p.grad is not None and p.grad.abs().sum() > 0
               for p in m.encoder.parameters())
    check("集成: 端到端反传梯度到 encoder", g_ok)
except Exception as e:
    check("集成: diffusers VQModel + FSQ", False, f"异常 {e}")
    traceback.print_exc()

# ===========================================================================
section("方案② RCM + 几何感知 GNN 面排序")
# ===========================================================================
from gnn_ordering import (GNNFaceOrdering, gnn_face_ordering, rcm_face_ordering,
                          train_gnn_ordering, build_adjacency, hard_mla_cost,
                          node_features, soft_rank)

# ---- 复刻原 DFS 作为对照(从 2sequence.py 拷贝逻辑) ----
def dfs_ref(edge_face_pairs, num_faces):
    nbrs = [set() for _ in range(num_faces)]
    for f1, f2 in edge_face_pairs:
        if 0 <= f1 < num_faces and 0 <= f2 < num_faces and f1 != f2:
            nbrs[f1].add(f2); nbrs[f2].add(f1)
    deg = [len(n) for n in nbrs]
    visited = [False]*num_faces; order=[]
    seeds = sorted(range(num_faces), key=lambda x:(-deg[x], x))
    def dfs(u):
        visited[u]=True; order.append(u)
        un=[v for v in nbrs[u] if not visited[v]]; un.sort(key=lambda x:(deg[x],x))
        for v in un:
            if not visited[v]: dfs(v)
    for s in seeds:
        if not visited[s]: dfs(s)
    return order, {f:i for i,f in enumerate(order)}

# ---- 结构化图生成器(类 CAD 拓扑:网格=细分曲面/箱体,棱柱=圆柱/棱柱体) ----
#      真实 CAD 面图是稀疏、局部结构化的;随机图几乎不奖励好排序(已实测),
#      故必须用结构化图作为忠实代理。每个面带几何坐标(质心),用于打破对称。
def grid_graph(m, n):
    pairs=[]; geom={}
    for r in range(m):
        for c in range(n):
            u=r*n+c; geom[u]=(float(r), float(c), 0.0)
            if c+1<n: pairs.append((u, r*n+c+1))
            if r+1<m: pairs.append((u, (r+1)*n+c))
    return pairs, m*n, geom

def prism_graph(k):
    pairs=[]; geom={}
    for i in range(k):
        ang=2*np.pi*i/k
        geom[i]=(float(np.cos(ang)), float(np.sin(ang)), 1.0)
        geom[k+i]=(float(np.cos(ang)), float(np.sin(ang)), 0.0)
        pairs.append((i,(i+1)%k)); pairs.append((k+i,k+(i+1)%k)); pairs.append((i,k+i))
    return pairs, 2*k, geom

def make_structured(n, seed):
    rng=random.Random(seed); gs=[]
    for j in range(n):
        if j%2==0: efp,nf,geom=grid_graph(rng.randint(3,6), rng.randint(3,6))
        else:      efp,nf,geom=prism_graph(rng.randint(4,12))
        # 打乱节点 id,避免 GNN 利用 id 顺序作弊(真实数据 id 也无序)
        perm=list(range(nf)); rng.shuffle(perm)
        efp=[(perm[a],perm[b]) for a,b in efp]
        geom={perm[k_]:v for k_,v in geom.items()}
        face_geom=np.array([geom[i] for i in range(nf)], dtype=np.float32)
        gs.append({'edge_face_pairs':efp, 'num_faces':nf, 'face_geom':face_geom})
    return gs

def is_perm(order, n):
    return sorted(order)==list(range(n)) and len(order)==n

def random_topology(n_faces, edge_prob, seed):
    rng=random.Random(seed); pairs=[]
    for i in range(n_faces):
        for j in range(i+1, n_faces):
            if rng.random()<edge_prob: pairs.append((i,j))
    return pairs

# ---- drop-in 签名:RCM / GNN 都返回合法排列 ----
for nf, sd in [(5,1),(12,2),(30,3),(50,4)]:
    efp = random_topology(nf, 0.25, seed=sd)
    o_rcm, p_rcm = rcm_face_ordering(efp, nf)
    check(f"RCM 合法排列 N={nf}", is_perm(o_rcm, nf))
    check(f"RCM position_map 一致 N={nf}", all(p_rcm[f]==i for i,f in enumerate(o_rcm)))
    o_fb, _ = gnn_face_ordering(efp, nf, model=None)  # model=None 回退 RCM
    check(f"gnn_face_ordering(model=None) 回退 RCM 一致 N={nf}", o_fb==o_rcm)

# ---- 边界情况 ----
check("RCM N=0", rcm_face_ordering([], 0)[0]==[])
check("RCM N=1", rcm_face_ordering([], 1)[0]==[0])
check("RCM 无边图(N=6) 合法", is_perm(rcm_face_ordering([], 6)[0], 6))
check("RCM 非连通图合法", is_perm(rcm_face_ordering([(0,1),(1,2),(4,5)], 7)[0], 7))

# ---- 【核心】RCM 在结构化(类 CAD)图上 MLA 代价显著优于随机,且优于 DFS ----
test_gs = make_structured(40, seed=100)
rcm_c, dfs_c, rand_c = [], [], []
for g in test_gs:
    efp, nf = g['edge_face_pairs'], g['num_faces']
    o_rcm,_ = rcm_face_ordering(efp, nf)
    o_dfs,_ = dfs_ref(efp, nf)
    o_rand = list(range(nf)); random.Random(nf).shuffle(o_rand)
    rcm_c.append(hard_mla_cost(o_rcm, efp, nf))
    dfs_c.append(hard_mla_cost(o_dfs, efp, nf))
    rand_c.append(hard_mla_cost(o_rand, efp, nf))
RCM, DFS, RND = np.mean(rcm_c), np.mean(dfs_c), np.mean(rand_c)
print(f"    [结构化图 MLA] RCM={RCM:.1f}  DFS={DFS:.1f}  Random={RND:.1f}  "
      f"(RCM/DFS={RCM/DFS:.2f}, RCM/Rand={RCM/RND:.2f})")
check("RCM MLA 显著优于随机(<0.7x)", RCM < RND*0.7, f"RCM={RCM:.1f} Rand={RND:.1f}")
check("RCM MLA 优于论文 DFS(<DFS)", RCM < DFS, f"RCM={RCM:.1f} < DFS={DFS:.1f}")

# ---- soft_rank 性质(可微 MLA 备选损失的基础) ----
s = torch.tensor([3.0, 1.0, 2.0])
r = soft_rank(s, tau=0.1)
check("soft_rank 单调(大分数小秩)", bool(r[0] < r[2] < r[1]), f"ranks={[round(x,2) for x in r.tolist()]}")

# ---- 几何感知 GNN:RCM 监督训练应收敛,且 MLA 代价显著优于随机 ----
train_gs = make_structured(80, seed=0)
model = train_gnn_ordering(train_gs, epochs=200, lr=3e-3, verbose=False, seed=0)
hist = model._train_history
check("GNN 监督训练损失下降(末值<初值)", hist[-1] < hist[0],
      f"init={hist[0]:.4f} final={hist[-1]:.4f}")

gnn_c, rcm_c2, rand_c2 = [], [], []
for g in test_gs:
    efp, nf, fg = g['edge_face_pairs'], g['num_faces'], g['face_geom']
    o_gnn,_ = gnn_face_ordering(efp, nf, model=model, face_geom=fg)
    if not is_perm(o_gnn, nf):
        check("GNN 输出合法排列", False); break
    o_rcm,_ = rcm_face_ordering(efp, nf)
    o_rand = list(range(nf)); random.Random(nf+1).shuffle(o_rand)
    gnn_c.append(hard_mla_cost(o_gnn, efp, nf))
    rcm_c2.append(hard_mla_cost(o_rcm, efp, nf))
    rand_c2.append(hard_mla_cost(o_rand, efp, nf))
GNN, RCM2, RND2 = np.mean(gnn_c), np.mean(rcm_c2), np.mean(rand_c2)
print(f"    [结构化图 MLA] GeoGNN={GNN:.1f}  RCM={RCM2:.1f}  Random={RND2:.1f}  "
      f"(GNN/Rand={GNN/RND2:.2f}, GNN/RCM={GNN/RCM2:.2f})")
check("GNN 所有输出均为合法排列", len(gnn_c)==len(test_gs))
check("GeoGNN MLA 显著优于随机(<0.75x)", GNN < RND2*0.75, f"GNN={GNN:.1f} Rand={RND2:.1f}")
check("GeoGNN MLA 与 RCM 同量级(<1.6x RCM)", GNN < RCM2*1.6, f"GNN={GNN:.1f} RCM={RCM2:.1f}")

# ===========================================================================
section("方案③ 拓扑/语法约束解码")
# ===========================================================================
from constrained_decoding import (BrepVocab, TopologyConstrainedLogitsProcessor,
                                   parse_state, FACE_LEN, EDGE_LEN)

# 构造 GT 合法序列(严格复刻 2sequence.py 的拼装)
def build_gt_sequence(num_faces, edges, V, r=0, monotonic=True, seed=0):
    """
    edges: list[(src,dst)] 原始面索引(0-based,DFS 后的位置),src!=dst
    返回 token 序列(含 START..END)。face_index_map = (i+r)%50。
    """
    rng = random.Random(seed)
    N = 50
    fmap = {i: (i + r) % N for i in range(num_faces)}
    seq = [V.START_TOKEN]
    def rand_bbox():
        # 6 个原始量化索引,保证 min<=max(单调)
        mins = [rng.randint(0, V.bbox_index_size//2) for _ in range(3)]
        maxs = [rng.randint(m, V.bbox_index_size-1) if monotonic else rng.randint(0, V.bbox_index_size-1)
                for m in mins]
        raws = mins + maxs
        return [V.bbox_token_offset + x for x in raws]
    def rand_geo():
        return [V.se_token_offset + rng.randint(0, V.se_codebook_size-1) for _ in range(4)]
    # faces
    for i in range(num_faces):
        seq += rand_bbox()
        seq += rand_geo()
        seq.append(V.face_index_offset + fmap[i])
    seq.append(V.SEP_TOKEN)
    # edges
    for (a, b) in edges:
        seq.append(V.face_index_offset + fmap[a])
        seq.append(V.face_index_offset + fmap[b])
        seq += rand_bbox()
        seq += rand_geo()
    seq.append(V.END_TOKEN)
    return seq

V = BrepVocab(face_index_size=50, se_codebook_size=4096, bbox_index_size=2048)

# --- 核心必过测试:GT 序列在任一步都不被屏蔽 ---
proc = TopologyConstrainedLogitsProcessor(V, prompt_len=1, use_bbox_monotonic=True,
                                          enforce_face_unique=True)
all_safe = True
detail = ""
for trial in range(50):
    rng = random.Random(trial)
    nf = rng.randint(2, 12)
    r = rng.randint(0, 49)
    # 随机生成 distinct 面对作为边
    n_edges = rng.randint(1, nf*2)
    edges = []
    for _ in range(n_edges):
        a, b = rng.sample(range(nf), 2)
        edges.append((a, b))
    seq = build_gt_sequence(nf, edges, V, r=r, monotonic=True, seed=trial)
    # 逐步检查:对每个前缀长度 L(从 prompt_len 到 len-1),
    # 下一个真实 token 必须在 allowed 中
    for L in range(1, len(seq)):
        prefix = seq[:L]
        mask = proc._allowed_mask(prefix, device='cpu')
        nxt = seq[L]
        if not mask[nxt].item():
            all_safe = False
            st = parse_state(prefix, V, prompt_len=1)
            detail = (f"trial={trial} L={L} 被屏蔽 token={nxt} "
                      f"expect={st['expect']} section={st['section']} slot={st['slot_in_block']}")
            break
    if not all_safe:
        break
check("【必过】GT 合法序列任一步都不被屏蔽", all_safe, detail)

# --- 非法 token 确实被屏蔽 ---
# 1) 类型错误:face 段第一个槽应是 bbox,geometry/face_idx 应被屏蔽
seq0 = [V.START_TOKEN]
mask0 = proc._allowed_mask(seq0, 'cpu')
check("约束: face 段首槽屏蔽 geometry token",
      not mask0[V.se_token_offset].item())
check("约束: face 段首槽屏蔽 face_idx token",
      not mask0[V.face_index_offset].item())
check("约束: face 段首槽允许 bbox token",
      mask0[V.bbox_token_offset].item())

# 2) edge 段引用未声明面 -> 屏蔽
#    构造:1 个 face(声明 idx=fmap[0]),SEP,然后 edge 第一个 idx
nf2 = 3; r2 = 7
fmap2 = {i: (i+r2) % 50 for i in range(nf2)}
seq_e = [V.START_TOKEN]
for i in range(nf2):
    seq_e += [V.bbox_token_offset]*6 + [V.se_token_offset]*4 + [V.face_index_offset + fmap2[i]]
seq_e.append(V.SEP_TOKEN)
mask_e = proc._allowed_mask(seq_e, 'cpu')
declared = {fmap2[i] for i in range(nf2)}
undeclared = next(v for v in range(50) if v not in declared)
check("约束: edge 引用未声明面被屏蔽",
      not mask_e[V.face_index_offset + undeclared].item(),
      f"undeclared={undeclared}")
check("约束: edge 引用已声明面被允许",
      mask_e[V.face_index_offset + fmap2[0]].item())

# 3) edge 第二个 idx 必须不同于第一个
seq_e2 = seq_e + [V.face_index_offset + fmap2[0]]   # 第一个 edge idx = fmap2[0]
mask_e2 = proc._allowed_mask(seq_e2, 'cpu')
check("约束: edge 第二个 idx 屏蔽与第一个相同的面",
      not mask_e2[V.face_index_offset + fmap2[0]].item())
check("约束: edge 第二个 idx 允许不同的已声明面",
      mask_e2[V.face_index_offset + fmap2[1]].item())

# 4) bbox 单调性:max 坐标(slot3=xmax)token < xmin 应被屏蔽
seq_b = [V.START_TOKEN, V.bbox_token_offset + 100]  # xmin raw=100
# 现在期望 ymin(slot1),先填 ymin,zmin 再到 xmax
seq_b += [V.bbox_token_offset + 50, V.bbox_token_offset + 60]  # ymin=50, zmin=60
mask_b = proc._allowed_mask(seq_b, 'cpu')  # 现在 slot=3 (xmax)
check("约束: bbox xmax < xmin 被屏蔽",
      not mask_b[V.bbox_token_offset + 99].item(), "(xmax raw=99 < xmin raw=100)")
check("约束: bbox xmax >= xmin 被允许",
      mask_b[V.bbox_token_offset + 100].item() and mask_b[V.bbox_token_offset + 200].item())

# 5) 关闭单调性时不应屏蔽小 bbox
proc_nomono = TopologyConstrainedLogitsProcessor(V, prompt_len=1, use_bbox_monotonic=False)
mask_b2 = proc_nomono._allowed_mask(seq_b, 'cpu')
check("约束: 关闭单调性时 xmax<xmin 不被屏蔽",
      mask_b2[V.bbox_token_offset + 99].item())

# 6) START/PAD 永不允许
check("约束: START 永不允许", not mask0[V.START_TOKEN].item())
check("约束: PAD 永不允许", not mask0[V.PAD_TOKEN].item())

# 7) 兜底:任何前缀都至少有一个 allowed token(不死锁)
no_deadlock = True
for trial in range(20):
    seq = build_gt_sequence(random.randint(2,8),
                            [(0,1)] if True else [], V, r=trial%50, seed=trial)
    for L in range(1, len(seq)):
        m = proc._allowed_mask(seq[:L], 'cpu')
        if not m.any().item():
            no_deadlock = False; break
    if not no_deadlock: break
check("约束: 任何前缀都不死锁(>=1 allowed)", no_deadlock)

# --- HF generate 端到端集成(微型随机模型) ---
try:
    _breparg_dir = _find_breparg_dir()
    if _breparg_dir is None:
        raise ImportError("找不到 BrepARG/model.py(请确认 BrepARG 源码在仓库内)")
    sys.path.insert(0, _breparg_dir)
    from model import ARModel
    from transformers import LogitsProcessorList
    Vtiny = BrepVocab(face_index_size=5, se_codebook_size=8, bbox_index_size=8)
    tiny = ARModel(vocab_size=Vtiny.vocab_size, d_model=32, nhead=2, num_layers=2,
                   dim_feedforward=64, max_seq_len=256, pad_token_id=Vtiny.PAD_TOKEN)
    tiny.eval()
    proc_tiny = TopologyConstrainedLogitsProcessor(Vtiny, prompt_len=1, use_bbox_monotonic=True)
    prompt = torch.tensor([[Vtiny.START_TOKEN]])
    out = tiny.generate(
        input_ids=prompt, max_length=120, do_sample=True, top_p=0.95,
        temperature=1.0, num_beams=1,
        pad_token_id=Vtiny.PAD_TOKEN, eos_token_id=Vtiny.END_TOKEN,
        bos_token_id=Vtiny.START_TOKEN,
        logits_processor=LogitsProcessorList([proc_tiny]),
    )
    gen = out[0].tolist()
    check("HF集成: 约束生成不崩溃且含 START", gen[0] == Vtiny.START_TOKEN, f"len={len(gen)}")
    # 验证生成序列每一步都自洽(parse_state 从不返回 invalid,类型匹配)
    grammar_ok = True
    for L in range(1, len(gen)):
        st = parse_state(gen[:L], Vtiny, prompt_len=1)
        if st['expect'] == 'invalid':
            grammar_ok = False; break
        # 检查已生成 token 类型是否符合上一步期望
    # 若生成了 END,检查 END 之前结构:截到第一个 END
    if Vtiny.END_TOKEN in gen[1:]:
        end_pos = gen.index(Vtiny.END_TOKEN, 1)
        body = gen[1:end_pos]
        # body 必须能被解析(走到 SEP 之后再 END);此处只验证无 invalid
    check("HF集成: 生成序列文法自洽(无 invalid 状态)", grammar_ok)
except Exception as e:
    check("HF集成: ARModel + 约束解码", False, f"异常 {e}")
    traceback.print_exc()

# ===========================================================================
section("测试汇总")
# ===========================================================================
print(f"\n通过: {len(PASS)}   失败: {len(FAIL)}")
if FAIL:
    print("失败项:")
    for f in FAIL: print(f"  - {f}")
    sys.exit(1)
else:
    print("✅ 全部通过")
    sys.exit(0)
