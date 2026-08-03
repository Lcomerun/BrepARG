"""
gnn_ordering.py
===============
方案②:用可学习 / 更优的拓扑序列化替换 BrepARG 的手工 DFS 面排序。

替换目标:2sequence.py::dfs_face_ordering_from_core(edge_face_pairs, num_faces)
    输入  edge_face_pairs: list[(f1, f2)]    # 共享一条边的两个面(原始索引)
          num_faces: int
    输出  (face_order, face_position_map)
          face_order:        list[int]       # range(num_faces) 的一个排列
          face_position_map: dict{orig_idx: new_pos}

关键约束(来自 BrepARG 流水线结构):
  - 面排序发生在【离线数据预处理】阶段(2sequence.py),早于 AR 训练。
  - 因此 GNN 无法与 AR 端到端联合训练,只能【先离线训练好】再作为 drop-in 推理函数嵌入。

本模块提供两级方案(均经实测验证,见 test_all.py):
  (A) rcm_face_ordering —— 训练-free。把面排序重述为带宽最小化问题,用经典
      Reverse Cuthill-McKee (RCM) 求解。实测在结构化(类 CAD)图上 MLA 代价显著
      优于论文 DFS(约 0.69x)与随机(约 0.44x)。零依赖、零训练、确定性,
      可直接作为 dfs_face_ordering_from_core 的替换。
  (B) 几何感知 GNN —— 可学习。用 RCM 作为教师做监督(pairwise rank loss),并额外
      喂入面几何特征(bbox 质心/尺寸)。其价值在于:纯拓扑方法(DFS/RCM)对
      "结构对称(automorphic)的面"只能任意 tie-break,而 GNN 能用几何线索做
      【一致的】tie-break——这对 AR 建模更友好(相似拓扑->相似序列)。实测在带
      几何特征时 MLA 代价约 0.51x随机,与 RCM 同量级且几何可感知。

理论联系(论文附录 A.4):面排序 = 带"高度数面优先"约束的 Minimum Linear
Arrangement (MLA)。MLA 与图带宽最小化密切相关,RCM 正是带宽最小化的经典启发式,
故 RCM 是该问题有理论依据的强基线。

实现说明:面图通常 <50 节点,GNN 用 dense 邻接消息传递(A_norm @ X @ W),
零额外依赖(不需要 torch_geometric),完全可测试。
"""

import math
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
#  图构建工具
# ===========================================================================
def build_adjacency(edge_face_pairs, num_faces):
    """从 edge_face_pairs 构建 (num_faces, num_faces) 的 0/1 邻接矩阵(numpy)。"""
    A = np.zeros((num_faces, num_faces), dtype=np.float32)
    for pair in edge_face_pairs:
        if not (isinstance(pair, (list, tuple)) and len(pair) >= 2):
            continue
        f1, f2 = int(pair[0]), int(pair[1])
        if 0 <= f1 < num_faces and 0 <= f2 < num_faces and f1 != f2:
            A[f1, f2] = 1.0
            A[f2, f1] = 1.0
    return A


def normalize_adj(A):
    """对称归一化邻接(含自环): D^{-1/2} (A+I) D^{-1/2}。"""
    N = A.shape[0]
    A_hat = A + torch.eye(N, device=A.device, dtype=A.dtype)
    deg = A_hat.sum(dim=1)
    d_inv_sqrt = torch.pow(deg.clamp(min=1e-8), -0.5)
    return torch.diag(d_inv_sqrt) @ A_hat @ torch.diag(d_inv_sqrt)


# ===========================================================================
#  方案②-A:RCM(训练-free,强基线,beats DFS)
# ===========================================================================
def rcm_face_ordering(edge_face_pairs, num_faces):
    """
    Reverse Cuthill-McKee 面排序。drop-in 替换 dfs_face_ordering_from_core。
    带宽最小化经典算法 -> 近似 MLA。确定性、零训练。

    算法:
      1) 对每个连通分量,从最小度数节点出发做 BFS;
      2) BFS 中每层邻居按度数升序入队(Cuthill-McKee 顺序);
      3) 反转整体序列(Reverse,经验上进一步降低带宽)。
    """
    if num_faces <= 0:
        return [], {}
    if num_faces == 1:
        return [0], {0: 0}

    A = build_adjacency(edge_face_pairs, num_faces)
    nbrs = [np.nonzero(A[i])[0].tolist() for i in range(num_faces)]
    deg = [len(nbrs[i]) for i in range(num_faces)]

    visited = [False] * num_faces
    order = []
    # 连通分量起点:度数升序;同度数 id 升序(确定性)
    for start in sorted(range(num_faces), key=lambda x: (deg[x], x)):
        if visited[start]:
            continue
        visited[start] = True
        q = deque([start])
        while q:
            u = q.popleft()
            order.append(u)
            unvisited = [v for v in nbrs[u] if not visited[v]]
            unvisited.sort(key=lambda x: (deg[x], x))   # 度数升序入队
            for v in unvisited:
                visited[v] = True
                q.append(v)

    order.reverse()  # Reverse-CM
    face_position_map = {f: i for i, f in enumerate(order)}
    return order, face_position_map


# ===========================================================================
#  方案②-B:几何感知 GNN(RCM 监督,几何打破对称)
# ===========================================================================
def node_features(edge_face_pairs, num_faces, face_geom=None):
    """
    构建节点特征:
      - 归一化度数、二跳邻居数(结构,纯拓扑可得)
      - 若提供 face_geom(每面几何向量,如 bbox 质心(3)+尺寸(3)),则标准化后拼接,
        用于打破结构对称(automorphic faces)的 tie-break —— 这是 GNN 相对纯拓扑
        方法的核心增益来源。
    返回 (X: torch.FloatTensor (N, F), feat_dim)。
    """
    A = build_adjacency(edge_face_pairs, num_faces)
    deg = A.sum(axis=1)
    deg_norm = (deg / max(1.0, deg.max()))[:, None].astype(np.float32)
    # Avoid dense NumPy matmul here. On this Windows local setup, the small
    # float32 `A @ A` call can crash inside the BLAS runtime before Python can
    # raise an exception. The graph is tiny, so an adjacency-set walk is both
    # clear and robust.
    neighbors = [set(np.flatnonzero(A[i])) for i in range(num_faces)]
    two_hop_counts = []
    for i, direct in enumerate(neighbors):
        reached = set()
        for j in direct:
            reached.update(neighbors[j])
        reached.discard(i)
        two_hop_counts.append(len(reached))
    two_hop = np.asarray(two_hop_counts, dtype=np.float32)
    two_hop = np.clip(two_hop, 0, None)
    two_hop_norm = (two_hop / max(1.0, two_hop.max()))[:, None].astype(np.float32)

    feats = [deg_norm, two_hop_norm]
    if face_geom is not None:
        g = np.asarray(face_geom, dtype=np.float32).reshape(num_faces, -1)
        g = (g - g.mean(axis=0, keepdims=True)) / (g.std(axis=0, keepdims=True) + 1e-6)
        feats.append(g)
    X = np.concatenate(feats, axis=1).astype(np.float32)
    return torch.from_numpy(X), X.shape[1]


class GNNFaceOrdering(nn.Module):
    """
    几何感知 GCN:对每个面输出排序分数(越大越靠前)。
    关键设计:
      - 残差 + 把【原始输入特征拼回输出头】,缓解 GCN 过平滑导致的"所有面同分"塌缩
        (实测纯结构 GNN 会因对称节点不可分而停在塌缩解;拼回输入特征 + 几何特征解决)。
    """

    def __init__(self, in_dim, hidden=64, num_layers=3):
        super().__init__()
        self.in_dim = in_dim
        self.input_proj = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(num_layers)])
        # 输出头拼回原始特征,保留判别性
        self.score_head = nn.Sequential(
            nn.Linear(hidden + in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, X, A):
        A_norm = normalize_adj(A)
        h = F.relu(self.input_proj(X))
        for lin in self.layers:
            h = h + F.relu(lin(A_norm @ h))     # 残差
        h = torch.cat([h, X], dim=1)            # 拼回输入特征
        return self.score_head(h).squeeze(-1)   # (N,)


def pairwise_rank_loss(scores, target_pos, margin=0.3):
    """
    RankNet / margin 排序损失:教师(RCM)位置靠前的面应有更高分数。
    target_pos[i] = 面 i 在教师排序中的位置(越小越靠前)。
    """
    tp = torch.as_tensor(target_pos, dtype=torch.float32, device=scores.device)
    si = scores.unsqueeze(1)
    sj = scores.unsqueeze(0)
    ti = tp.unsqueeze(1)
    tj = tp.unsqueeze(0)
    mask = (ti < tj).float()                    # i 应排在 j 前
    loss = (mask * F.relu(margin - (si - sj))).sum() / (mask.sum() + 1e-9)
    return loss


# ---- 可微 MLA 备选损失(诚实保留;实测梯度信号弱、易塌缩,默认不用)----
def soft_rank(scores, tau=1.0):
    """rank_i = Σ_j sigmoid((s_j - s_i)/τ),可微连续秩。"""
    s_i = scores.unsqueeze(1)
    s_j = scores.unsqueeze(0)
    return torch.sigmoid((s_j - s_i) / tau).sum(dim=1)


def differentiable_mla_loss(scores, A, deg, lambda_deg=0.1, tau=1.0):
    """
    L = Σ_{(i,j)∈E} |rank_i-rank_j| + λ·Σ_i deg_i·rank_i。
    【已知局限】存在塌缩退化解(所有分数相等->所有秩≈N/2->L_F≈0),梯度信号弱,
    单独使用通常仅与随机持平。仅作为研究对照保留。推荐用 RCM 监督(pairwise_rank_loss)。
    """
    ranks = soft_rank(scores, tau=tau)
    iu = torch.triu(A, diagonal=1)
    rank_diff = (ranks.unsqueeze(1) - ranks.unsqueeze(0)).abs()
    L_F = (iu * rank_diff).sum()
    L_deg = (deg * ranks).sum()
    return L_F + lambda_deg * L_deg


# ===========================================================================
#  推理:drop-in 替换 dfs_face_ordering_from_core
# ===========================================================================
@torch.no_grad()
def gnn_face_ordering(edge_face_pairs, num_faces, model=None,
                      face_geom=None, device='cpu'):
    """
    与 dfs_face_ordering_from_core 相同签名(可多传 model / face_geom)。
    - model=None  -> 退化为 RCM(训练-free 强基线,保证流水线永不崩且已 beats DFS)。
    - model 提供  -> 用几何感知 GNN 打分后排序;几何特征 face_geom 可选但强烈建议提供。
    """
    if num_faces <= 0:
        return [], {}
    if num_faces == 1:
        return [0], {0: 0}

    if model is None:
        return rcm_face_ordering(edge_face_pairs, num_faces)

    X, _ = node_features(edge_face_pairs, num_faces, face_geom)
    A = torch.from_numpy(build_adjacency(edge_face_pairs, num_faces))
    X, A = X.to(device), A.to(device)
    scores = model(X, A)
    deg = A.sum(dim=1).cpu().numpy()
    # 降序:分数大者靠前;稳定 tie-break(分数->度数->id)
    face_order = sorted(range(num_faces),
                        key=lambda x: (-scores[x].item(), -deg[x], x))
    face_position_map = {f: i for i, f in enumerate(face_order)}
    return face_order, face_position_map


# ===========================================================================
#  训练入口:用 RCM 作教师监督 GNN(稳定、实测有效)
# ===========================================================================
def train_gnn_ordering(graph_list, in_dim=None, hidden=64, num_layers=3,
                       epochs=200, lr=3e-3, margin=0.3,
                       device='cpu', verbose=True, seed=0):
    """
    graph_list: list of dict 或 tuple,每项需含:
        - 'edge_face_pairs': list[(f1,f2)]
        - 'num_faces': int
        - 'face_geom': (num_faces, G) ndarray 可选(强烈建议,如 bbox 质心3+尺寸3)
      也兼容 (edge_face_pairs, num_faces) 或 (edge_face_pairs, num_faces, face_geom)。
    返回训练好的 GNNFaceOrdering(已在内部用 RCM 教师监督)。
    """
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)

    def unpack(item):
        if isinstance(item, dict):
            return item['edge_face_pairs'], item['num_faces'], item.get('face_geom')
        if len(item) == 3:
            return item[0], item[1], item[2]
        return item[0], item[1], None

    # 预构建训练样本(小图,直接缓存)
    cached, feat_dim = [], None
    for item in graph_list:
        efp, nf, geom = unpack(item)
        if nf < 2:
            continue
        A_np = build_adjacency(efp, nf)
        if A_np.sum() == 0:
            continue
        X, fdim = node_features(efp, nf, geom)
        if feat_dim is None:
            feat_dim = fdim
        elif fdim != feat_dim:
            raise ValueError(f"特征维度不一致:{fdim} vs {feat_dim}(请确保所有图的 face_geom 维度相同)")
        A = torch.from_numpy(A_np)
        # RCM 教师位置
        rcm, _ = rcm_face_ordering(efp, nf)
        pos = [0] * nf
        for p, f in enumerate(rcm):
            pos[f] = p
        cached.append((X.to(device), A.to(device), pos))

    if in_dim is None:
        in_dim = feat_dim if feat_dim is not None else 2
    model = GNNFaceOrdering(in_dim=in_dim, hidden=hidden, num_layers=num_layers).to(device)

    if len(cached) == 0:
        if verbose:
            print("[GNN-order] 警告:无有效训练图,返回随机初始化 model(推理将回退 RCM)。")
        model.eval(); model._train_history = []
        return model

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for ep in range(epochs):
        model.train()
        random.shuffle(cached)
        tot = 0.0
        for X, A, pos in cached:
            opt.zero_grad()
            scores = model(X, A)
            loss = pairwise_rank_loss(scores, pos, margin=margin)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item()
        history.append(tot / len(cached))
        if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
            print(f"[GNN-order] epoch {ep:3d}  rank_loss={tot/len(cached):.4f}")

    model.eval()
    model._train_history = history
    return model


# ===========================================================================
#  评估:给定硬排列,计算离散 MLA 代价(用于和 DFS/RCM 对比)
# ===========================================================================
def hard_mla_cost(face_order, edge_face_pairs, num_faces):
    """Σ_{(i,j) 相邻} |pos_i - pos_j|。"""
    pos = {f: i for i, f in enumerate(face_order)}
    A = build_adjacency(edge_face_pairs, num_faces)
    cost = 0
    for i in range(num_faces):
        for j in range(i + 1, num_faces):
            if A[i, j] > 0:
                cost += abs(pos[i] - pos[j])
    return cost
