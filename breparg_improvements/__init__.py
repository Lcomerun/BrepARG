"""
breparg_improvements
=====================
针对 BrepARG(CVPR 2026, ../BrepARG/)的三个改进方案的 drop-in 实现。

每个模块都设计为**不改动 BrepARG 上游核心逻辑**即可替换对应组件:

  方案① fsq_quantise.FSQQuantiser
      drop-in 替换 BrepARG/quantise.py::VectorQuantiser(用于 trainer.py 的 VQVAE)。

  方案② gnn_ordering.rcm_face_ordering / gnn_face_ordering
      drop-in 替换 BrepARG/2sequence.py::dfs_face_ordering_from_core(离线预处理面排序)。

  方案③ constrained_decoding.TopologyConstrainedLogitsProcessor
      作为 HuggingFace LogitsProcessor 注入 BrepARG/generate_brep.py 的 model.generate(),
      无需重训。

详见同目录 README.md 与 方案.md;测试见 test_all.py(`python test_all.py`,58 项全过才可提交)。
"""

from .fsq_quantise import FSQ, FSQQuantiser
from .gnn_ordering import (
    rcm_face_ordering,
    gnn_face_ordering,
    GNNFaceOrdering,
    train_gnn_ordering,
)
from .constrained_decoding import (
    BrepVocab,
    TopologyConstrainedLogitsProcessor,
)

__all__ = [
    "FSQ", "FSQQuantiser",
    "rcm_face_ordering", "gnn_face_ordering", "GNNFaceOrdering", "train_gnn_ordering",
    "BrepVocab", "TopologyConstrainedLogitsProcessor",
]
