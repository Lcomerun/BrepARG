"""
fsq_quantise.py
================
方案①:用 Finite Scalar Quantization (FSQ) 替换 BrepARG 的朴素 VQ-VAE 量化器。

设计目标:作为 `quantise.py::VectorQuantiser` 的 **drop-in 替换**,保持完全相同的
forward 接口,使 trainer / 2sequence / generate 三处代码几乎无需改动即可切换。

原始 VectorQuantiser 接口(必须严格匹配):
    输入  z:  (B, C, H, W)        # BrepARG 中 C=64(quant_conv 之后),H=W=2
    输出  (z_q, loss, (perplexity, min_encodings, encoding_indices))
        z_q:             (B, C, H, W)  与输入同形状
        loss:            标量 tensor(FSQ 无 commitment loss,返回 0)
        encoding_indices: 展平为 (B*H*W,) 的 long tensor,token id ∈ [0, codebook_size)
                          展平顺序为 (b, h, w) row-major,与原 VQ 一致(reshape(N,4) 时对齐)

FSQ 核心思想(Mentzer et al., "Finite Scalar Quantization: VQ-VAE Made Simple", 2023):
    - 不使用可学习 codebook,直接对每个潜在维度做有限标量量化;
    - 有效 codebook 大小 = prod(levels),是隐式的,无需存储;
    - 无 codebook collapse、无 commitment/codebook loss、无最近邻搜索;
    - 与 BrepARG 论文中 position 用标量量化的成功经验逻辑自洽。

参考实现遵循 Mentzer 原论文与 lucidrains/vector-quantize-pytorch 的规范版本,
并补充了 (a) 64→d 投影/反投影以适配 BrepARG 的 64 通道瓶颈,
        (b) 与原 VQ 完全一致的返回签名。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def round_ste(z: torch.Tensor) -> torch.Tensor:
    """Round with straight-through gradient estimator."""
    return z + (z.round() - z).detach()


class FSQ(nn.Module):
    """
    纯 FSQ 量化核心(对一个 (..., d) 的连续向量做量化)。
    levels: list[int],每个维度的量化级数,例如 [8,8,8,8] -> 有效 codebook 4096。
    """

    def __init__(self, levels):
        super().__init__()
        _levels = torch.tensor(levels, dtype=torch.int64)
        self.register_buffer("_levels", _levels, persistent=False)

        # 进制基:用于把多维离散坐标打包成单个整数 token id
        _basis = torch.cumprod(
            torch.tensor([1] + list(levels[:-1]), dtype=torch.int64), dim=0
        )
        self.register_buffer("_basis", _basis, persistent=False)

        self.dim = len(levels)
        self.codebook_size = int(torch.prod(_levels).item())

    def bound(self, z: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        """
        把任意实数 z 软约束到每个维度的合法范围。
        对偶数 level 做 0.5 偏移,保证 round 之后落在 0..L-1 的整数网格(规范做法)。
        """
        levels = self._levels.to(z.device)
        half_l = (levels - 1) * (1 - eps) / 2
        offset = torch.where(levels % 2 == 0, torch.tensor(0.5, device=z.device),
                             torch.tensor(0.0, device=z.device))
        shift = torch.tan(offset / half_l)  # 使 tanh 平移后对齐网格
        return torch.tanh(z + shift) * half_l - offset

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """量化到 [-1, 1] 归一化的码值(可微,straight-through)。"""
        quantized = round_ste(self.bound(z))
        half_width = (self._levels.to(z.device) // 2).float()
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: torch.Tensor) -> torch.Tensor:
        half_width = (self._levels.to(zhat_normalized.device) // 2).float()
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: torch.Tensor) -> torch.Tensor:
        half_width = (self._levels.to(zhat.device) // 2).float()
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat_normalized: torch.Tensor) -> torch.Tensor:
        """归一化码值 (..., d) -> 单个整数 token id (...,)。"""
        zhat = self._scale_and_shift(zhat_normalized)
        basis = self._basis.to(zhat.device).float()
        return (zhat * basis).sum(dim=-1).round().long()

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """token id (...,) -> 归一化码值 (..., d),供 decoder 使用。"""
        indices = indices.unsqueeze(-1)
        levels = self._levels.to(indices.device)
        basis = self._basis.to(indices.device)
        codes_non_norm = (indices // basis) % levels
        return self._scale_and_shift_inverse(codes_non_norm.float())

    def forward(self, z: torch.Tensor):
        """z: (..., d) -> (codes_normalized (..., d), indices (...,))"""
        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)
        return codes, indices


class FSQQuantiser(nn.Module):
    """
    BrepARG drop-in 替换:与 quantise.py::VectorQuantiser 接口完全一致。

    构造参数保持与 VectorQuantiser 类似(num_embed/embed_dim/beta 等),以便最小改动;
    但额外引入 fsq_levels 决定 FSQ 的级数配置。

    注意:
      - num_embed(有效 codebook)应等于 prod(fsq_levels),否则下游 se_codebook_size
        必须改成 prod(fsq_levels)。构造时会断言一致。
      - in_dim 为量化器输入通道(BrepARG 中为 64,即 quant_conv 输出)。
    """

    def __init__(self, num_embed, embed_dim, beta=0.25,
                 fsq_levels=(8, 8, 8, 8), in_dim=None, **kwargs):
        super().__init__()
        self.fsq = FSQ(list(fsq_levels))
        self.num_embed = self.fsq.codebook_size
        # 容许 num_embed 传入用于校验
        if num_embed is not None:
            assert num_embed == self.fsq.codebook_size, (
                f"传入 num_embed={num_embed} 与 prod(fsq_levels)={self.fsq.codebook_size} 不一致;"
                f"请把下游 se_codebook_size 设为 {self.fsq.codebook_size},或调整 fsq_levels。"
            )
        self.embed_dim = embed_dim          # 保留字段以兼容外部读取
        self.beta = beta

        d = self.fsq.dim
        in_dim = in_dim if in_dim is not None else embed_dim
        self.in_dim = in_dim
        # 64 -> d 投影 与 d -> 64 反投影(1x1 conv,保持 (B,C,H,W) 语义)
        self.proj_in = nn.Conv2d(in_dim, d, kernel_size=1)
        self.proj_out = nn.Conv2d(d, in_dim, kernel_size=1)

        # 兼容字段:某些外部代码会读 self.embedding.num_embeddings / weight
        # 提供一个只读占位 Embedding(不参与计算),避免 AttributeError
        self.embedding = nn.Embedding(self.fsq.codebook_size, d)
        self.embedding.weight.requires_grad_(False)

    @torch.no_grad()
    def _update_placeholder_embedding(self):
        """把 FSQ 的隐式码本物化进占位 embedding(仅供需要 .embedding.weight 的外部代码)。"""
        ids = torch.arange(self.fsq.codebook_size, device=self.embedding.weight.device)
        codes = self.fsq.indices_to_codes(ids)  # (K, d)
        self.embedding.weight.data.copy_(codes)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits is False, "Only for interface compatible with Gumbel"
        assert return_logits is False, "Only for interface compatible with Gumbel"

        B, C, H, W = z.shape
        assert C == self.in_dim, f"输入通道 {C} 必须等于 in_dim {self.in_dim}"

        # 64 -> d
        zd = self.proj_in(z)                       # (B, d, H, W)
        # 转成 (B, H, W, d) 做逐位置 FSQ
        zd_hwd = rearrange(zd, 'b d h w -> b h w d').contiguous()
        # 关键(AMP 安全):FSQ 的 round/进制打包必须用 fp32。
        # fp16 mantissa 仅能精确表示 <=2048 的整数,而 codebook 可达 8192,
        # autocast 下会让 codes_to_indices 越界 -> one_hot device-side assert。
        # 故在此显式关闭 autocast 并转 fp32 计算量化与 token id。
        with torch.cuda.amp.autocast(enabled=False):
            codes, indices = self.fsq(zd_hwd.float())   # codes:(B,H,W,d) indices:(B,H,W)
        codes = codes.to(zd.dtype)

        # 反投影回 64 通道
        codes_dchw = rearrange(codes, 'b h w d -> b d h w').contiguous()
        z_q = self.proj_out(codes_dchw)            # (B, C, H, W)

        # FSQ 无 commitment / codebook loss
        loss = torch.zeros((), device=z.device, dtype=z.dtype)

        # encoding_indices: 展平为 (B*H*W,) 且顺序为 (b,h,w),与原 VQ 一致
        # 防御性 clamp:保证落在 [0, codebook_size) 内(避免任何数值边界导致 one_hot 越界)。
        encoding_indices = indices.reshape(-1).long().clamp_(0, self.fsq.codebook_size - 1)

        # perplexity(基于本 batch 的 token 使用直方图)
        with torch.no_grad():
            onehot = F.one_hot(encoding_indices, num_classes=self.fsq.codebook_size).float()
            avg_probs = onehot.mean(dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # min_encodings 在 BrepARG 下游未被使用(只用 indices[2]),返回 None 占位
        min_encodings = None
        return z_q, loss, (perplexity, min_encodings, encoding_indices)


# ============================================================================
#  集成说明(写进 trainer.py / quantise.py 时按此修改)
# ============================================================================
#
# 在 quantise.py 末尾 import 本文件的 FSQQuantiser,然后在 trainer.py 的
# VQVAE.__init__ 中,把:
#
#     self.quantize = VectorQuantiser(num_embed=old_quant.n_e,
#                                     embed_dim=old_quant.vq_embed_dim, ...)
# 替换为:
#     from fsq_quantise import FSQQuantiser
#     self.quantize = FSQQuantiser(
#         num_embed=4096,            # 必须 == prod(fsq_levels)
#         embed_dim=old_quant.vq_embed_dim,   # 64
#         fsq_levels=(8, 8, 8, 8),   # -> 4096
#         in_dim=old_quant.vq_embed_dim,      # 64,即 quant_conv 输出通道
#     )
#
# 不要再 copy old codebook 权重(FSQ 无 codebook)。
# load_se_vqvae_model(utils.py) 中的 VQVAE 同理替换。
# 由于 prod(levels) 决定有效词表,config.json 的 se_codebook_size 也要相应设为该值。
