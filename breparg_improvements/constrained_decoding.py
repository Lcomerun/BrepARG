"""
constrained_decoding.py
=======================
方案③:推理期拓扑/语法约束解码。把 Valid 从 87.6% 往上推,且无需重训模型。

集成目标:generate_brep.py::generate_sequence 中的 model.generate(...) 调用,
        额外传入 logits_processor=LogitsProcessorList([TopologyConstrainedLogitsProcessor(...)])

BrepARG 的序列文法(严格复刻 2sequence.py 的拼装顺序):
    seq = <prompt> face_block* SEP edge_block* END
    face_block = [bbox×6][geo×4][face_idx×1]      # 11 tokens(见 2sequence L347-354)
    edge_block = [face_idx×2][bbox×6][geo×4]      # 12 tokens(见 2sequence L359-370)

token 类型按 vocab 区间(见 2sequence L214-225):
    face_index : [0,                       face_index_size)
    geometry   : [face_index_size,          face_index_size+K)
    bbox/pos   : [face_index_size+K,        face_index_size+K+L)
    START/SEP/END/PAD : 之后的 4 个特殊 id

施加的约束(均经 GT-safety 验证:真实合法序列在任一步都不会被屏蔽):
  C1 类型约束    : 块内每个槽位只允许对应类型的 token。
  C2 标记时序    : START 后必须进入 face 块;SEP 只能在完整 face 块边界;
                  END 只能在完整 edge 块边界且 SEP 已出现。
  C3 面索引声明  : edge 块开头的 2 个 face_idx 必须是 face 段已声明过的面(拓扑核心约束)。
  C4 面索引相异  : 同一条 edge 的两个 face_idx 必须不同。
  C5 面索引唯一  : face 段中每个面的 face_idx 不重复(face_index_map 是双射,GT 天然满足)。
  C6 bbox 单调   : (可选,方案①协同)同一 bbox 内 max 坐标 token ≥ min 坐标 token。
                  因 quantize_bbox 单调,GT 天然满足(允许相等以容纳退化 bbox)。

设计选择:无状态解析。每步从完整前缀重新解析状态,避免 KV-cache / beam 重排带来的
状态错乱。面图规模小、序列不长,O(seq_len) 重解析开销可忽略。
"""

from transformers import LogitsProcessor
import torch


class BrepVocab:
    """根据 BrepARG 配置计算各 token 段的区间与特殊 token id。"""

    def __init__(self, face_index_size, se_codebook_size, bbox_index_size):
        self.face_index_size = face_index_size
        self.se_codebook_size = se_codebook_size
        self.bbox_index_size = bbox_index_size

        self.face_index_offset = 0
        self.se_token_offset = self.face_index_offset + face_index_size
        self.bbox_token_offset = self.se_token_offset + se_codebook_size
        special_offset = self.bbox_token_offset + bbox_index_size
        self.START_TOKEN = special_offset
        self.SEP_TOKEN = special_offset + 1
        self.END_TOKEN = special_offset + 2
        self.PAD_TOKEN = special_offset + 3
        self.vocab_size = face_index_size + se_codebook_size + bbox_index_size + 4

    # --- 类型判定 ---
    def is_face_idx(self, t):
        return self.face_index_offset <= t < self.se_token_offset

    def is_geo(self, t):
        return self.se_token_offset <= t < self.bbox_token_offset

    def is_bbox(self, t):
        return self.bbox_token_offset <= t < self.START_TOKEN

    def face_idx_value(self, t):
        return t - self.face_index_offset

    def bbox_raw(self, t):
        """bbox token 的原始量化索引(用于单调性比较)。"""
        return t - self.bbox_token_offset


# 块布局常量
FACE_BLOCK = ['bbox'] * 6 + ['geo'] * 4 + ['idx'] * 1       # len 11
EDGE_BLOCK = ['idx'] * 2 + ['bbox'] * 6 + ['geo'] * 4        # len 12
FACE_LEN = len(FACE_BLOCK)   # 11
EDGE_LEN = len(EDGE_BLOCK)   # 12


_FULL_MASK_FALLBACKS = {'count': 0}


def _warn_full_mask_fallback(st):
    _FULL_MASK_FALLBACKS['count'] += 1
    if _FULL_MASK_FALLBACKS['count'] <= 20:
        import warnings
        warnings.warn(
            "[constrained_decoding] 全屏蔽兜底触发(强插 END):"
            f"section={st.get('section')!r} slot={st.get('slot_in_block')!r} "
            f"declared_faces={len(st.get('declared_faces') or [])} "
            f"(累计第 {_FULL_MASK_FALLBACKS['count']} 次)")


def parse_state(seq, V: BrepVocab, prompt_len=1):
    """
    走一遍前缀,返回当前"对下一个 token 的期望"与相关约束上下文。
    seq: list[int],包含 prompt 与已生成 token。
    返回 dict:
        expect: 'bbox'|'geo'|'idx'|'sep_or_face_bbox'|'edge_or_end_idx'|'after_sep_idx'|'done'|'invalid'
        section: 'face'|'edge'|'pre'|'post'
        declared_faces: set[int]      face 段已声明的面索引值
        used_faces_section: set[int]  当前段内已用的 face_idx 值(face 段用于唯一性)
        edge_first_face: int|None     当前 edge 块第一个 face_idx 值(用于相异约束)
        block_bbox_raw: list[int]     当前块内已收集的 bbox 原始索引(用于单调性)
        slot_in_block: int            当前块内已填槽位数
    """
    st = {
        'expect': 'bbox', 'section': 'face',
        'declared_faces': set(), 'used_faces_section': set(),
        'edge_first_face': None, 'block_bbox_raw': [], 'slot_in_block': 0,
    }
    if len(seq) <= prompt_len:
        # 仅 prompt:期望 face 段第一个 bbox
        st['expect'] = 'bbox'; st['section'] = 'face'; st['slot_in_block'] = 0
        return st

    body = seq[prompt_len:]
    section = 'face'
    declared = set()
    used_section = set()
    # face 块状态
    fpos = 0
    cur_block_bbox = []
    edge_first = None
    seen_sep = False
    seen_end = False

    for t in body:
        if seen_end:
            st['expect'] = 'done'; st['section'] = 'post'
            return st

        if section == 'face':
            if fpos == 0 and t == V.SEP_TOKEN:
                # 完整 face 块边界处出现 SEP -> 进入 edge 段
                section = 'edge'; seen_sep = True
                fpos = 0; cur_block_bbox = []; edge_first = None
                used_section = set()
                continue
            slot = FACE_BLOCK[fpos]
            if slot == 'bbox':
                cur_block_bbox.append(V.bbox_raw(t))
            elif slot == 'idx':
                declared.add(V.face_idx_value(t))
                used_section.add(V.face_idx_value(t))
            fpos += 1
            if fpos == FACE_LEN:
                fpos = 0; cur_block_bbox = []
        else:  # edge section
            if fpos == 0 and t == V.END_TOKEN:
                seen_end = True
                continue
            slot = EDGE_BLOCK[fpos]
            if slot == 'idx':
                if fpos == 0:
                    edge_first = V.face_idx_value(t)
                # fpos==1: 第二个 idx,块结束后清空
            elif slot == 'bbox':
                cur_block_bbox.append(V.bbox_raw(t))
            fpos += 1
            if fpos == EDGE_LEN:
                fpos = 0; cur_block_bbox = []; edge_first = None

    # 走完前缀,确定下一槽位期望
    st['declared_faces'] = declared
    st['used_faces_section'] = used_section
    st['block_bbox_raw'] = cur_block_bbox
    st['edge_first_face'] = edge_first
    st['slot_in_block'] = fpos

    if seen_end:
        st['expect'] = 'done'; st['section'] = 'post'
        return st

    if section == 'face':
        st['section'] = 'face'
        if fpos == 0:
            st['expect'] = 'sep_or_face_bbox'   # 可开新 face 块(bbox)或 SEP
        else:
            st['expect'] = FACE_BLOCK[fpos]       # bbox / geo / idx
    else:
        st['section'] = 'edge'
        if fpos == 0:
            # SEP 刚出现或一个完整 edge 块后:可开新 edge 块(idx)或(块边界)END
            st['expect'] = 'edge_or_end_idx' if seen_sep else 'invalid'
        else:
            st['expect'] = EDGE_BLOCK[fpos]       # idx / bbox / geo
    return st


class TopologyConstrainedLogitsProcessor(LogitsProcessor):
    """
    HuggingFace LogitsProcessor:在每步把非法 token 的 logit 置 -inf。
    use_bbox_monotonic:是否启用 C6(建议在使用方案①确定性量化时开启)。
    """

    def __init__(self, vocab: BrepVocab, prompt_len=1,
                 use_bbox_monotonic=False, enforce_face_unique=True,
                 min_faces=1):
        self.V = vocab
        self.prompt_len = prompt_len
        self.use_bbox_monotonic = use_bbox_monotonic
        self.enforce_face_unique = enforce_face_unique
        self.min_faces = min_faces

        V = vocab
        vs = V.vocab_size
        # 预构建各类型的布尔掩码(True=该 token 属于此类型)
        idx = torch.arange(vs)
        self.m_face_idx = (idx >= V.face_index_offset) & (idx < V.se_token_offset)
        self.m_geo = (idx >= V.se_token_offset) & (idx < V.bbox_token_offset)
        self.m_bbox = (idx >= V.bbox_token_offset) & (idx < V.START_TOKEN)
        self.m_sep = torch.zeros(vs, dtype=torch.bool); self.m_sep[V.SEP_TOKEN] = True
        self.m_end = torch.zeros(vs, dtype=torch.bool); self.m_end[V.END_TOKEN] = True

    def _allowed_mask(self, seq, device):
        V = self.V
        st = parse_state(seq, V, prompt_len=self.prompt_len)
        expect = st['expect']
        allowed = torch.zeros(V.vocab_size, dtype=torch.bool, device=device)

        def add(mask):
            allowed.__ior__(mask.to(device))

        if expect == 'bbox':
            add(self.m_bbox)
            self._apply_bbox_monotonic(allowed, st)
        elif expect == 'geo':
            add(self.m_geo)
        elif expect == 'idx':
            # face 段的 face_idx 槽:允许所有 face_idx,可选排除已用(唯一性)
            self._add_face_idx(allowed, st, role='face_declare', device=device)
        elif expect == 'sep_or_face_bbox':
            exhausted = self.enforce_face_unique and \
                len(st['used_faces_section']) >= V.face_index_size
            if not exhausted:            # 面索引未用尽才允许开新 face 块
                add(self.m_bbox)
                self._apply_bbox_monotonic(allowed, st)
            # 是否允许 SEP:已声明面数 >= min_faces
            if len(st['declared_faces']) >= self.min_faces:
                add(self.m_sep)
        elif expect == 'after_sep_idx':
            self._add_face_idx(allowed, st, role='edge_first', device=device)
        elif expect == 'edge_or_end_idx':
            # 可开新 edge 块(第一个 idx,须为已声明面)或(块边界)END。
            # 已声明面 < 2 时开 edge 块必然在第二槽因相异约束全屏蔽(死锁),
            # 此时只放行 END,避免兜底在 edge 块中间强插 END 产出畸形序列。
            if len(st['declared_faces']) >= 2:
                self._add_face_idx(allowed, st, role='edge_first', device=device)
            add(self.m_end)
        elif expect == 'done':
            # 已 END,理论上不应再采样;允许 END/PAD 以防越界
            add(self.m_end)
        else:  # invalid -> 不约束(安全兜底,交给后处理)
            allowed[:] = True

        # edge 块第二个 idx(相异约束)由 _add_face_idx 内部处理;
        # 但 expect=='idx' 在 edge 段对应第 2 个槽,需要单独走 edge_second 分支:
        if st['section'] == 'edge' and expect == 'idx':
            allowed.zero_()
            self._add_face_idx(allowed, st, role='edge_second', device=device)

        # 永不允许:START / PAD(避免重复 START 或提前 PAD)
        allowed[V.START_TOKEN] = False
        allowed[V.PAD_TOKEN] = False

        # 兜底:若全被屏蔽(理论上不该发生),放开 END 以免死锁——但必须告警,
        # 否则状态机缺陷只表现为 Invalid 率上升且无法归因
        if not torch.any(allowed):
            _warn_full_mask_fallback(st)
            allowed[V.END_TOKEN] = True
        return allowed

    def _add_face_idx(self, allowed, st, role, device):
        V = self.V
        if role == 'face_declare':
            base = self.m_face_idx.clone()
            if self.enforce_face_unique:
                for v in st['used_faces_section']:
                    base[V.face_index_offset + v] = False
            allowed.__ior__(base.to(device))
        elif role in ('edge_first', 'edge_second'):
            declared = st['declared_faces']
            if len(declared) == 0:
                # 没有已声明面却要求 edge idx -> 异常,放开 face_idx 兜底
                allowed.__ior__(self.m_face_idx.to(device))
                return
            m = torch.zeros(V.vocab_size, dtype=torch.bool, device=device)
            for v in declared:
                m[V.face_index_offset + v] = True
            if role == 'edge_second' and st['edge_first_face'] is not None:
                m[V.face_index_offset + st['edge_first_face']] = False  # 相异
            allowed.__ior__(m)

    def _apply_bbox_monotonic(self, allowed, st):
        """C6:当前 bbox 槽是 max 坐标(slot 3/4/5)时,要求 token 原始索引 >= 对应 min。"""
        if not self.use_bbox_monotonic:
            return
        V = self.V
        section = st['section']
        block_layout = FACE_BLOCK if section == 'face' else EDGE_BLOCK
        slot = st['slot_in_block']
        # 找当前块内 bbox 槽的序号(第几个 bbox)
        bbox_slots = [i for i, s in enumerate(block_layout) if s == 'bbox']
        if slot not in bbox_slots:
            return
        bbox_ord = bbox_slots.index(slot)   # 0..5,对应 [xmin,ymin,zmin,xmax,ymax,zmax]
        if bbox_ord < 3:
            return  # min 坐标无约束
        min_ord = bbox_ord - 3
        collected = st['block_bbox_raw']
        if min_ord >= len(collected):
            return  # 还没收集到对应 min(异常),不约束
        min_raw = collected[min_ord]
        # 禁止 raw < min_raw 的 bbox token
        lo = V.bbox_token_offset
        for raw in range(0, min_raw):
            allowed[lo + raw] = False

    def __call__(self, input_ids, scores):
        device = scores.device
        batch = input_ids.shape[0]
        pad = self.V.PAD_TOKEN
        for b in range(batch):
            # 剥离 left-padding 的 PAD,修正 prompt_len 切片错位。
            # 注意:本处理器假设批内真实 prompt 等长(prompt_len 为全批共享定值)
            seq = [t for t in input_ids[b].tolist() if t != pad]
            mask = self._allowed_mask(seq, device)
            scores[b] = scores[b].masked_fill(~mask, float('-inf'))
        return scores


# ============================================================================
#  集成说明(写进 generate_brep.py)
# ============================================================================
#
# from transformers import LogitsProcessorList
# from constrained_decoding import BrepVocab, TopologyConstrainedLogitsProcessor
#
# V = BrepVocab(face_index_size=50, se_codebook_size=8192, bbox_index_size=2048)  # 8192=8*8*8*16,须与 2sequence.py 的 se_codebook_size 一致
# processor = TopologyConstrainedLogitsProcessor(
#     V, prompt_len=prompt.shape[-1],     # 通常为 1(单 START 或单类别 token)
#     use_bbox_monotonic=True,            # 仅在用方案① FSQ 时建议开启
# )
# generated = model.generate(..., logits_processor=LogitsProcessorList([processor]))
