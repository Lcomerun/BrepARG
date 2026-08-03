"""
generate_validate.py
====================
用训练好的 AR 模型(+ FSQ-VQVAE 产生的词表)做**生成验证**:
  1) 从 START 自回归采样若干序列;
  2) 分别在「无约束」与「方案③ 约束解码」两种设置下生成;
  3) 用严格文法校验器统计 Valid%(语法/拓扑合法率),证明方案③把 Valid 往上推。

注意:这里的 Valid 是**序列层 (token-grammar) 合法率**(块结构正确、SEP/END 时序正确、
edge 的 face_idx 必须是已声明面且两面相异)。它是 OCC 重建 Valid 的必要前置;
OCC 网格重建需要 OCC+torch 同环境,当前 brepgen 环境暂不可同时满足(见 HANDOFF「下一步」)。

用法:
    CUDA_VISIBLE_DEVICES=1 python generate_validate.py --run newscheme_run1 --n 64 --max_new 400
输入:  /data/public/luol/breparg_data/<run>/{sequences_fsq_rcm.pkl, ar_best.pt}
输出:  repro_outputs/<run>/gen_report.json  (+ 控制台摘要)
"""
import os, sys, json, time, pickle, argparse
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)

def _find_breparg():
    d = _HERE
    for _ in range(6):
        if os.path.exists(os.path.join(d, 'BrepARG', 'model.py')):
            return os.path.join(d, 'BrepARG')
        p = os.path.dirname(d)
        if p == d: break
        d = p
    return None
BREPARG = _find_breparg(); assert BREPARG
sys.path.insert(0, BREPARG)

from transformers import LogitsProcessorList
from constrained_decoding import BrepVocab, TopologyConstrainedLogitsProcessor, FACE_BLOCK, EDGE_BLOCK, FACE_LEN, EDGE_LEN

DATA = os.environ.get('NS_OUTBASE', '/data/public/luol/breparg_data')  # /data 只读时设 /home 上的输出根


def validate_sequence(body, V: BrepVocab):
    """严格文法 + 拓扑校验。返回 (ok:bool, reason:str, n_faces:int, n_edges:int)。"""
    section = 'face'; fpos = 0
    declared = set(); used = set(); seen_sep = False
    edge_first = None; nfaces = 0; nedges = 0
    for i, t in enumerate(body):
        if t == V.PAD_TOKEN:
            return False, 'PAD before END', nfaces, nedges
        if section == 'face':
            if fpos == 0 and t == V.SEP_TOKEN:
                if nfaces < 1:
                    return False, 'SEP before any face', nfaces, nedges
                section = 'edge'; seen_sep = True; fpos = 0; used = set(); continue
            slot = FACE_BLOCK[fpos]
            if slot == 'bbox' and not V.is_bbox(t): return False, 'face bbox slot', nfaces, nedges
            if slot == 'geo' and not V.is_geo(t):   return False, 'face geo slot', nfaces, nedges
            if slot == 'idx':
                if not V.is_face_idx(t): return False, 'face idx slot', nfaces, nedges
                fv = V.face_idx_value(t)
                if fv in used: return False, 'dup face idx', nfaces, nedges
                declared.add(fv); used.add(fv); nfaces += 1
            fpos += 1
            if fpos == FACE_LEN: fpos = 0
        else:  # edge
            if fpos == 0 and t == V.END_TOKEN:
                rest = [x for x in body[i + 1:] if x != V.PAD_TOKEN]
                if rest: return False, 'tokens after END', nfaces, nedges
                if not seen_sep: return False, 'END before SEP', nfaces, nedges
                return True, 'ok', nfaces, nedges
            slot = EDGE_BLOCK[fpos]
            if slot == 'idx':
                if not V.is_face_idx(t): return False, 'edge idx slot', nfaces, nedges
                fv = V.face_idx_value(t)
                if fv not in declared: return False, 'edge idx undeclared', nfaces, nedges
                if fpos == 0: edge_first = fv
                elif fv == edge_first: return False, 'edge two faces equal', nfaces, nedges
            elif slot == 'bbox' and not V.is_bbox(t): return False, 'edge bbox slot', nfaces, nedges
            elif slot == 'geo' and not V.is_geo(t):   return False, 'edge geo slot', nfaces, nedges
            fpos += 1
            if fpos == EDGE_LEN: fpos = 0; edge_first = None; nedges += 1
    return False, 'truncated (no END)', nfaces, nedges


def load_ar(meta, ar_pt, device):
    from model import ARModel
    PAD = meta['special_tokens']['PAD_TOKEN']
    ck = torch.load(ar_pt, map_location=device)
    max_seq_len = int((ck.get('config') or {}).get('max_seq_len') or ck.get('max_seq_len') or 1024)
    ar = ARModel(vocab_size=meta['vocab_size'], d_model=ck.get('d_model', 256),
                 nhead=8, num_layers=ck.get('layers', 8), dim_feedforward=ck.get('d_model', 256) * 4,
                 dropout=0.1, max_seq_len=max_seq_len, pad_token_id=PAD).to(device).eval()
    ar.model.load_state_dict(ck['model_state_dict'])
    # GPT2 generate needs eos/bos set
    ar.config.eos_token_id = meta['special_tokens']['END_TOKEN']
    ar.config.bos_token_id = meta['special_tokens']['START_TOKEN']
    ar.config.pad_token_id = PAD
    return ar


def gen_batch(ar, V, n, max_new, device, constrained, temperature=0.7, top_p=0.9):
    START = V.START_TOKEN
    out = []
    bs = 16
    for s in range(0, n, bs):
        b = min(bs, n - s)
        ids = torch.full((b, 1), START, dtype=torch.long, device=device)
        att = torch.ones_like(ids)
        kw = dict(max_new_tokens=max_new, do_sample=True, temperature=temperature,
                  top_p=top_p, top_k=0, pad_token_id=V.PAD_TOKEN, eos_token_id=V.END_TOKEN)
        if constrained:
            proc = TopologyConstrainedLogitsProcessor(V, prompt_len=1, use_bbox_monotonic=True,
                                                      enforce_face_unique=True, min_faces=1)
            kw['logits_processor'] = LogitsProcessorList([proc])
        with torch.no_grad():
            g = ar.generate(input_ids=ids, attention_mask=att, **kw)
        for row in g.tolist():
            out.append(row[1:])  # strip START
    return out


def evaluate(seqs, V):
    res = [validate_sequence(s, V) for s in seqs]
    ok = [r for r in res if r[0]]
    reasons = {}
    for r in res:
        if not r[0]: reasons[r[1]] = reasons.get(r[1], 0) + 1
    nf = [r[2] for r in ok]; ne = [r[3] for r in ok]
    return {
        'n': len(seqs), 'valid': len(ok), 'valid_pct': round(100 * len(ok) / max(1, len(seqs)), 1),
        'avg_faces': round(float(np.mean(nf)), 2) if nf else 0,
        'avg_edges': round(float(np.mean(ne)), 2) if ne else 0,
        'fail_reasons': dict(sorted(reasons.items(), key=lambda x: -x[1])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='newscheme_run1')
    ap.add_argument('--n', type=int, default=64)
    ap.add_argument('--max_new', type=int, default=400)
    ap.add_argument('--temperature', type=float, default=0.7)
    ap.add_argument('--top_p', type=float, default=0.9)
    a = ap.parse_args()

    rundir = os.path.join(DATA, a.run)
    seq_pkl = os.path.join(rundir, 'sequences_fsq_rcm.pkl')
    ar_pt = os.path.join(rundir, 'ar_best.pt')
    assert os.path.exists(seq_pkl), seq_pkl
    assert os.path.exists(ar_pt), ar_pt
    meta = pickle.load(open(seq_pkl, 'rb'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    V = BrepVocab(face_index_size=meta['face_index_size'],
                  se_codebook_size=meta['se_codebook_size'],
                  bbox_index_size=meta['bbox_index_size'])
    print(f"[gen] vocab={V.vocab_size} START={V.START_TOKEN} device={device} n={a.n} max_new={a.max_new}")
    ar = load_ar(meta, ar_pt, device)

    t0 = time.time()
    print("[gen] sampling UNCONSTRAINED ...")
    free = gen_batch(ar, V, a.n, a.max_new, device, constrained=False,
                     temperature=a.temperature, top_p=a.top_p)
    print("[gen] sampling CONSTRAINED (方案③) ...")
    cons = gen_batch(ar, V, a.n, a.max_new, device, constrained=True,
                     temperature=a.temperature, top_p=a.top_p)
    ev_free = evaluate(free, V); ev_cons = evaluate(cons, V)
    mins = round((time.time() - t0) / 60, 1)

    report = {'run': a.run, 'n': a.n, 'max_new': a.max_new, 'temperature': a.temperature,
              'top_p': a.top_p, 'minutes': mins,
              'unconstrained': ev_free, 'constrained': ev_cons,
              'delta_valid_pct': round(ev_cons['valid_pct'] - ev_free['valid_pct'], 1)}
    evid = os.path.join(_HERE, 'repro_outputs', a.run); os.makedirs(evid, exist_ok=True)
    json.dump(report, open(os.path.join(evid, 'gen_report.json'), 'w'), ensure_ascii=False, indent=2)
    print("\n=== GENERATION VALIDATION ===")
    print(f"unconstrained : Valid {ev_free['valid_pct']}%  ({ev_free['valid']}/{ev_free['n']})  "
          f"avg_faces={ev_free['avg_faces']} avg_edges={ev_free['avg_edges']}")
    print(f"  fails: {ev_free['fail_reasons']}")
    print(f"constrained③  : Valid {ev_cons['valid_pct']}%  ({ev_cons['valid']}/{ev_cons['n']})  "
          f"avg_faces={ev_cons['avg_faces']} avg_edges={ev_cons['avg_edges']}")
    print(f"  fails: {ev_cons['fail_reasons']}")
    print(f"Δ Valid = +{report['delta_valid_pct']} pts   ({mins} min)")
    print(f"saved -> {os.path.join(evid, 'gen_report.json')}")


if __name__ == '__main__':
    main()
