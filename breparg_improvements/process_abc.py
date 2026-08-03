"""
process_abc.py
======================
统一集中处理:把 **完整 ABC 原始数据集**(/data/public/luol/ABC/step/abc_XXXX_step_v00/<model>/*.step,
约 100 chunk × 10000 ≈ 1,000,000 个模型)解析成 parsed pkl(surf_ncs/edge_ncs/edgeFace_adj/...)。

要点(重要):
  - 解析这一步与旧方案**完全一致**(见 repro_outputs/DATA_FORMAT_COMPARISON.md),所以已存在的
    /data/public/luol/breparg_data/abc_parsed_50c(131,857 条 <=50 面子集)**可直接复用,无需重解析**。
  - 本脚本只用于**把已解析池扩大到更大规模**(解析更多 raw chunk)。新方案真正要"重处理"的是
    序列(FSQ+RCM)与 VQ-VAE,那一步在 train_newscheme_abc.py 里做(不在这里)。

特性:**断点续跑**(输出 pkl 已存在则跳过)、**多进程并行**、按 chunk 范围分批,日志可审计。
所有产物写 /data(/home 已 99% 满)。

用法:
  conda activate brepgen
  cd breparg_improvements
  # 解析前 5 个 chunk(约 5 万模型,<=50 面的会留下,其余被过滤)到一个新池:
  python process_abc.py --chunks 0-4 --out /data/public/luol/breparg_data/abc_parsed_full --workers 32
  # 解析全部 100 chunk(~1M,数小时;parsed 体量可能 ~400G,先确认 /data 余量):
  python process_abc.py --chunks all --out /data/public/luol/breparg_data/abc_parsed_full --workers 48

注意:MAX_FACE 过滤在 process_brep 内(>200 面直接丢);<=50 面才是 AR 用的;真正训练用的子集会更小。
"""

import os, sys, glob, time, pickle, argparse, signal
from multiprocessing import Pool

# 期望的 parsed 字段(与现有 abc_parsed_50c 完全一致;正确性校验用)
EXPECTED_FIELDS = {
    'surf_wcs', 'edge_wcs', 'surf_ncs', 'edge_ncs', 'corner_wcs',
    'edgeFace_adj', 'edgeCorner_adj', 'faceEdge_adj',
    'surf_bbox_wcs', 'edge_bbox_wcs', 'corner_unique',
}

_HERE = os.path.dirname(os.path.abspath(__file__))
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
sys.path.insert(0, os.path.join(BREPARG, 'process_data'))

RAW_ROOT = '/data/public/luol/ABC/step'


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def _parse_one(arg):
    """解析单个 .step。带**每文件超时**(SIGALRM),避免个别病态 STEP 让 worker 永久卡死。
    写出用临时文件 + 原子 rename,避免中断产生半截 pkl(保证断点续跑的正确性)。"""
    step_path, out_dir, timeout = arg
    name = os.path.splitext(os.path.basename(step_path))[0]
    out_pkl = os.path.join(out_dir, name + '.pkl')
    if os.path.exists(out_pkl):          # 断点续跑:已完成则跳过
        return ('skip', name)
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(int(timeout))
    try:
        import process_brep
        from occwl.io import load_step
        solids = load_step(step_path)
        if len(solids) != 1:
            return ('multi', name)
        data = process_brep.parse_solid(solids[0])
        if data is None:
            return ('filtered', name)
        # 正确性:字段必须与既有 parsed 一致
        if not EXPECTED_FIELDS.issubset(set(data.keys())):
            return ('badfields', name)
        tmp = out_pkl + f'.tmp{os.getpid()}'
        with open(tmp, 'wb') as f:
            pickle.dump(data, f)
        os.replace(tmp, out_pkl)         # 原子落盘
        return ('ok', name)
    except _Timeout:
        return ('timeout', name)
    except Exception:
        return ('error', name)
    finally:
        signal.alarm(0)                  # 取消闹钟


def enumerate_steps(chunk_spec):
    chunks = sorted(glob.glob(os.path.join(RAW_ROOT, 'abc_*_step_v00')))
    if chunk_spec != 'all':
        lo, hi = (chunk_spec.split('-') + [chunk_spec])[:2]
        lo, hi = int(lo), int(hi)
        chunks = chunks[lo:hi + 1]
    steps = []
    for c in chunks:
        steps.extend(sorted(glob.glob(os.path.join(c, '*', '*.step'))))
    return chunks, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chunks', default='0-0', help="'all' 或 'lo-hi'(含端点,0-based chunk 序号)")
    ap.add_argument('--out', default='/data/public/luol/breparg_data/abc_parsed_full')
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--timeout', type=int, default=60, help='每个 .step 解析超时秒数(防卡死)')
    ap.add_argument('--limit', type=int, default=0, help='只处理前 N 个(0=不限,用于冒烟)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    chunks, steps = enumerate_steps(a.chunks)
    if a.limit > 0:
        steps = steps[:a.limit]
    print(f"[{time.strftime('%H:%M:%S')}] chunks={len(chunks)} steps={len(steps)} out={a.out} "
          f"workers={a.workers} timeout={a.timeout}s", flush=True)

    counts = {'ok': 0, 'skip': 0, 'multi': 0, 'filtered': 0, 'badfields': 0, 'timeout': 0, 'error': 0}
    t0 = time.time()
    with Pool(a.workers, maxtasksperchild=200) as pool:  # 周期回收 worker,防 OCC 内存泄漏累积
        for i, (status, _) in enumerate(pool.imap_unordered(
                _parse_one, [(s, a.out, a.timeout) for s in steps], chunksize=8)):
            counts[status] += 1
            if (i + 1) % 2000 == 0:
                el = time.time() - t0
                rate = (i + 1) / max(1e-9, el)
                print(f"[{time.strftime('%H:%M:%S')}] {i+1}/{len(steps)}  {counts}  "
                      f"{rate:.0f}/s  ETA {(len(steps)-i-1)/max(1e-9,rate)/60:.0f}min", flush=True)
    el = (time.time() - t0) / 60
    total_parsed = len(glob.glob(os.path.join(a.out, '*.pkl')))
    print(f"[{time.strftime('%H:%M:%S')}] DONE {counts} in {el:.1f}min  total_pkls_in_out={total_parsed}", flush=True)


if __name__ == '__main__':
    main()
