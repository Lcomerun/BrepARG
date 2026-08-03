# 对话恢复的事故记录

本文件只记录曾由用户粘贴终端输出、但原始服务器日志未必仍在本机的事件。它们属于 `conversation_derived_audit_record`，不是打包器伪造的原始日志。

## C001：服务器 VQ resume 与 scratch

- resume：从旧 full epoch100 best 的历史摘要恢复，`start_epoch=86`，bs 256，lr `5e-5`；early stop epoch 164，best epoch 148，validation 约从 `7e-5` 到 `5e-5`。
- scratch：bs 128，lr `1e-4`；early stop epoch 107，best epoch 93，best validation 约 `1e-4`。
- scratch continuation：从 epoch 108 继续，目标 500；最终 early stop epoch 440，best epoch 340，best validation 约 `5e-5`。
- 独立 50k patch 评测中 resume overall mean/median/p95 最优；scratch max 较低但总体分布更差。
- 结论：选择 resume-best 是有独立 patch 评测依据的；不能仅按训练 validation 四舍五入后的 `0.00005` 判断 scratch 等价。

## C002：sequence smoke 依赖失败

- 首次从 `parsed_abc_0000.pkl.zst` 做 200 records smoke 时抛出 `ModuleNotFoundError: chamferdist`。
- sequence 文件没有生成，后续 source-path audit 和 AR preflight 又因 FileNotFound 连续失败。
- 结论：依赖失败是根错误；后续 FileNotFound 是级联，不是三个独立问题。

## C003：AR batch size 调整

- 初始 ctx2048/bs8 时 GPU 约 28%，显存约 6.7 GiB；用户停止后从 `ar_latest.pt` epoch 20 恢复，bs32，GPU 约 96%，显存约 24.1 GiB。
- 提升 batch 的吞吐收益真实存在，但余下显存不足以保证所有 batches 安全，因为动态长度/padding 和 loss logits float conversion 会造成峰值。
- 结论：`nvidia-smi` 的瞬时余量不是 bs64 安全证明。

## C004：AR OOM 与发散

- epoch 104/106/110/112 val CE 逐步到约 `0.3097`；epoch 115 running CE 从约 0.47 持续升至 2.32，随后 CUDA OOM，报一次需要 2.34 GiB。
- 后续错误恢复中 epoch 124/125 CE 升至 1.9/6.5；epoch 130/140 `train_CE=0.0000 val_CE=nan`。
- checkpoint audit：epoch 117 best finite；epoch 142 latest 的 `transformer.wte.weight` nonfinite。
- 结论：不是单纯 OOM；OOM 前已有优化发散，OOM 后继续运行/恢复污染模型。

## C005：推理长度不匹配

- `generate_validate.py` 曾创建 `max_seq_len=1024` 的 AR 模型，却加载 `[2048,256]` 的 `transformer.wpe.weight`。
- PyTorch strict load 正确地报告 shape mismatch。
- 后续代码改为从 checkpoint config 读取 max sequence length。

## C006：低验证损失与生成质量冲突

- AR finite best validation CE 约 0.299，但生成样本仍偏简单。
- same-data BrepARG 长训 validation CE 比短训更好，复杂/strict 数量也有改善，但 faces/edges 中位数仍低。
- 结论：CE 是必要训练指标，不是 CAD validity/complexity/diversity 的替代指标。

## C007：磁盘复制和删除口径

- 用户观察到只向 C 盘复制约 10 GB，而 D 盘删除约 58 GB。
- 后续项目审计确认 sequence 存在 NTFS hardlink，同一逻辑内容可能有多个路径；压缩 archive、logical size、allocated size 和被选目录范围也不同。
- 结论：没有 exact deletion manifest 时不能由两个资源管理器数字反推丢失；今后必须记录路径、link count、logical/physical bytes、hash 和目标验证。

## C008：夜间重启与恢复训练

- 电脑重启中断了长训练；后续按 best checkpoint 恢复，而不是从未知进程状态继续。
- 结论：PID 不是 durable state；checkpoint、history、effective config 和 source/data identity 才是恢复依据。

## C009：CPU 高而 GPU 低

- 某些阶段 GPU 利用率下降而 CPU 高，可能是 sequence 数据读取、decompression、pickle、OCC/STEP validation/render 或 AR validation。
- 当时不能仅凭 `nvidia-smi` 断言训练卡死；必须关联 PID、log stage、CPU command line 和最近输出。
- 本包的 `status`/run manifest 旨在保存明确阶段和日志位置。

## C010：生成时过滤不解决模型

- 用户比较约束生成和原始 BrepARG 逻辑后，二者都有大量不合格结果。
- quality gate 能避免把不水密结果交付为幸存者，但不能把被拒绝 attempts 从 Valid 分母中删除。
- 结论：过滤可用于安全输出和候选选择，不能作为模型质量改善结论。
