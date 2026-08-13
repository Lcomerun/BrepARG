# P0-B Windows 训练窗口预检

生成时间：`2026-08-13T23:04:12+08:00`。

## 当前结论

- Windows Update active hours：`12:00–06:00 (18 h, source=ux_settings)`。
- 本次动作前原值：`09:00–03:00 (18 h, source=ux_settings)`。
- Pending restart：`generic PendingFileRenameOperations signal only; Windows Update RebootRequired and CBS markers are clear. Reboot at a convenient maintenance point; automatic resume remains required`。
- 本次动作：`set_active_hours` / `applied`。
- active hours 只能降低时段内自动重启概率，并不是训练不中断保证；正式训练仍必须使用原子 checkpoint 和自动续训。

## GPU 与存储

- GPU 0: NVIDIA GeForce RTX 3060，driver 591.86，显存 1042/12288 MiB，利用率 0.0%，温度 47.0 °C。
- `repository_volume`：free 53.83 GiB / total 631.2 GiB，used 91.5%。

## 建议与恢复

建议训练窗口使用 `12:00–06:00`（18 小时上限），把较可能的自动重启窗口留在白天。若当前窗口已经是该值，不必重复设置。设置命令必须在提升权限的终端中运行；权限不足、策略覆盖、自动 active hours、原值不可读或备份失败时，工具会拒绝写入并在 `audit.json` 留证：

    python tools/audit_windows_training_window.py --report-dir reports/p0b_stability_preflight_20260813 --set-active-hours 12 6 --confirm-active-hours-change

如果设置成功，工具会先生成不可覆盖的 `active_hours_backup.json`；同一报告目录再次设置会被拒绝，以免丢失原值。恢复原值：

    python tools/audit_windows_training_window.py --report-dir reports/p0b_stability_preflight_20260813 --restore-active-hours reports/p0b_stability_preflight_20260813/active_hours_backup.json --confirm-active-hours-change

工具只允许写入固定 Windows Update UX 键中的 `ActiveHoursStart` 与 `ActiveHoursEnd`，不禁用 Windows Update 服务，不改计划任务，也不写策略键。若企业/本机策略覆盖 active hours，工具会 fail closed，需由管理员按组织策略处理。

## 隐私与证据边界

报告不记录主机名、用户名、绝对路径、IP/内网地址、GPU UUID/序列号或进程列表。GPU 仅保留型号、驱动、显存、利用率和温度；磁盘仅用逻辑标签记录容量。`artifact_manifest.json` 为轻量报告文件提供 SHA-256，checkpoint、数据和原始系统日志不进入仓库。
