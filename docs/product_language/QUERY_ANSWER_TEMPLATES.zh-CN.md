# 查询回答模板

## 有证据支持

`基于 {available_days}/30 天数据，找到可追溯证据，已按证据链回答。`

必须同时返回：

- `answer.status = supported`
- 至少一个带 `evidence_item_ids`、`ledger_event_ids`、`evidence_bundle_ids`、`material_ids`、`keyframe_refs`、`keyclip_refs` 或 `evidence_links` 的 claim
- `partial_window_notice_zh = 基于 {available_days}/30 天数据`，当窗口不完整时

## 无可引用证据

`未找到可引用动作事件或证据包，不能形成强结论。`

必须同时返回：

- `answer.status = unsupported`
- `claims = []`
- `confidence = 0.0`
- `human_confirmation_status = unconfirmed`

## 部分窗口

`基于 {available_days}/30 天数据；当前不是完整 30 天态势快照。`

使用条件：

- `partial_window = true`
- `is_full_30_day_memory = false`
- 可用日期数小于 30，或快照标记为 partial

## 人工反馈记录

`该动作序列模式来自 30 天动作记忆聚类，仍需人工反馈记录确认。`

使用条件：

- claim 只有自动聚类或 VLM/YOLO 推断
- `human_confirmation_status` 不是 `human_confirmed`
- 不能输出 SOP、项目、样品名等强事实
