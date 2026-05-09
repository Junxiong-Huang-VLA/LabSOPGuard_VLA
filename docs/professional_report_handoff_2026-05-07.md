# RealityLoop 专业实验分析报告交接文档

日期：2026-05-07  
项目目录：`D:\LabCapability\LabSOPGuard`  
实验 ID：`2190fe06-3619-45fc-96ef-1bb8afb9bdf9`  
运行 ID：`fb8b9d22-e99b-47b7-86a0-0147178bb205`

## 1. 当前结论

本轮已将专业实验分析 PDF 从“文字摘要型报告”升级为“证据链型专业报告”。

当前 PDF 已包含：

- 公司徽与正式封面
- Qwen 报告文本
- 报告可信度/归档等级
- 证据图谱
- SOP 标准对照矩阵
- 关键动作证据
- 微动作证据
- 双视角互证分析
- 风险告警证据帧
- 关键素材追溯图
- 局限性与审计信息
- 签字页：系统生成标识、审核人、批准人

当前报告等级为：`需复核后归档`  
当前归档判断为：`不可直接用于合规终审`

## 2. 最终产物

PDF：

`D:\LabCapability\LabSOPGuard\outputs\experiments\2190fe06-3619-45fc-96ef-1bb8afb9bdf9\reports\professional_report_qwen36max.pdf`

HTML 预览：

`D:\LabCapability\LabSOPGuard\outputs\experiments\2190fe06-3619-45fc-96ef-1bb8afb9bdf9\reports\professional_report_qwen36max.html`

JSON sidecar：

`D:\LabCapability\LabSOPGuard\outputs\experiments\2190fe06-3619-45fc-96ef-1bb8afb9bdf9\reports\professional_report_qwen36max.json`

渲染预览图目录：

`D:\LabCapability\LabSOPGuard\outputs\experiments\2190fe06-3619-45fc-96ef-1bb8afb9bdf9\reports\preview_pages_qwen36max`

最新 PDF 状态：

- 页数：14 页
- 大小：约 1.29 MB
- 证据图：8 张
- SOP 对照行：9 行
- 微动作证据：2 条
- 风险告警聚合：5 类
- 关键素材追溯：2 条

## 3. 主要代码改动

### `src/labsopguard/professional_report.py`

核心改动：

- 扩展标准报告结构：
  - `report_reliability`
  - `evidence_gallery`
  - `sop_compliance_matrix`
  - `multiview_evidence.alignment`
  - `key_action_evidence.actions[].micro_actions`
  - `risk_alerts.alerts[].evidence_image_path`
  - `materials_traceability.materials[].evidence_image_path`
- 新增实验目录证据读取逻辑：
  - 从 `analysis/analysis.json` 聚合风险告警、时间范围、帧范围和证据图
  - 从 `step_candidates.json` 生成 SOP 标准对照
  - 从 `key_action_index/indexed_db/micro_segments/*/metadata.json` 读取微动作证据
  - 从 `key_action_index/keyframes/seg_000001` 读取双视角关键帧
  - 从 `_material_review_queue/关键帧` 读取关键素材证据图
  - 从 `key_action_index/metadata/view_alignment.json` 读取双视角同步信息
  - 从 `key_action_index/metadata/process_quality_report.json` 读取质量检查状态
- 新增图片 base64 压缩嵌入，避免 PDF 依赖外部图片路径。
- `_normalize_report()` 现在会保留 Qwen 文本，同时用本地结构化证据覆盖/补强关键证据字段。
- 默认 Qwen 模型仍为 `qwen3.6-max-preview`。

### `templates/reports/professional_report.html.j2`

核心改动：

- 重做封面为正式报告卷宗式排版。
- 增加报告可信度页。
- 增加证据图谱卡片。
- 增加 SOP 标准对照表。
- 将关键动作由横向大表改为证据卡片。
- 增加微动作证据表和对应证据帧。
- 增加双视角同步指标与互证说明。
- 风险告警改为卡片式展示，带代表性证据图。
- 关键素材追溯改为图文卡片。
- 保留最终签字页。

### `src/experiment/service.py`

修复了一个测试兼容问题：

```python
try:
    analysis = self._step_reasoner.analyze_frame(path, detected_objects=detections)
except TypeError as exc:
    if "detected_objects" not in str(exc):
        raise
    analysis = self._step_reasoner.analyze_frame(path)
```

目的：

- 真实流程仍把 YOLO 检测结果传给 `analyze_frame`
- 旧测试或旧 mock 如果只接受 `analyze_frame(path)`，不会失败

## 4. PDF 生成技术栈

当前 PDF 使用：

- `WeasyPrint`
- `Jinja2`
- HTML 模板：`templates/reports/professional_report.html.j2`
- 报告生成器：`src/labsopguard/professional_report.py`

Windows 下 WeasyPrint 依赖 GTK runtime。当前可用路径：

`C:\Program Files\GTK3-Runtime Win64\bin`

如果手动重新渲染 PDF，建议先设置：

```powershell
$gtkNew='C:\Program Files\GTK3-Runtime Win64\bin'
$gtkOld='C:\Program Files\Gtk-Runtime\bin'
$parts=$env:PATH -split ';' | Where-Object { $_ -and ($_ -ne $gtkOld) -and ($_ -ne $gtkNew) }
$env:PATH=($gtkNew + ';' + ($parts -join ';'))
$env:PYTHONPATH='src'
$env:PYTHONIOENCODING='utf-8'
```

## 5. 重新渲染当前 PDF 的命令

该命令不会重新调用 Qwen，只使用已有 sidecar 里的 Qwen 文本和本地证据重新渲染：

```powershell
$gtkNew='C:\Program Files\GTK3-Runtime Win64\bin'
$gtkOld='C:\Program Files\Gtk-Runtime\bin'
$parts=$env:PATH -split ';' | Where-Object { $_ -and ($_ -ne $gtkOld) -and ($_ -ne $gtkNew) }
$env:PATH=($gtkNew + ';' + ($parts -join ';'))
$env:PYTHONPATH='src'
$env:PYTHONIOENCODING='utf-8'
@'
from pathlib import Path
import json
from labsopguard.professional_report import render_pdf_report, _enrich_context_from_files, _normalize_report

p = Path(r'outputs\experiments\2190fe06-3619-45fc-96ef-1bb8afb9bdf9\reports\professional_report_qwen36max.json')
d = json.loads(p.read_text(encoding='utf-8'))
context = _enrich_context_from_files(d['context'])
report = _normalize_report(d['report'], context, d.get('qwen', {}))
out = Path(d['pdf_path'])

render_pdf_report(report=report, context=context, output_path=out, qwen_meta=d.get('qwen', {}))
d['context'] = context
d['report'] = report
p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding='utf-8')
print(out.resolve(), out.stat().st_size)
'@ | python -
```

## 6. Qwen 使用状态

报告文本使用过 Qwen：

- 模型：`qwen3.6-max-preview`
- `qwen_used`: `True`
- 原始 Qwen 响应已保存在 sidecar：

`professional_report_qwen36max.json`

本轮最后几次只是重渲染和本地证据增强，没有重新调用 Qwen。

## 7. 真实实验证据来源

本次 PDF 使用的关键本地证据包括：

- `analysis/analysis.json`
  - 风险告警、告警时间、告警帧、PPE 缺失信息
- `step_candidates.json`
  - SOP 标准步骤、候选证据、promotion decision、blocking issues
- `steps.json`
  - 结构化步骤列表
- `key_action_index/metadata/view_alignment.json`
  - 双视角同步偏移、同步置信度、解释
- `key_action_index/metadata/process_quality_report.json`
  - 质量检查状态，当前整体为 `fail`，分数约 `0.6667`
- `key_action_index/keyframes/seg_000001`
  - 第一人称、第三人称和交互关键帧
- `key_action_index/indexed_db/micro_segments/*/metadata.json`
  - 微动作证据
- `_material_review_queue/关键帧`
  - 素材追溯证据帧

## 8. 验证结果

已完成验证：

```powershell
python -m compileall -q src backend
```

通过。

```powershell
npm run build
```

通过。

```powershell
pytest -q
```

通过：

`220 passed in 163.17s`

目标失败用例也单独验证通过：

```powershell
pytest -q tests\test_formal_pipeline.py::test_experiment_service_auto_aligns_visual_flash_between_streams
```

结果：

`1 passed`

## 9. 已知限制

当前报告虽然比之前专业很多，但仍有这些现实限制：

- 关键动作主片段仍只有 `seg_000001`，只是通过微动作进一步拆出了 2 个证据点。
- SOP 步骤仍全部为 candidate/hold_for_review，不应直接用于合规终审。
- 风险告警虽然已有证据帧，但仍需人工确认是否存在遮挡、误检或人员未进入操作区。
- 双视角同步信息显示 offset `0.00s`、confidence `1.00`，但仍建议在正式归档前抽查原始视频。
- 质量检查状态来自 `process_quality_report.json`，当前整体为 `fail`，因此报告等级被设为 `需复核后归档`。

## 10. 新对话接续建议

新对话开始时可以直接说：

> 请先阅读 `D:\LabCapability\LabSOPGuard\docs\professional_report_handoff_2026-05-07.md`，继续完善 RealityLoop 专业实验分析 PDF。当前重点是提高关键动作细分能力、增强 SOP 合规判定、继续优化 PDF 版式和报告可信度。

如果下一步继续做，优先级建议：

1. 把 `seg_000001` 自动拆成更多专业动作：取样、称量、转移、放回、记录。
2. 将 SOP 对照从 candidate 提升到 confirmed/failed，需要更强规则或人工审核入口。
3. 将报告导出入口接入前端“分析报告”按钮。
4. PDF 中增加目录页和术语说明页。
5. 将 Qwen 生成从“完整报告 JSON”改成“只生成专业结论段”，结构化证据全部由后端确定，减少模型漂移。

## 11. 注意事项

- 当前目录不是 git 仓库，`git status` 会报：

`fatal: not a git repository (or any of the parent directories): .git`

- 本项目 AGENTS.md 要求保留 `src/key_action_index` 独立性，不要把关键动作索引逻辑耦合进 LabSOPGuard 主应用。
- 每次改动后建议至少跑：
  - `python -m compileall -q src backend`
  - `npm run build`
  - `pytest -q`

