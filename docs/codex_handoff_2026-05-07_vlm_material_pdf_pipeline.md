# Codex Handoff - YOLO/VLM 素材管线与专业 PDF 报告

日期：2026-05-07  
项目目录：`D:\LabCapability`  
应用目录：`D:\LabCapability\LabSOPGuard`  
当前实验 ID：`2190fe06-3619-45fc-96ef-1bb8afb9bdf9`  
当前前端：`http://127.0.0.1:5177/experiments/2190fe06-3619-45fc-96ef-1bb8afb9bdf9/workspace`

## 新对话开场建议

请先读取并遵守：

- `D:\LabCapability\AGENTS.md`
- `D:\LabCapability\CODEX_HANDOFF.md`
- `D:\LabCapability\LabSOPGuard\docs\codex_handoff_2026-05-07_vlm_material_pdf_pipeline.md`
- `D:\LabCapability\LabSOPGuard\docs\professional_report_handoff_2026-05-07.md`

当前重点：继续固化 RealityLoop 的实验分析主线：YOLO 物理动作证据、VLM 辅助理解、前端审核入库、专业 PDF 报告自动生成与报告页入口。

## 已完成的核心工作

### 1. YOLO/VLM/前端审核素材 pipeline 已固化

新增通用管线模块：

`D:\LabCapability\src\key_action_indexer\yolo_vlm_pipeline.py`

当前固定流程：

```text
YOLO连续检测
-> 物理动作候选
-> VLM受YOLO证据约束的语义复核
-> YOLO物理证据回验
-> 前端审核入口
-> 审核通过后同步进入 material_references 文件夹和关键素材库
```

关键规则：

- YOLO 是物理证据底座。
- VLM 只能确认 YOLO evidence packet 里的 `allowed_confirmed_objects`。
- VLM 不能把显示器、键盘、电线、反光背景等凭空确认成实验器材。
- 不被 YOLO 支撑的对象只能进入 `uncertain_objects`。
- 候选素材必须前端审核通过后才进入正式文件夹和关键素材库。

相关改动：

- `D:\LabCapability\src\key_action_indexer\material_references.py`
  - `build_yolo_material_candidates(...)` 已接入 `apply_yolo_vlm_review_pipeline(...)`
  - 候选索引会写入 `pipeline_status`、`yolo_recheck`、`vlm_semantics`
  - 会生成 `_material_review_queue\pipeline_summary.json`
- `D:\LabCapability\LabSOPGuard\backend\main.py`
  - key-action 分析任务会创建可选 VLM client
  - 环境变量：
    - `KEY_ACTION_ENABLE_VLM_ASSIST`，默认开启
    - `KEY_ACTION_VLM_MODEL`，默认 `qwen3.6-plus`
    - `KEY_ACTION_VLM_MAX_GROUPS`，默认 `8`
  - 候选 API 已透出 pipeline 状态
- `D:\LabCapability\LabSOPGuard\frontend-app\src\pages\MaterialSearch.tsx`
  - 前端候选审核入口显示 YOLO 有效帧、VLM 状态、审核后同步入库提示

当前实验的候选 API 已验证：

```text
total = 2
pending_total = 0
pipeline_status = vlm_assisted_yolo_recheck_passed
yolo_recheck.status = passed
vlm_semantics.status = aligned
confirmed_objects = balance, gloved_hand
```

## 专业 PDF 报告能力接入状态

用户给的交接文档：

`D:\LabCapability\LabSOPGuard\docs\professional_report_handoff_2026-05-07.md`

其中已有专业报告生成器：

`D:\LabCapability\LabSOPGuard\src\labsopguard\professional_report.py`

核心入口：

```python
generate_professional_report_pdf(
    overview=...,
    key_actions=...,
    materials=...,
    output_pdf_path=...,
    logo_path=...,
)
```

目标输出固定为：

```text
D:\LabCapability\LabSOPGuard\outputs\experiments\{experiment_id}\reports\professional_report_qwen36max.pdf
D:\LabCapability\LabSOPGuard\outputs\experiments\{experiment_id}\reports\professional_report_qwen36max.html
D:\LabCapability\LabSOPGuard\outputs\experiments\{experiment_id}\reports\professional_report_qwen36max.json
D:\LabCapability\LabSOPGuard\outputs\experiments\{experiment_id}\reports\professional_report_manifest.json
```

已经接入的后端封装：

- `D:\LabCapability\LabSOPGuard\backend\main.py`
  - `_professional_report_output_paths(...)`
  - `_generate_professional_report_for_experiment(...)`
  - `_attach_professional_report_output_paths(...)`
  - `_experiment_output_artifact_paths(...)` 已新增：
    - `professional_report_pdf`
    - `professional_report_html`
    - `professional_report_json`
    - `professional_report_manifest_json`
  - `_build_analysis_overview(...)` 的 `artifacts` 已透出：
    - `professional_report_pdf`
    - `professional_report_html`
    - `professional_report_json`

接入点已加到：

- `_run_experiment_pipeline(...)`
- `_run_experiment_service_only(...)`
- attach existing video analysis 的完成路径
- `_run_key_action_index_task(...)`

设计原则：

- 默认自动生成，环境变量 `EXPERIMENT_PROFESSIONAL_REPORT_ENABLED=0` 可关闭。
- PDF 生成失败默认不让主分析失败，只写 `professional_report_manifest.json`。
- 环境变量 `EXPERIMENT_PROFESSIONAL_REPORT_FAIL_PIPELINE=1` 可改成失败即中断。
- 报告页 `ExperimentReport.tsx` 已加“专业PDF报告”按钮，只有 artifact ready 时显示。

## 当前未完成/需要继续处理的问题

### 1. WeasyPrint/GTK runtime 冲突

当前手动补跑当前实验 PDF 时出现过：

```text
function/symbol 'pango_context_set_round_glyph_positions' not found
in library 'C:\Program Files\Gtk-Runtime\bin\libpango-1.0-0.dll'
```

原因：

- Windows PATH 中旧版 GTK 路径 `C:\Program Files\Gtk-Runtime\bin` 排在或参与了 DLL 解析。
- 新版可用路径是：

```text
C:\Program Files\GTK3-Runtime Win64\bin
```

当前 `professional_report.py` 里已有 `_ensure_weasyprint_runtime_path()`，但因为 Python 进程已加载 DLL 或 PATH 中旧 runtime 干扰，仍可能失败。

建议下一步：

1. 修改 `_ensure_weasyprint_runtime_path()`，强制把旧路径从 `PATH` 移除，并优先 `GTK3-Runtime Win64`。
2. 确认 uvicorn 启动前 PATH 里不要出现旧 `Gtk-Runtime`。
3. 重启后端 `8002`。
4. 再调用 `_generate_professional_report_for_experiment(...)` 补当前实验。

当前旧 PDF 文件仍存在：

```text
D:\LabCapability\LabSOPGuard\outputs\experiments\2190fe06-3619-45fc-96ef-1bb8afb9bdf9\reports\professional_report_qwen36max.pdf
```

但 `professional_report_manifest.json` 目前记录的是一次失败状态，需要修复后重写。

### 2. 当前补跑命令曾超时

曾执行：

```powershell
$env:PYTHONPATH='D:\LabCapability\src;D:\LabCapability\LabSOPGuard\src'
$env:PYTHONIOENCODING='utf-8'
python - << script calling _generate_professional_report_for_experiment
```

结果超时，随后 manifest 显示 GTK/Pango 错误。没有发现残留 professional report 生成进程。

### 3. “Selected model is at capacity”

用户问过这是什么意思。解释：

- 这不是 RealityLoop 本身的业务错误。
- 通常是模型服务侧提示当前选中模型满载。
- 处理方式：稍后重试，或切换模型。
- 对本地 PDF 渲染的 GTK/Pango 错误没有直接关系。

## 已验证

已通过：

```powershell
python -m compileall -q src backend tests
python -m pytest tests\test_experiment_empty_state.py::test_professional_report_generation_uses_standard_experiment_reports_folder tests\test_analysis_overview_contract.py -q
npm run build
```

此前完整回归也通过过：

```powershell
python -m pytest -q
npm run test -- --run
npm run build
```

注意：专业 PDF 自动接入后尚未重新跑完整 `pytest -q`，建议修完 GTK runtime 后再跑一次完整回归。

## 新增/修改文件清单

核心新增：

- `D:\LabCapability\src\key_action_indexer\yolo_vlm_pipeline.py`
- `D:\LabCapability\LabSOPGuard\docs\codex_handoff_2026-05-07_vlm_material_pdf_pipeline.md`

核心修改：

- `D:\LabCapability\src\key_action_indexer\material_references.py`
- `D:\LabCapability\LabSOPGuard\backend\main.py`
- `D:\LabCapability\LabSOPGuard\frontend-app\src\pages\MaterialSearch.tsx`
- `D:\LabCapability\LabSOPGuard\frontend-app\src\pages\ExperimentReport.tsx`
- `D:\LabCapability\tests\test_material_references.py`
- `D:\LabCapability\LabSOPGuard\tests\test_experiment_empty_state.py`

## 新对话建议执行顺序

1. 读取本交接文档和 `professional_report_handoff_2026-05-07.md`。
2. 修复 WeasyPrint GTK runtime 选择：
   - 重点检查 `src\labsopguard\professional_report.py::_ensure_weasyprint_runtime_path`
   - 确保旧 `C:\Program Files\Gtk-Runtime\bin` 不参与 DLL 解析
3. 重启后端 `8002`，确保 PATH 生效。
4. 补跑当前实验的专业 PDF manifest。
5. 打开 `/experiments/{id}/report` 检查“专业PDF报告”按钮是否出现。
6. 运行：

```powershell
python -m compileall -q src backend tests
python -m pytest tests\test_experiment_empty_state.py::test_professional_report_generation_uses_standard_experiment_reports_folder tests\test_analysis_overview_contract.py -q
npm run build
```

7. 若时间允许，运行完整：

```powershell
python -m pytest -q
npm run test -- --run
npm run build
```

