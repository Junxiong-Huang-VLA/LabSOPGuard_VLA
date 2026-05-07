# win-remote-control 升级前后对比：Chromeleon 7 操控效果

## 背景

OpenClaw 通过 `win-remote-control` skill 远程操控 Windows 上的 Thermo Chromeleon 7 离子色谱工作站。Chromeleon 是典型的 WinForms 仪器软件，UI 控件密集、字号小、自绘 DataGrid 多，对点击精度要求苛刻——偏 20px 就可能点错按钮或者选错行。

本次升级在 `bridge.py` 中新增 OCR 定位层，形成 **a11y → OCR → 视觉** 三级降级机制（`smart_click`），替代此前以 VLM 视觉估坐标为主的流程。

---

## 一、架构对比

### 升级前

```
LLM 决定目标 → screenshot → VLM 看图估坐标 → patch_click 裁剪局部图
→ VLM 再看裁剪图精修坐标 → finalize 点击 → screenshot 验证
```

- 每次点击 **至少 2 轮截图 + 2 轮 VLM 推理**
- 坐标精度完全依赖 VLM 空间推理能力
- 遇到密集小按钮（Off/On）或长文字伪链接（Click here to add a new injection）容易偏行

### 升级后

```
LLM 决定目标 → smart_click --name "目标"
  ├─ Level 1: a11y_element 搜索 → 命中就用 UIAutomation rect → 点击（<200ms）
  ├─ Level 2: OCR 截图 → 模糊匹配 → 用 bbox 中心点击（1-2s）
  └─ Level 3: 返回 fallback → 降级到 patch_click 视觉流程
```

- 有文字标签的元素 **一条命令完成**，无需截图-看图-估坐标循环
- 坐标来自测量（a11y rect 或 OCR bbox），不靠空间推理

---

## 二、Chromeleon 关键操作的逐项对比

### 测试环境
- 远端：Windows 11，Chromeleon 7.2.10，1920×1080
- 截图来源：实际桥接截图，非模拟
- OCR 引擎：RapidOCR（PP-OCR v3 ONNX），纯 CPU

### 2.1 Pump_ECD 面板按钮

| 目标 | 升级前（VLM） | 升级后（smart_click） | 改善 |
|---|---|---|---|
| **Off 按钮** | 需截图 → 看图 → 估坐标 → finalize，VLM 容易把 Suppressor 的 Off 和 Pump_ECD 的 Off 搞混（两个 Off 相距 ~150px）| `smart_click --name "Off"` → OCR 命中 (775,305)，1px 误差 | 从 4 步 → 1 步；消除双 Off 混淆 |
| **On 按钮** | 同上，且 On 在 Off 旁边 23px，VLM 经常偏到 Off | OCR 命中 (776,276)，2px 误差 | 消除 On/Off 互相误点 |
| **Connect** | VLM 能看到但坐标常偏 | OCR 命中 (466,351)，原文 "Cornect"（误读），模糊匹配 ratio=0.86 兜住 | 即使 OCR 误读也能定位 |
| **Disconnect** | 同上 | OCR 命中 (547,352)，原文 "Disconmect"，ratio=0.88 | 同上 |
| **Continue** | 可靠 | OCR 命中 (766,352)，精确匹配 | 持平，但少一轮截图 |
| **Prime** | 可靠 | OCR 命中 (768,505)，精确匹配 | 少一轮截图 |
| **Settings** | 可靠 | OCR 命中 (754,548)，精确匹配 | 少一轮截图 |

### 2.2 标签页切换

| 目标 | 升级前 | 升级后 | 改善 |
|---|---|---|---|
| **Queue 标签** | VLM 能看到但坐标经常偏到旁边的 Audit 标签（间距仅 ~50px）| `smart_click --name "Queue"` → OCR 命中 (882,95)，10px 误差但在 71×23 的可点击区域内 | 消除邻近标签误点 |
| **Pump_ECD 标签** | 类似，标签页密集 | OCR 命中 (444,387)，精确 | 同上 |
| **Home / Audit** | VLM 需要精确分辨多个标签页 | OCR 按名称精确匹配 | 消除 Tab 混淆 |

### 2.3 目录树操作（Data 视图）

| 目标 | 升级前 | 升级后 | 改善 |
|---|---|---|---|
| **realityloop 节点** | VLM 在密集树形结构中估坐标，偏差 10-30px 常见 | OCR 命中 (103,354)，7px 误差 | 可靠命中 |
| **Li-test 节点** | VLM 同上 | OCR 原文 "Utest"（`Li` 合并成 `U`），但加 `--x 220` 区域约束后 bbox 中心 (124,388) 仅偏 6px | 需模糊匹配，但坐标准确 |
| **20260322-Li-test** | VLM 在子节点级别容易选错层级 | OCR 命中 (147,372)，2px 误差 | 精确 |
| **origin 节点** | VLM 困难 | OCR "onigin"，模糊匹配命中 (120,405)，5px 误差 | OCR 误读但定位准确 |

### 2.4 特殊 UI 元素

| 目标 | 升级前 | 升级后 | 改善 |
|---|---|---|---|
| **"Click here to add a new injection"** | 灰色长文字伪链接，VLM 最容易偏行的目标——往往偏到上一行或下一行数据 | OCR 命中 (1113,510)，**1px 误差**，置信度 0.91 | **最显著改善**：从"经常偏行"到"1px 精准" |
| **Queue 表格中的 Pending 状态** | VLM 需要在表格密集行中定位特定文字 | `ocr_find --name "Pending"` 精确返回坐标 + 匹配文本（不点击，用于状态读取）| 新能力：直接读表格内容 |
| **Flow / Pressure 数值** | VLM 看截图读数 | `ocr_find` 可提取数值文本 | 新能力：结构化读取仪器参数 |

### 2.5 桌面图标（Chromeleon 7 入口）

| 目标 | 升级前 | 升级后 | 改善 |
|---|---|---|---|
| **Chromeleon 7 vs GasLab** | 两个绿色图标相邻，VLM 曾混淆过 | OCR 找到 "Chromeleon" 和 "Gaslab 2.1" 两个标签，文字层无二义性；但纯图标（如果标签被遮挡）仍需 VLM | 文字标签可区分时彻底解决；纯图标场景暂无改善 |

---

## 三、关键指标汇总

| 指标 | 升级前 | 升级后 | 变化 |
|---|---|---|---|
| 单次点击所需步骤 | 4-6 步（截图→裁剪→VLM看图→估坐标→finalize→验证截图）| 1 步（`smart_click`）| **-80%** |
| 单次点击耗时 | 5-15s（含 2 轮 LLM 推理）| 1-2s（OCR）或 <200ms（a11y）| **-70~95%** |
| 文字按钮定位精度 | ±10-50px（VLM 空间推理）| ±1-7px（OCR bbox 测量）| **提升 5-10x** |
| Off/On 双按钮消歧 | 经常混淆 | 精确区分 | 从不可靠→可靠 |
| 密集标签页误点率 | 偶发 | 按名称精确匹配 | 基本消除 |
| 长文字伪链接命中 | 经常偏行 | 1px 精度 | 从不可靠→精准 |
| DataGrid 内容读取 | 只能靠 VLM 看截图描述 | `ocr_find` 返回结构化文本+坐标 | 新能力 |
| 仪器参数读取 | VLM 看图读数 | OCR 提取文本 | 新能力 |
| LLM token 消耗/次 | ~2000 token（截图 base64 + 视觉推理）| ~0（OCR 本地推理，不走 LLM）| **-100%** |

---

## 四、仍存在的局限

| 场景 | 原因 | 应对 |
|---|---|---|
| a11y 在 Chromeleon 深层节点搜索 miss | Windows 端 a11y_element 实现可能只走浅层匹配 | OCR 作为 Level 2 已完整兜住 |
| OCR 小字体误读 | `Li-test→Utest`、`Connect→Cornect`、`origin→onigin` | 模糊匹配（SequenceMatcher ratio≥0.55）可兜住 |
| 同名元素多处出现 | `Li-test` 在树节点/标题栏/状态栏各一个 | `--x`/`--y` 区域约束 + LLM 给区域提示 |
| 纯图标目标（无邻近文字） | OCR 只认文字 | 仍走 patch_click 视觉流程 |
| DataGrid 行的点击（非读取） | OCR 能定位文字但行高仅 20px | 命中率高但需注意相邻行 |

---

## 五、对完整实验流程的影响

以"离子色谱完整跑样"为例，流程约 15-20 步：

```
打开 Chromeleon 7 → 进入 Data 视图 → 展开 realityloop 目录 → 选择 Li-test →
查看 Queue → 确认 Pending 状态 → 切到 Pump_ECD → 检查 Flow/Pressure →
点击 On → 等待稳定 → 点击 Connect → 点击 Continue →
回到 Queue → 确认状态变化 → ...
```

| 步骤类型 | 升级前 | 升级后 |
|---|---|---|
| 按钮点击（Off/On/Connect/Continue/Prime） | 每步 4-6 步 + 偶发误点 | `smart_click` 一步到位 |
| 标签页切换（Queue/Pump_ECD/Home） | 截图+定位，偶发误点邻近标签 | `smart_click` 按名称精确切换 |
| 目录树选择（realityloop/Li-test） | 截图+在密集树中估坐标，高失败率 | `smart_click --x 220` 区域约束 |
| 状态读取（Pending/Idle/Flow 值） | VLM 看截图口述 | `ocr_find` 返回结构化数据 |
| 桌面图标双击 | 截图+视觉定位，偶发混淆 | 有标签时 `smart_click`，无标签时仍走视觉 |

**预估整体效果**：一个 15 步的跑样流程，升级前平均需要 40-80 轮 tool call（含截图、裁剪、VLM 看图、finalize），偶发 2-3 次误点需要回退重试。升级后大部分步骤缩减到 1 轮 `smart_click`，总 tool call 降到 15-25 轮，误点率大幅下降。
