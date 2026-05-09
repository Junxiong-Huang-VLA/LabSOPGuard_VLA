# 角色四：VLM 集成开发者（VLM Integration Developer）

## 职责定位

负责 Qwen/DashScope 多模态大模型的调用集成、语义增强、帧分析与 prompt 工程。是 AI 语义理解层的第一责任人。

## 核心工作内容

### API 约束（必须遵守）
- **唯一提供商**：Qwen / DashScope（阿里云）
- **禁止**引入 `anthropic`、原生 `openai`、`cohere` 等其他厂商 SDK
- Key 来源：`.env` → `DASHSCOPE_API_KEY`（禁止硬编码）
- 文本生成：`openai.OpenAI(api_key=..., base_url=DASHSCOPE_BASE_URL)` + `chat.completions.create`
- 多模态：`dashscope.MultiModalConversation.call`

### 主要代码区域
```
src/labsopguard/
├── video_analysis.py           # 视频帧抽样 + Qwen 帧分析
├── reasoning.py                # 合规推理 + Qwen 文本推理
├── material_publishing/
│   └── semantic_enhancer.py    # Qwen VLM 语义命名增强
├── qwen_writeback.py           # 帧分析结果写回
└── asr.py                      # 语音转文字（ASR）
```

### 语义命名流程
`display_name` 优先级：
1. `event.display_name`（EventProposalBuilder 生成）
2. Qwen live 调用（`MATERIAL_DISPLAY_NAME_QWEN_ENABLED=true`）
3. rule_based（降级）

控制开关（`.env`）：
```bash
MATERIAL_DISPLAY_NAME_QWEN_ENABLED=true
MATERIAL_DISPLAY_NAME_QWEN_MODEL=qwen3.6-flash
```

### Prompt 管理
- 所有 prompt 模板集中在 `src/labsopguard/reasoning.py` 和 `video_analysis.py`
- 新增 prompt 必须有版本注释，方便 A/B 测试
- 帧分析 prompt 结构：`系统角色 + 实验背景 + 帧图片列表 + 输出格式要求`

### 调试
```bash
# 检查 Qwen 集成
python tools/check_qwen_integration.py
# 多模态评估
python tools/eval_multimodal_videos.py
```

### 关注指标
- 帧分析成功率（Qwen API 调用成功率）≥ 99%
- `display_name` 语义质量（人工抽查 20 条/周）
- API 响应时间 p95 ≤ 5s

## 禁止事项
- `display_name` 不得使用 `evidence_summary` 字段（含调试信息）
- 不得在 event_preprocessing 或 step_bridge 内直接调用 Qwen
- API Key 不得出现在任何代码文件、日志或 Git commit 中
