# Workflow

从新项目接入到部署的完整流程：

## 1. 初始化工程与环境

1. 拉取项目代码。
2. 执行环境初始化脚本（固定环境名 `LabSOPGuard`）。
3. 运行 `14_check_environment.py` 生成环境检查报告。

## 2. 接入 SOP 与数据

1. 将 SOP 文档放入 `data/raw/sop_docs`。
2. 将视频数据放入 `data/raw/videos`。
3. 准备 `data/raw/annotations.jsonl`。
4. 更新 `configs/sop/rules.yaml` 与 `configs/data/dataset.yaml`。

## 3. 数据质量检查

1. 执行 `scripts/data_check.py`。
2. 查看 `outputs/reports/data_check_report.json`。
3. 修复字段缺失、路径错误、标注异常。

## 4. 数据划分

1. 执行 `scripts/data_split.py`。
2. 生成 train/val/test split。
3. 固定随机种子保证可复现。

## 5. 模型训练与调参

1. 按 `configs/model/vla_model.yaml` 运行训练。
2. 保存训练历史与关键参数。
3. 在实验日志记录每次改动。

## 6. 推理与违规识别

1. 执行 `scripts/infer.py` 进行单视频验证。
2. 执行 `scripts/export_results.py` 批量导出。
3. 校验事件结构化输出字段完整性。

## 7. 实时告警联调

1. 执行 `scripts/run_monitor.py`。
2. 验证违规触发与告警冷却逻辑。
3. 检查告警归档 `alerts.jsonl`。

## 8. 报告生成与归档

1. 使用报告输入构建模块聚合事件。
2. 生成 PDF/TXT 报告。
3. 归档到 `outputs/reports`。

## 9. 部署与运维

1. 部署到边缘节点或服务器。
2. 配置多摄像头时间对齐参数。
3. 建立巡检和报警通道监控。

## 10. 版本发布

1. 更新 README/docs/web。
2. 记录数据版本与规则版本。
3. 提交代码并打发布标签。
