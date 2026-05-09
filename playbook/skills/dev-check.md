# dev-check — 开发前检查

每次开始开发前执行，确保环境和配置一致。

## 执行步骤

1. 读取 `LabSOPGuard.md` 和 `CLAUDE.md` 的约束清单
2. 检查 YOLO 权重路径：`configs/model/detection_runtime.yaml` → `model` 字段，确认 `.pt` 文件存在
3. 检查 `.env` 中 `DASHSCOPE_API_KEY` 是否存在
4. 确认当前要修改的代码路径在 `LabSOPGuard/src/labsopguard/` 下，而非 `lab_preprocessing/`
5. 运行快速健康检查：
   ```bash
   cd D:/LabEmbodiedVLA/LabSOPGuard
   python -c "from labsopguard.detectors import resolve_yolo26_weights_path; print(resolve_yolo26_weights_path())"
   ```
6. 运行核心测试：
   ```bash
   pytest -q tests/test_model_data_enhancements.py tests/test_material_production_features.py
   ```
7. 报告检查结果，列出任何发现的问题

若检查全部通过，输出"✓ 开发前检查通过，可以开始。"
