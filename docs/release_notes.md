# 发布说明

## 版本

- 日期：`2026-04-18`
- 版本状态：开发联调可用版

## 本次主要变更

### 1. 默认检测权重切换

系统默认 YOLO 权重切换为：

- `outputs/training/yolo26s_autodl_8_1_1/weights/best.pt`

### 2. 视频分析链路打通

完成了从前端上传到后端分析再到结果下载的完整链路：

- 前端页面：`/video-analysis`
- 后端接口：`/api/v1/video-analysis/*`

### 3. Qwen 与 YOLO 联动

视频分析支持同时输出：

- 检测框
- 场景描述
- 活动识别
- 步骤提示
- PPE 状态
- 报警详情

### 4. 报警规则配置化

报警规则迁移到：

- `configs/alerts/alerting.yaml`

当前已配置：

- 手套缺失
- 护目镜缺失
- 实验服缺失

### 5. JSON 下载编码修正

分析 JSON 下载已明确返回 UTF-8。

### 6. 视频标注渲染优化

已完成：

- 中文字体渲染替换
- 目标框样式优化
- 报警边框强化
- 尾段汇总显示

### 7. 标注视频行为修正

已从“抽帧闪现框”改为：

- 全程逐帧显示检测框
- 末尾汇总展示报警文字

## 新增或重点文档

本次补充：

- `docs/deployment.md`
- `docs/api.md`
- `docs/configuration.md`
- `docs/model_card.md`
- `docs/test_and_acceptance.md`
- `docs/operation_guide.md`
- `docs/training_report.md`
- `docs/video_analysis_behavior.md`
- `docs/handoff.md`

## 已知限制

1. 当前视频分析仍为离线处理，不是浏览器实时推理
2. 护目镜告警仍主要依赖 Qwen 语义判断
3. Windows 中文文件路径在部分 OpenCV 场景下仍可能不稳定

## 建议下阶段工作

1. 增加目标跟踪，减少逐帧框抖动
2. 扩展更多告警规则
3. 增加前端视频预览内联比对
4. 视需求评估 WebSocket 流式预览
