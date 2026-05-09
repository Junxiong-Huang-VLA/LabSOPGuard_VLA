---
role_id: project_owner
invoke: /me
---

# 角色零：项目主理人（Project Owner）

## 身份定位

**独立全栈负责人**，一人承担 LabSOPGuard 项目的全部角色：

| 角色 | 实际工作 |
|------|---------|
| 产品负责人 | 所有优先级决策、架构取舍、约束规范制定 |
| 检测工程师 | AutoDL 训练、权重管理、评估 |
| 数据标注员 | Roboflow 云端标注（数据可能未同步本地） |
| DevOps | AutoDL 服务器、本地双机、conda 环境管理 |
| 核心研发 | 五类事件设计、step_bridge 链路、整体架构 |

## 工作方向

**研究 + 工程双轨并行，共用同一套基础设施：**

- **研究方向**：VLA（Vision-Language-Action）/ 具身智能
- **工程方向**：实验室 SOP 合规检测（LabSOPGuard）
- YOLO + Qwen VLM 感知层是两个方向的共同底座

## 工作风格与偏好

- 直接给结论，不需要铺垫和解释显而易见的事
- 优先减少重复操作、降低认知负担
- 决策由我来，执行交给 Claude Code
- 数据/文件找不到时，优先查云端（Roboflow）或其他设备，而不是直接下"不存在"的结论

## 当前项目状态（2026-04-22）

- **主项目**：`D:/LabEmbodiedVLA/LabSOPGuard/`
- **系统评级**：B（本地可用，内部 Demo 就绪）
- **当前最佳权重**：`yolo26s_autodl_8_1_1`（mAP50=0.977）
- **P0 任务**：tube/tube-cap/spearhead/pipette 标注补充（Roboflow 云端有，未同步本地）
- **API 提供商**：Qwen / DashScope（唯一，禁止替换）
- **启动命令**：`.\scripts\start_full_stack.ps1 -Restart -SkipRedis`

## Claude Code 协作约定


调用 `/me` 后，Claude Code 应当：

1. 默认我已了解所有技术细节，不做基础解释
2. 遇到文件/数据缺失，先查其他盘和云端再下结论
3. 修改任何代码前先对照 `LabSOPGuard.md` 的约束清单
4. 优先用 skill 命令完成重复操作，不要每次重新发明流程
5. 技术建议兼顾研究可扩展性和工程稳定性，不偏向任一侧
