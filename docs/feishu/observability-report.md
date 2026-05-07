# OpenClaw 观测系统报告：Langfuse vs Logfire

## 一、Langfuse 集成

**状态：已配置但当前禁用**

### 部署架构

Langfuse 采用**自托管模式**，通过 `langfuse-compose.yml` 部署完整的服务栈：

| 组件 | 版本 | 用途 |
|------|------|------|
| PostgreSQL | 16 | 元数据存储（项目、用户、Trace） |
| ClickHouse | 24.12 | 高性能分析查询引擎 |
| Redis | 7 | 缓存、会话管理、队列 |
| MinIO | — | S3 兼容对象存储（媒体/事件上传） |
| Langfuse Web | v3 | Web UI，端口 3000 |
| Langfuse Worker | — | 后台任务处理 |

### 启动方式

```bash
# 启动全部服务
docker compose --env-file .env.langfuse -f langfuse-compose.yml up -d

# 查看日志
docker compose --env-file .env.langfuse -f langfuse-compose.yml logs -f langfuse-web

# 停止
docker compose --env-file .env.langfuse -f langfuse-compose.yml down
```

启动后访问 `http://localhost:3000`，使用 `.env.langfuse` 中配置的初始管理员账号登录：

- 邮箱：`admin@openclaw.local`
- 组织：`openclaw-org` / 项目：`openclaw-project`

### 插件实现（langfuse-observer）

位于 `extensions/langfuse-observer/`，使用 Langfuse JS SDK v3.38.20，通过 7 个 Hook 实现完整的 Agent 生命周期追踪：

```
before_dispatch -> 创建根 Trace（每个对话轮次一个 Trace）
  └── before_agent_start -> 打开 Agent Span
        ├── llm_input -> 开启 Generation（捕获 Prompt、系统提示、历史消息）
        ├── llm_output -> 关闭 Generation（记录 Token 用量、缓存命中）
        ├── before_tool_call -> 开启 Tool Span（工具名、参数）
        ├── after_tool_call -> 关闭 Tool Span（结果、耗时、错误）
        └── agent_end -> 关闭 Agent Span，异步 Flush
```

**SDK 配置参数：**

- 批量发送阈值：15 条事件
- 自动 Flush 间隔：5 秒
- 重试次数：2 次，间隔 1 秒
- 错误隔离：所有 Langfuse 操作均包裹在 try-catch 中，不阻塞主流程

### 启用方式

1. 启动 Docker 服务（见上方命令）
2. 修改 `openclaw.json` 中的插件配置：

```json
"langfuse-observer": {
  "enabled": true
}
```

3. 确保环境变量可达：`LANGFUSE_BASE_URL=http://localhost:3000`

**当前禁用原因：** 日志显示网络连接错误 (`LangfuseFetchNetworkError`)，可能是 Docker 服务未启动或系统代理干扰。

---

## 二、Logfire (OTEL) 集成

**状态：已启用，正常运行**

### 配置（openclaw.json）

```json
"diagnostics": {
  "enabled": true,
  "otel": {
    "enabled": true,
    "endpoint": "https://logfire-us.pydantic.dev",
    "serviceName": "lab-openclaw",
    "traces": true,
    "metrics": true,
    "logs": true
  }
}
```

### 架构特点

- **协议：** OpenTelemetry Protocol (OTLP/Protobuf)
- **后端：** Pydantic Logfire 云托管服务（无需本地基础设施）
- **作为内置插件 `diagnostics-otel` 运行**，网关启动时自动加载

### 数据信号

| 信号类型 | 内容 |
|---------|------|
| **Traces** | 分布式链路追踪 |
| **Metrics** | 服务健康指标、请求指标、事件速率 |
| **Logs** | 结构化网关日志（OTLP Log Exporter） |

### 使用方式

无需额外操作，配置启用后自动工作。在 Logfire 控制台中查看 `lab-openclaw` 服务的遥测数据。

---

## 三、功能对比

| 维度 | Langfuse | Logfire / OTEL |
|------|----------|----------------|
| **定位** | LLM 专用可观测性 | 通用系统可观测性 |
| **部署方式** | 自托管 Docker Compose（6 个服务） | 云托管（零基础设施） |
| **数据协议** | Langfuse 私有 SDK (HTTP) | OpenTelemetry 标准协议 (OTLP) |
| **运维成本** | 高（需维护 PG/CH/Redis/MinIO） | 低（仅配置 endpoint） |
| **观测粒度** | Agent 级别（Prompt、Token、工具调用） | 网关/系统级别（请求、指标、日志） |
| **LLM 特化功能** | Generation 追踪、Token 用量、缓存指标、Prompt 管理 | 无 LLM 专用功能 |
| **系统指标** | 仅 Agent 层 | Traces + Metrics + Logs 三合一 |
| **集成方式** | 自定义插件（7 个 Hook） | 内置网关插件（自动注入） |
| **数据主权** | 完全自控（数据在本地） | 数据发送到 Pydantic 云端 |
| **UI/分析** | 自带 Web UI + ClickHouse 分析 | Logfire 云端仪表盘 |
| **当前状态** | 禁用 | 运行中 |

---

## 四、互补关系与建议

两套系统**互补而非重叠**：

- **Logfire/OTEL** 提供系统层面的宏观视角 -- 服务健康、请求链路、错误日志
- **Langfuse** 提供 LLM 层面的微观视角 -- 每次推理的 Prompt、Token 消耗、工具调用详情、缓存效率

如果需要深入调试 Agent 行为（如为什么某次工具调用耗时过长、Token 用量异常），Langfuse 是更合适的工具。如果需要监控整体系统运行状况，Logfire 已覆盖。

### 注意事项

- Langfuse 插件使用的是 SDK v3.38.20（已标记为 deprecated），未来可能需要升级到 v4
- 启用 Langfuse 前需确保 Docker 服务栈正常运行且本地网络无代理干扰
