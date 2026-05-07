# LabSOPGuard 多路视频无线传输方案报告

**日期：** 2026-04-19  
**范围：** 多路摄像无线链路、带宽预算、低延迟传输、断流恢复、安全隔离、与 LabSOPGuard 实时采集链路对接  
**结论：** 推荐采用“摄像头/边缘编码器 -> Wi-Fi 6/6E/专网 5G -> 有线回程网关 -> LabSOPGuard 工作站”的混合架构。无线只承担最后一段移动/布线困难链路，核心计算和长期存储仍放在有线稳定节点。

---

## 一、目标与边界

无线传输的核心目标是让多机位摄像头在实验台、移动设备、临时布控场景下快速接入，同时不牺牲同步、索引和证据回切能力。

| 目标 | 说明 | 优先级 |
|---|---|---|
| 多路低延迟接入 | 4-8 路 1080p 视频可稳定进入 LabSOPGuard | P0 |
| 可回切证据片段 | 无线流进入系统后仍可边录边索引 | P0 |
| 时间同步可解释 | 每路视频必须保留 offset、anchor、drift 信息 | P0 |
| 断流可恢复 | 弱网下自动重连，缺口可标记 | P1 |
| 安全隔离 | 摄像头网络与办公网/公网隔离 | P1 |
| 移动部署 | 支持临时实验台、移动机器人、手持视角 | P2 |

---

## 二、推荐网络架构

### 2.1 总体链路

```text
多路摄像头 / 手机 / 工业相机
  -> 本地编码 H.264/H.265
  -> Wi-Fi 6/6E / 专网 5G / Mesh 无线链路
  -> 实验台网关或边缘盒子
  -> 有线千兆/2.5G/10G 回程
  -> LabSOPGuard 工作站
      -> upload/stream 注册
      -> 边录边索引
      -> 多机位同步
      -> material_index.sqlite 检索
```

### 2.2 为什么不建议“全无线直连工作站”

| 方案 | 问题 | 建议 |
|---|---|---|
| 所有摄像头直接连工作站热点 | 信号、带宽、驱动、漫游都不可控 | 只用于 demo |
| 每路相机独立连办公 Wi-Fi | 信道拥塞，安全和 QoS 不可控 | 不建议生产使用 |
| 无线摄像头 -> 专用 AP -> 有线工作站 | 稳定性和可运维性最好 | 推荐 |
| 无线摄像头 -> 边缘盒子 -> RTSP 转发 | 可本地缓存、断点续传 | 推荐高可靠场景 |

---

## 三、无线方案对比

| 方案 | 典型带宽 | 典型延迟 | 优点 | 风险 | 适用场景 |
|---|---|---|---|---|---|
| Wi-Fi 6 5GHz | 300Mbps-1Gbps 实际共享 | 20-80ms | 成本低、部署快 | 受干扰，穿墙衰减 | 固定实验台、同房间 |
| Wi-Fi 6E 6GHz | 500Mbps-1.5Gbps 实际共享 | 10-50ms | 干扰少、频段干净 | 设备兼容性要求高 | 高密度多机位 |
| Wi-Fi 7 | 1Gbps+ 实际共享 | 低 | 高吞吐、低延迟 | 成本高，设备生态仍在演进 | 未来升级 |
| 5G CPE | 50-300Mbps | 40-120ms | 远距离、移动部署 | 公网抖动、上行不稳定 | 临时外场 |
| 专网 5G | 可规划 | 20-60ms | 可控、安全、覆盖强 | 建设成本高 | 大型实验空间 |
| 无线 HDMI | 点对点高码率 | 低 | 画质好、部署简单 | 不适合多路索引和网络管理 | 单路监看 |
| Mesh Wi-Fi | 共享带宽 | 抖动较大 | 覆盖范围大 | 多跳后延迟和丢包升高 | 非关键辅助机位 |

---

## 四、带宽预算

### 4.1 单路码率参考

| 分辨率/帧率 | H.264 码率 | H.265 码率 | 备注 |
|---|---|---|---|
| 720p30 | 2-4 Mbps | 1-2.5 Mbps | 低成本监控 |
| 1080p30 | 4-8 Mbps | 2-5 Mbps | 推荐默认 |
| 1080p60 | 8-16 Mbps | 5-10 Mbps | 快速手部动作 |
| 4K30 | 20-40 Mbps | 10-25 Mbps | 精细读数/标签 |

### 4.2 多路预算

| 机位数量 | 1080p30 H.264 | 1080p30 H.265 | 建议网络 |
|---|---|---|---|
| 2 路 | 8-16 Mbps | 4-10 Mbps | 普通 Wi-Fi 6 可用 |
| 4 路 | 16-32 Mbps | 8-20 Mbps | 专用 Wi-Fi 6 AP |
| 8 路 | 32-64 Mbps | 16-40 Mbps | Wi-Fi 6E/有线回程 |
| 12 路 | 48-96 Mbps | 24-60 Mbps | 多 AP 分信道 + 2.5G/10G 回程 |

实际规划时建议按理论码率的 2-3 倍预留无线容量，因为无线链路会受到重传、干扰、漫游和突发码率影响。

---

## 五、传输协议选择

| 协议 | 优点 | 缺点 | 建议 |
|---|---|---|---|
| RTSP | 摄像头支持广，OpenCV/FFmpeg 易接入 | 弱网下可能卡顿 | 默认推荐 |
| RTMP | 推流生态成熟 | 延迟通常高于 RTSP | 直播网关可用 |
| SRT | 抗丢包、适合公网 | 相机原生支持少 | 边缘盒子转发推荐 |
| WebRTC | 低延迟强 | 服务端集成复杂 | 需要浏览器预览时使用 |
| UDP/RTP | 低延迟 | 丢包处理需自建 | 可用于受控内网 |
| HLS | 兼容好 | 延迟高 | 不推荐实时检测主链 |

推荐组合：

```text
固定 IP 摄像头：RTSP over TCP
移动/弱网场景：相机 -> 边缘盒子 -> SRT/RTSP 转发
浏览器预览：后端 RTSP ingest -> WebRTC/HLS preview
```

---

## 六、无线传输与时间同步

无线链路会引入抖动，但不能把网络到达时间当作真实拍摄时间。生产链路必须区分：

| 时间 | 含义 | 是否可作为同步主依据 |
|---|---|---|
| capture timestamp | 相机采集帧的时间 | 可以 |
| hardware timecode | 相机/采集卡硬件时间 | 最推荐 |
| sync board event | 同步板触发事件 | 推荐 |
| flash/audio anchor | 画面或声音里的共同事件 | 可用 |
| network receive time | 工作站收到帧的时间 | 不推荐 |

推荐同步策略：

```text
每路视频保留本地 timestamp
  -> 开始时触发一次闪光/蜂鸣/同步板脉冲
  -> 长视频中每 5-10 分钟追加 anchor
  -> 结束时再触发一次 anchor
  -> TimeSyncCalibrator 拟合 offset + drift
  -> material_stream 使用 global timestamp
```

---

## 七、稳定性设计

### 7.1 断流与弱网处理

| 问题 | 影响 | 处理 |
|---|---|---|
| 瞬时丢包 | 画面花屏、帧缺失 | RTSP over TCP 或 SRT |
| AP 漫游 | 数秒中断 | 固定 AP，禁用自动漫游 |
| 信道拥塞 | 延迟和卡顿 | 专用 SSID、固定信道、6GHz 优先 |
| 摄像头掉线 | 素材流缺口 | 自动重连并记录 gap event |
| 工作站负载高 | 解码积压 | 降帧、队列背压、边缘转码 |

### 7.2 网络配置建议

| 项目 | 建议 |
|---|---|
| SSID | 摄像头专用 SSID，不与办公网混用 |
| 频段 | 优先 6GHz，其次 5GHz；避免 2.4GHz 主链路 |
| 回程 | AP 到工作站必须有线，建议 2.5G 或以上 |
| IP | 摄像头静态 DHCP 绑定 |
| 安全 | WPA3、设备白名单、VLAN 隔离 |
| QoS | 视频流 DSCP/QoS 优先级 |
| 监控 | 每路记录 FPS、bitrate、packet loss、reconnect count |

---

## 八、与 LabSOPGuard 的对接

### 8.1 注册无线摄像头流

```json
POST /api/v1/experiments/{experiment_id}/upload/stream
{
  "source": "rtsp://10.20.30.41/live/main",
  "source_type": "rtsp",
  "camera_id": "cam_wireless_front",
  "sync_group": "bench_01",
  "capture_duration_sec": 300.0,
  "sync_method": "sync_board",
  "sync_board_offset_sec": 0.012,
  "clock_drift_ppm": 80.0
}
```

### 8.2 检索无线机位素材

```text
GET /api/v1/experiments/{id}/materials/search
  ?camera_id=cam_wireless_front
  &objects=pipette,tube
  &actions=transfer
  &start_time_sec=30
  &end_time_sec=90
  &clip_exists=true
  &text=liquid
```

### 8.3 输出链路

```text
无线视频流
  -> recordings/*.mp4
  -> clips/*.mp4
  -> material_stream.json
  -> preprocessing.json
  -> material_index.sqlite
  -> API 检索 / 报告证据 / 回放
```

---

## 九、测试方法

### 9.1 单机位无线测试

| 测试项 | 方法 | 通过标准 |
|---|---|---|
| 连续采集 | 单路 RTSP 运行 30 分钟 | 无崩溃，缺口可记录 |
| 延迟 | LED 计时器画面 + 工作站显示对比 | 平均 < 500ms，P95 < 1s |
| 帧率 | 统计实际写入 FPS | 不低于目标 FPS 的 95% |
| 回切 | 随机选 10 个时间点生成 clip | 10/10 成功 |

### 9.2 多机位无线测试

| 测试项 | 方法 | 通过标准 |
|---|---|---|
| 4 路并发 | 4 路 1080p30 同时采集 30 分钟 | 无持续掉帧 |
| 8 路并发 | 8 路 1080p30 同时采集 30 分钟 | CPU/GPU/网络不打满 |
| 同步误差 | 多机位同时拍摄闪光锚点 | 误差进入目标阈值 |
| 断流恢复 | 手动断开一路 10s 后恢复 | 自动重连，gap 可见 |
| 带宽压力 | 增加背景流量 | 系统降级但不崩溃 |

---

## 十、现存差距与后续优化

| 问题 | 当前状态 | 后续处理 |
|---|---|---|
| 无线链路监控尚未产品化 | 采集链路有基础输出，但无线质量指标未统一入库 | 增加 stream health table |
| 自动重连策略需要增强 | 目前依赖基础读取逻辑 | 增加 per-camera reconnect worker |
| SRT/WebRTC 未作为正式入口 | RTSP/HTTP/USB 优先 | 增加协议 adapter |
| 多 AP 信道规划未自动化 | 依赖人工部署 | 增加部署模板和验收脚本 |
| 真实硬件时间码读取未接 SDK | 字段和同步模型已准备 | 接入相机/采集卡 SDK |

---

## 十一、推荐落地配置

| 场景 | 推荐配置 |
|---|---|
| 2-4 路 demo | Wi-Fi 6 AP + RTSP H.264 1080p30 + 手动/闪光 anchor |
| 4-8 路实验台 | 专用 Wi-Fi 6E AP + 有线 2.5G 回程 + H.265 + 同步板 |
| 8 路以上 | 多 AP 分区 + 10G 上联 + 边缘盒子转码/缓存 |
| 移动机器人视角 | 5G/Wi-Fi 6 + 边缘缓存 + SRT/RTSP 转发 |
| 高精度同步实验 | 有线优先；无线仅传输，时间以硬件时间码/同步板为准 |

---

## 十二、相关文件

| 文件 | 作用 |
|---|---|
| `backend/main.py` | 注册无线视频流、查询素材索引 |
| `src/experiment/service.py` | 读取网络流、抽帧、录制、同步和输出 |
| `src/labsopguard/time_sync.py` | offset/drift/anchor 校准 |
| `src/labsopguard/stream_buffer.py` | 环形缓存和历史 clip 回切 |
| `src/labsopguard/retrieval.py` | 按机位、时间、对象、动作、clip 联合检索 |
| `docs/workspace_consolidation.md` | 项目整合和运行目录说明 |

