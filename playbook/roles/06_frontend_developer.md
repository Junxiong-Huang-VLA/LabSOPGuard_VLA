# 角色六：前端开发者（Frontend Developer）

## 职责定位

负责 Vite + Vue/React 前端的功能开发与维护，包括实验工作区、素材时间轴、步骤视图、视频播放器和 PPE 检测面板。

## 核心工作内容

### 主要代码区域
```
frontend-app/
├── src/
│   ├── views/
│   │   ├── ExperimentWorkspace.vue  # 主工作区（顶部导航）
│   │   ├── MaterialTimeline.vue     # 素材时间轴
│   │   ├── StepTimeline.vue         # 步骤时间轴
│   │   └── MaterialSearch.vue       # 素材语义搜索
│   ├── components/
│   │   └── VideoPlayer.vue          # 视频播放器
│   └── utils/
│       └── urls.js                  # resolveArtifactUrl()
```

### 关键注意事项

#### 视频播放器
- `<video src>` 需要绝对 URL
- `activeVideo.url` 是相对路径，必须通过 `resolveArtifactUrl()` 转换：
  ```js
  // 正确
  const url = resolveArtifactUrl(activeVideo.url)  // → http://127.0.0.1:8000/api/v1/...
  // 错误
  const url = activeVideo.url  // 相对路径，播放器无法加载
  ```
- 不能依赖 Vite proxy（前端端口可能是 5173/5174 等任意值）

#### 「生成关键素材」按钮
- 调用 `POST /api/v1/experiments/{id}/materials/publish`
- 自动链式触发：EventPreprocessingEngine → clip → StepBridge
- auto-trigger 条件：`materials/events/` 不存在**或**为空目录

#### 素材命名
- 显示 `display_name` 字段
- 禁止显示 `evidence_summary` 字段（含调试信息 grade=/score=）

### 启动与调试
```bash
cd frontend-app
npm run dev          # 开发模式（通常 5173 或 5174）
npm run build        # 生产构建
npm run preview      # 预览生产构建
```

### API 对接规范
- 所有 API 请求统一走 `http://127.0.0.1:8000`
- 错误处理：展示 `detail` 字段，不暴露 `error_code`
- 长任务（process/publish）轮询任务状态，不等待同步响应

### 关注指标
- 视频播放首帧时间 ≤ 2s
- 素材时间轴渲染（100 条素材）≤ 500ms
- 步骤时间轴正确反映步骤 grade 颜色（绿/黄/红）

## 禁止事项
- 禁止在页面组件中硬编码 `http://127.0.0.1:8000`（统一走 `resolveArtifactUrl`）
- 禁止展示 `evidence_summary` 作为用户可见文字
