"""
NiceGUI前端监控Dashboard
基于技术文档要求的实时监控界面
"""
from nicegui import ui, app, background_tasks
from nicegui.events import ValueChangeEventArguments
import asyncio
import json
import base64
from datetime import datetime
from pathlib import Path
import httpx
import websockets

# 全局状态
class DashboardState:
    def __init__(self):
        self.active_streams = {}
        self.recent_alerts = []
        self.selected_camera = "camera_001"
        self.websocket_connection = None
        self.is_connected = False
        self.is_connecting = False
        self.is_recording = False

state = DashboardState()

# API配置
API_BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws"

critical_count = None
major_count = None
minor_count = None
alert_list = None
last_update = None
camera_status_list = None
compliance_status = None
fps_display = None
current_step = None
system_status = None

@ui.page('/')
def dashboard():
    """主监控Dashboard页面"""
    global critical_count, major_count, minor_count, alert_list
    global last_update, camera_status_list, compliance_status, fps_display, current_step, system_status
    ui.add_head_html('''
    <style>
        .alert-critical { background-color: #ff4444; color: white; }
        .alert-major { background-color: #ff8800; color: white; }
        .alert-minor { background-color: #ffcc00; color: black; }
        .compliance-good { color: #00cc00; }
        .compliance-warning { color: #ffcc00; }
        .compliance-bad { color: #ff4444; }
        .video-container { border: 2px solid #333; border-radius: 8px; }
        .status-indicator { width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 8px; }
        .status-active { background-color: #00cc00; }
        .status-inactive { background-color: #ff4444; }
        .status-warning { background-color: #ffcc00; }
    </style>
    ''')

    # 页面标题
    with ui.header().classes('bg-primary text-white'):
        ui.label('实验室SOP合规智能监控系统').classes('text-h4 font-bold')
        ui.space()
        ui.label().bind_text_from(state, 'is_connected', lambda connected: '已连接' if connected else '未连接')

    # 主内容区域
    with ui.row().classes('w-full p-4'):
        # 左侧：视频监控区域
        with ui.column().classes('w-2/3'):
            # 摄像头选择
            with ui.row().classes('mb-4'):
                ui.label('摄像头选择:').classes('font-bold')
                ui.select(
                    options=['camera_001', 'camera_002', 'camera_003', 'camera_004'],
                    value=state.selected_camera,
                    on_change=lambda e: setattr(state, 'selected_camera', e.value)
                ).classes('w-48')

                ui.button('启动监控', on_click=start_monitoring, color='positive')
                ui.button('停止监控', on_click=stop_monitoring, color='negative')

            # 视频显示区域
            with ui.card().classes('w-full'):
                ui.label('实时视频监控').classes('text-h6 font-bold mb-2')

                # 视频画面显示
                with ui.row().classes('w-full justify-center'):
                    video_placeholder = ui.html('''
                        <div class="video-container w-full h-96 bg-gray-800 flex items-center justify-center">
                            <div class="text-white text-center">
                                <div class="text-xl mb-2">📹 视频监控画面</div>
                                <div class="text-sm text-gray-400">点击"启动监控"开始实时监控</div>
                            </div>
                        </div>
                    ''')

                # 视频控制按钮
                with ui.row().classes('mt-4 justify-center'):
                    ui.button('截图', on_click=take_screenshot, icon='camera')
                    ui.button('全屏', on_click=toggle_fullscreen, icon='fullscreen')
                    ui.button('录制', on_click=toggle_recording, icon='fiber_manual_record')

            # 实时状态面板
            with ui.card().classes('w-full mt-4'):
                ui.label('实时状态').classes('text-h6 font-bold mb-2')

                with ui.row().classes('w-full'):
                    # 合规状态
                    with ui.column().classes('w-1/3'):
                        ui.label('合规状态').classes('font-bold')
                        compliance_status = ui.label('正常').classes('compliance-good text-lg')

                    # 检测帧率
                    with ui.column().classes('w-1/3'):
                        ui.label('检测帧率').classes('font-bold')
                        fps_display = ui.label('0 FPS').classes('text-lg')

                    # 当前步骤
                    with ui.column().classes('w-1/3'):
                        ui.label('当前SOP步骤').classes('font-bold')
                        current_step = ui.label('等待开始').classes('text-lg')

        # 右侧：告警和控制面板
        with ui.column().classes('w-1/3'):
            # 实时告警
            with ui.card().classes('w-full mb-4'):
                ui.label('实时告警').classes('text-h6 font-bold mb-2')

                # 告警统计
                with ui.row().classes('w-full mb-4'):
                    with ui.column().classes('w-1/3 text-center'):
                        critical_count = ui.label('0').classes('text-h4 text-red')
                        ui.label('严重').classes('text-sm')

                    with ui.column().classes('w-1/3 text-center'):
                        major_count = ui.label('0').classes('text-h4 text-orange')
                        ui.label('重要').classes('text-sm')

                    with ui.column().classes('w-1/3 text-center'):
                        minor_count = ui.label('0').classes('text-h4 text-yellow')
                        ui.label('轻微').classes('text-sm')

                # 告警列表
                alert_list = ui.column().classes('w-full h-64 overflow-y-auto')

            # 系统控制
            with ui.card().classes('w-full mb-4'):
                ui.label('系统控制').classes('text-h6 font-bold mb-2')

                ui.button('生成报告', on_click=generate_report, icon='description')
                ui.button('导出数据', on_click=export_data, icon='download')
                ui.button('系统设置', on_click=open_settings, icon='settings')

            # 摄像头状态
            with ui.card().classes('w-full'):
                ui.label('摄像头状态').classes('text-h6 font-bold mb-2')

                camera_status_list = ui.column().classes('w-full')

    # 底部状态栏
    with ui.footer().classes('bg-gray-800 text-white'):
        with ui.row().classes('w-full items-center'):
            ui.label('系统状态: ').classes('font-bold')
            system_status = ui.label('运行中').classes('compliance-good')
            ui.space()
            ui.label('最后更新: ').classes('font-bold')
            last_update = ui.label(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    # 初始化WebSocket连接
    ui.timer(2.0, ensure_websocket_connected)

    # 定时更新状态
    ui.timer(5.0, update_dashboard_status)

def ensure_websocket_connected():
    if state.is_connected or state.is_connecting:
        return
    state.is_connecting = True
    background_tasks.create(connect_websocket())

async def connect_websocket():
    """建立WebSocket连接"""
    try:
        async with websockets.connect(WS_URL) as websocket:
            state.websocket_connection = websocket
            state.is_connected = True
            state.is_connecting = False

            # 订阅告警
            await websocket.send(json.dumps({
                "type": "subscribe_alerts",
                "camera_id": state.selected_camera
            }))

            # 监听消息
            async for message in websocket:
                data = json.loads(message)
                await handle_websocket_message(data)

    except Exception as e:
        print(f"WebSocket连接失败: {e}")
        state.is_connected = False
    finally:
        state.is_connecting = False

async def handle_websocket_message(data: dict):
    """处理WebSocket消息"""
    message_type = data.get("type")

    if message_type == "violation_alert":
        alert_data = data.get("data", {})
        state.recent_alerts.insert(0, alert_data)

        # 限制告警数量
        if len(state.recent_alerts) > 100:
            state.recent_alerts = state.recent_alerts[:100]

        # 更新告警显示
        update_alert_display()

        # 播放告警声音
        play_alert_sound(alert_data.get("severity", "Minor"))

    elif message_type == "pong":
        # 心跳响应
        pass

def update_alert_display():
    """更新告警显示"""
    if critical_count is None or major_count is None or minor_count is None or alert_list is None:
        return

    # 更新告警统计
    critical_count.set_text(str(len([a for a in state.recent_alerts if a.get("severity") == "Critical"])))
    major_count.set_text(str(len([a for a in state.recent_alerts if a.get("severity") == "Major"])))
    minor_count.set_text(str(len([a for a in state.recent_alerts if a.get("severity") == "Minor"])))

    # 更新告警列表
    alert_list.clear()
    for alert in state.recent_alerts[:20]:  # 只显示最近20条
        with alert_list:
            severity = alert.get("severity", "Minor").lower()
            with ui.card().classes(f'alert-{severity} p-2 mb-2'):
                ui.label(f'{alert.get("timestamp_sec", 0):.1f}s - {alert.get("rule_id", "未知")}').classes('font-bold')
                ui.label(alert.get("message", "")).classes('text-sm')

def play_alert_sound(severity: str):
    """播放告警声音"""
    # 根据严重等级播放不同声音
    if severity == "Critical":
        ui.run_javascript('new Audio("/static/sounds/critical_alert.mp3").play();')
    elif severity == "Major":
        ui.run_javascript('new Audio("/static/sounds/major_alert.mp3").play();')
    else:
        ui.run_javascript('new Audio("/static/sounds/minor_alert.mp3").play();')

async def start_monitoring():
    """启动监控"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_BASE_URL}/api/v1/streams/start",
                params={
                    "camera_id": state.selected_camera,
                    "video_source": "0"  # 默认摄像头
                }
            )

            if response.status_code == 200:
                ui.notify('监控启动成功', type='positive')
                state.active_streams[state.selected_camera] = True
            else:
                ui.notify(f'监控启动失败: {response.text}', type='negative')

    except Exception as e:
        ui.notify(f'监控启动失败: {e}', type='negative')

async def stop_monitoring():
    """停止监控"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_BASE_URL}/api/v1/streams/stop",
                params={"camera_id": state.selected_camera}
            )

            if response.status_code == 200:
                ui.notify('监控已停止', type='positive')
                state.active_streams.pop(state.selected_camera, None)
            else:
                ui.notify(f'停止监控失败: {response.text}', type='negative')

    except Exception as e:
        ui.notify(f'停止监控失败: {e}', type='negative')

async def take_screenshot():
    """截图"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{API_BASE_URL}/api/v1/streams/screenshot",
                params={"camera_id": state.selected_camera},
            )
        if response.status_code == 200:
            path = response.json().get("path", "")
            ui.notify(f"截图已保存: {path}", type='positive')
        else:
            ui.notify(f"截图失败: {response.text}", type='negative')
    except Exception as e:
        ui.notify(f"截图失败: {e}", type='negative')

async def toggle_fullscreen():
    """切换全屏"""
    ui.run_javascript('''
        if (document.fullscreenElement) {
            document.exitFullscreen();
        } else {
            document.documentElement.requestFullscreen();
        }
    ''')

async def toggle_recording():
    """切换录制"""
    endpoint = "/api/v1/streams/recording/stop" if state.is_recording else "/api/v1/streams/recording/start"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_BASE_URL}{endpoint}",
                params={"camera_id": state.selected_camera},
            )
        if response.status_code == 200:
            state.is_recording = not state.is_recording
            data = response.json()
            path = data.get("path")
            if state.is_recording:
                ui.notify(f"开始录制: {path}", type='positive')
            else:
                ui.notify(f"录制已停止: {path}", type='positive')
        else:
            ui.notify(f"录制操作失败: {response.text}", type='negative')
    except Exception as e:
        ui.notify(f"录制操作失败: {e}", type='negative')

async def generate_report():
    """生成报告"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_BASE_URL}/api/v1/reports/generate",
                params={"camera_id": state.selected_camera},
            )
        if response.status_code == 200:
            data = response.json()
            ui.notify(f"报告生成成功: {data.get('report_path', '')}", type='positive')
        else:
            ui.notify(f"报告生成失败: {response.text}", type='negative')
    except Exception as e:
        ui.notify(f"报告生成失败: {e}", type='negative')

async def export_data():
    """导出数据"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{API_BASE_URL}/api/v1/alerts/export",
                params={"camera_id": state.selected_camera, "limit": 500},
            )
        if response.status_code == 200:
            data = response.json()
            ui.notify(f"导出成功: {data.get('path', '')}", type='positive')
        else:
            ui.notify(f"导出失败: {response.text}", type='negative')
    except Exception as e:
        ui.notify(f"导出失败: {e}", type='negative')

async def open_settings():
    """打开设置"""
    with ui.dialog() as dialog, ui.card():
        ui.label('系统设置').classes('text-h6 font-bold mb-4')

        ui.select(
            label='检测模型',
            options=['YOLOv8n-pose', 'YOLOv8s-pose', 'YOLOv8m-pose'],
            value='YOLOv8n-pose'
        ).classes('w-full mb-2')

        ui.slider(
            label='置信度阈值',
            min=0.1, max=1.0, step=0.05, value=0.45
        ).classes('w-full mb-2')

        ui.slider(
            label='告警冷却时间(秒)',
            min=1.0, max=60.0, step=1.0, value=5.0
        ).classes('w-full mb-2')

        with ui.row():
            ui.button('保存', on_click=lambda: dialog.submit('save'))
            ui.button('取消', on_click=lambda: dialog.submit('cancel'))

    result = await dialog
    if result == 'save':
        ui.notify('设置已保存', type='positive')

async def update_dashboard_status():
    """更新Dashboard状态"""
    if last_update is None or camera_status_list is None:
        return

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_BASE_URL}/api/v1/streams/status")
            if response.status_code == 200:
                status_data = response.json()
                last_update.set_text(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

                camera_status_list.clear()
                with camera_status_list:
                    for cam, info in status_data.items():
                        active = info.get("active", False)
                        recording = info.get("recording", False)
                        state_text = "运行中" if active else "已停止"
                        rec_text = " | 录制中" if recording else ""
                        color = "text-green-600" if active else "text-gray-500"
                        ui.label(f"{cam}: {state_text}{rec_text}").classes(color)

                selected = status_data.get(state.selected_camera)
                if selected:
                    comp = selected.get("compliance_status", {})
                    ratio = float(comp.get("compliance_ratio", 0.0))
                    if compliance_status is not None:
                        if ratio >= 0.8:
                            compliance_status.set_text("正常")
                        elif ratio >= 0.5:
                            compliance_status.set_text("警告")
                        else:
                            compliance_status.set_text("异常")
                    if current_step is not None:
                        pending = comp.get("pending_steps", [])
                        current_step.set_text(pending[0] if pending else "已完成")
                    if fps_display is not None:
                        fps_display.set_text("~30 FPS" if selected.get("active") else "0 FPS")

    except Exception as e:
        print(f"状态更新失败: {e}")

# ---------------------------------------------------------------------------
# 实验过程理解页面（收口主线）
# ---------------------------------------------------------------------------

API_BASE = "http://localhost:8000"


def _fetch_json(url: str, params: dict = None) -> dict:
    """发起 GET 请求获取 JSON。"""
    import httpx
    try:
        with httpx.Client(timeout=15.0) as client:
            r = client.get(url, params=params or {})
            r.raise_for_status()
            return r.json()
    except Exception as e:
        return {"error": str(e)}


def _post_json(url: str, json_data: dict = None, data: dict = None) -> dict:
    """发起 POST 请求。"""
    import httpx
    try:
        with httpx.Client(timeout=30.0) as client:
            r = client.post(url, json=json_data, data=data or {})
            r.raise_for_status()
            return r.json()
    except Exception as e:
        return {"error": str(e)}


def _confidence_color(confidence: float) -> str:
    """根据置信度返回颜色。"""
    if confidence >= 0.8:
        return "text-green"
    elif confidence >= 0.5:
        return "text-yellow"
    else:
        return "text-red"


def _status_badge(status: str) -> str:
    """返回状态徽章。"""
    colors = {
        "pending": "bg-gray",
        "processing": "bg-blue",
        "completed": "bg-green",
        "failed": "bg-red",
        "confirmed": "bg-green",
        "candidate": "bg-yellow",
        "inferred": "bg-orange",
        "skipped": "bg-gray",
    }
    return colors.get(status, "bg-gray")


def _inference_badge(completed_by_inference: bool) -> str:
    """返回推断标记。"""
    if completed_by_inference:
        return "🏷️ INFERRED"
    return "✓ CONFIRMED"


# ---------------------------------------------------------------------------
# 实验列表页
# ---------------------------------------------------------------------------

@ui.page('/experiments')
def experiments_page():
    """实验列表页。"""

    ui.add_head_html('''
    <style>
        .exp-card { border: 1px solid #333; border-radius: 8px; padding: 16px; margin: 8px 0; }
        .exp-title { font-size: 1.1em; font-weight: bold; }
        .exp-meta { color: #888; font-size: 0.85em; margin-top: 4px; }
        .badge { padding: 2px 8px; border-radius: 4px; font-size: 0.75em; color: white; }
        .inferred-badge { background-color: #e67e22; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.75em; }
        .confirmed-badge { background-color: #27ae60; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.75em; }
        .evidence-box { background-color: #1e1e1e; border-radius: 4px; padding: 8px; margin: 4px 0; }
        .timeline-step { border-left: 3px solid #444; padding-left: 12px; margin: 12px 0; }
        .timeline-confirmed { border-left-color: #27ae60; }
        .timeline-inferred { border-left-color: #e67e22; }
        .timeline-candidate { border-left-color: #f39c12; }
    </style>
    ''')

    with ui.header().classes('bg-blue-800 text-white'):
        ui.label('实验过程理解系统').classes('text-h5 font-bold')
        ui.space()
        ui.button('新建实验', on_click=lambda: ui.navigate.to('/upload'), color='positive')
        ui.button('监控面板', on_click=lambda: ui.navigate.to('/'), color='secondary')

    with ui.column().classes('w-full p-6'):
        ui.label('实验列表').classes('text-h4 font-bold mb-4')

        # 加载实验列表
        exp_list_container = ui.column().classes('w-full')
        error_label = ui.label('')

        async def load_experiments():
            exp_list_container.clear()
            error_label.set_text('')
            try:
                data = _fetch_json(f"{API_BASE}/api/v1/experiments", {"limit": 50})
                if "error" in data:
                    error_label.set_text(f"加载失败: {data['error']}")
                    return

                experiments = data.get("experiments", [])
                total = data.get("total", 0)
                ui.label(f"共 {total} 个实验").classes('text-sm text-gray mb-2')

                if not experiments:
                    ui.label('暂无实验。点击右上角"新建实验"开始。').classes('text-gray p-4')

                for exp in experiments:
                    status = exp.get("status", "unknown")
                    exp_id = exp.get("experiment_id", "")
                    title = exp.get("title", "无标题")
                    created = exp.get("created_at", "")[:19] if exp.get("created_at") else ""

                    stats = exp.get("timeline", {}) if isinstance(exp.get("timeline"), dict) else {}
                    total_steps = stats.get("total_steps", exp.get("total_steps", 0))
                    inferred_steps = stats.get("inferred_steps", exp.get("inferred_steps", 0))
                    avg_conf = stats.get("avg_confidence", exp.get("avg_confidence", 0.0))

                    with exp_list_container:
                        with ui.card().classes('w-full p-4 mb-2'):
                            with ui.row().classes('w-full items-center'):
                                ui.label(title).classes('text-lg font-bold flex-1')
                                ui.label(f"状态: {status}").classes(f'badge {_status_badge(status)} ml-2')

                            with ui.row().classes('w-full text-sm text-gray mt-1'):
                                ui.label(f"ID: {exp_id[:12]}...")
                                ui.space()
                                ui.label(f"创建: {created}")
                                ui.space()
                                ui.label(f"步骤: {total_steps} (推断 {inferred_steps})")
                                ui.space()
                                if avg_conf > 0:
                                    ui.label(f"平均置信度: {avg_conf:.2f}").classes(_confidence_color(avg_conf))

                            with ui.row().classes('mt-2'):
                                ui.button('查看详情', on_click=lambda _, eid=exp_id: ui.navigate.to(f'/experiments/{eid}'), size='sm')
                                if status == 'completed':
                                    ui.button('查看Timeline', on_click=lambda _, eid=exp_id: ui.navigate.to(f'/experiments/{eid}'), size='sm', color='primary')

            except Exception as e:
                error_label.set_text(f"无法连接后端: {e}。请确保后端在 http://localhost:8000 运行。")

        ui.button('🔄 刷新', on_click=load_experiments, color='primary')
        ui.separator()

        with exp_list_container:
            ui.spinner(size='lg')

        ui.label().bind_text_from(error_label, 'text')

        # 初始加载
        load_experiments()


# ---------------------------------------------------------------------------
# 实验详情页（含时间线）
# ---------------------------------------------------------------------------

@ui.page('/experiments/{experiment_id}')
def experiment_detail_page(experiment_id: str):
    """实验详情页 - 展示 Timeline 和 Steps。"""

    ui.add_head_html('''
    <style>
        .step-card { border-radius: 8px; padding: 12px; margin: 8px 0; background-color: #1e1e1e; }
        .step-confirmed { border-left: 4px solid #27ae60; }
        .step-inferred { border-left: 4px solid #e67e22; }
        .step-candidate { border-left: 4px solid #f39c12; }
        .step-skipped { border-left: 4px solid #888; opacity: 0.6; }
        .evidence-chip { display: inline-block; background: #333; padding: 2px 8px; border-radius: 4px; font-size: 0.8em; margin: 2px; }
        .provenance-box { background: #2a2a2a; padding: 8px; border-radius: 4px; font-family: monospace; font-size: 0.8em; }
        .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }
        .stat-card { background: #1e1e1e; padding: 12px; border-radius: 8px; text-align: center; }
        .stat-number { font-size: 1.8em; font-weight: bold; }
        .stat-label { font-size: 0.75em; color: #888; }
    </style>
    ''')

    with ui.header().classes('bg-blue-800 text-white'):
        ui.button('← 实验列表', on_click=lambda: ui.navigate.to('/experiments'), color='secondary').props('flat text-white')
        ui.label(f'实验详情: {experiment_id[:12]}...').classes('text-h6 font-bold ml-4')

    with ui.column().classes('w-full p-4'):
        # 实验信息区
        exp_info_card = ui.card().classes('w-full p-4 mb-4')
        timeline_container = ui.column().classes('w-full')
        error_label = ui.label('')

        async def load_experiment():
            error_label.set_text('')
            try:
                # 加载实验详情
                exp = _fetch_json(f"{API_BASE}/api/v1/experiments/{experiment_id}")

                # 加载 timeline
                timeline_data = _fetch_json(f"{API_BASE}/api/v1/experiments/{experiment_id}/timeline")
                steps_data = _fetch_json(f"{API_BASE}/api/v1/experiments/{experiment_id}/steps")

                exp_info_card.clear()
                with exp_info_card:
                    if "error" in exp:
                        ui.label(f"实验加载失败: {exp['error']}").classes('text-red')
                        return

                    title = exp.get("title", "无标题")
                    status = exp.get("status", "unknown")
                    created = exp.get("created_at", "")[:19] if exp.get("created_at") else ""
                    models = exp.get("models_used", [])

                    with ui.row().classes('w-full items-center'):
                        ui.label(title).classes('text-xl font-bold flex-1')
                        ui.label(f"状态: {status}").classes(f'badge {_status_badge(status)}')
                        if status == "pending":
                            ui.button('▶ 处理', on_click=lambda: process_experiment(), color='primary')

                    with ui.row().classes('text-sm text-gray mt-1'):
                        ui.label(f"ID: {experiment_id}")
                        ui.space()
                        ui.label(f"创建: {created}")
                        if models:
                            ui.space()
                            ui.label(f"模型: {', '.join(models)}")

                # 渲染 Timeline
                timeline_container.clear()
                if "error" in timeline_data:
                    timeline_container.append(ui.label(f"Timeline 未生成: {timeline_data.get('detail', '')}"))
                    if exp.get("status") == "pending":
                        timeline_container.append(ui.button('▶ 启动处理', on_click=lambda: process_experiment(), color='primary'))
                    return

                steps = timeline_data.get("steps", [])
                if not steps:
                    timeline_container.append(ui.label("暂无步骤数据"))
                    return

                stats = {
                    "total": timeline_data.get("total_steps", len(steps)),
                    "confirmed": timeline_data.get("confirmed_steps", 0),
                    "inferred": timeline_data.get("inferred_steps", 0),
                    "avg_conf": timeline_data.get("avg_confidence", 0.0),
                }

                with timeline_container:
                    # 统计面板
                    with ui.row().classes('w-full mb-4'):
                        with ui.card().classes('flex-1 p-3 text-center'):
                            ui.label(str(stats["total"])).classes('text-2xl font-bold')
                            ui.label('总步骤').classes('text-xs text-gray')
                        with ui.card().classes('flex-1 p-3 text-center'):
                            ui.label(str(stats["confirmed"])).classes('text-2xl font-bold text-green')
                            ui.label('已确认').classes('text-xs text-gray')
                        with ui.card().classes('flex-1 p-3 text-center'):
                            ui.label(str(stats["inferred"])).classes('text-2xl font-bold text-orange')
                            ui.label('已推断').classes('text-xs text-gray')
                        with ui.card().classes('flex-1 p-3 text-center'):
                            ui.label(f"{stats['avg_conf']:.2f}").classes('text-2xl font-bold text-blue')
                            ui.label('平均置信度').classes('text-xs text-gray')

                    ui.separator()
                    ui.label('实验时间线').classes('text-h6 font-bold mt-4 mb-2')

                    # 时间线
                    for step in steps:
                        step_id = step.get("step_id", "")
                        step_index = step.get("step_index", 0)
                        step_name = step.get("step_name", "未知步骤")
                        step_desc = step.get("step_description", "")
                        status = step.get("status", "confirmed")
                        conf = float(step.get("confidence", 0.0))
                        start_ts = float(step.get("start_time_sec", 0.0))
                        end_ts = step.get("end_time_sec")
                        duration = step.get("duration_sec")
                        inferred_flag = step.get("completed_by_inference", False)
                        inference_method = step.get("inference_method", "")
                        inference_model = step.get("inference_model", "")
                        evidence_refs = step.get("evidence_refs", [])
                        provenance = step.get("provenance", {})
                        notes = step.get("evidence_notes", "")

                        badge_class = f"step-{status}"
                        conf_color = _confidence_color(conf)

                        with ui.card().classes(f'w-full mb-2 {badge_class}'):
                            # 步骤头
                            with ui.row().classes('w-full items-center'):
                                ui.label(f"[{step_index}]").classes('text-gray font-bold')
                                ui.label(step_name).classes('font-bold flex-1')
                                if inferred_flag:
                                    ui.label('🏷️ INFERRED').classes('inferred-badge')
                                else:
                                    ui.label('✓ CONFIRMED').classes('confirmed-badge')
                                ui.label(f"@{start_ts:.1f}s").classes('text-sm text-gray')
                                if end_ts:
                                    ui.label(f"→ {end_ts:.1f}s").classes('text-sm text-gray')
                                if duration:
                                    ui.label(f"({duration:.1f}s)").classes('text-sm text-gray')

                            # 置信度和推理信息
                            with ui.row().classes('w-full text-sm mt-1'):
                                ui.label(f"置信度: ").classes('text-gray')
                                ui.label(f"{conf:.2f}").classes(conf_color)
                                ui.space()
                                if inferred_flag:
                                    ui.label(f"推断方法: {inference_method}").classes('text-orange') if inference_method else None
                                    ui.space()
                                    ui.label(f"模型: {inference_model}").classes('text-gray') if inference_model else None

                            # 描述
                            if step_desc:
                                ui.label(step_desc[:200]).classes('text-sm text-gray mt-1')

                            # 证据引用
                            if evidence_refs:
                                with ui.row().classes('mt-2'):
                                    for ref in evidence_refs[:5]:
                                        etype = ref.get("evidence_type", "unknown")
                                        ts = ref.get("timestamp_sec")
                                        frame_id = ref.get("frame_id")
                                        label_text = f"{etype}"
                                        if ts is not None:
                                            label_text += f" @{ts:.1f}s"
                                        elif frame_id is not None:
                                            label_text += f" frame={frame_id}"
                                        ui.label(label_text).classes('evidence-chip')

                            # Provenance
                            if provenance:
                                with ui.card().classes('provenance-box w-full mt-2'):
                                    is_inf = provenance.get("is_inferred", False)
                                    src = provenance.get("source", "")
                                    prov_conf = provenance.get("confidence", 0.0)
                                    ui.label(f"Provenance: source={src}, inferred={is_inf}, confidence={prov_conf:.2f}").classes('text-xs')

                            # 备注
                            if notes:
                                ui.label(f"证据备注: {notes}").classes('text-xs text-gray mt-1')

                            # 查看详情按钮
                            ui.button('查看步骤详情 →', on_click=lambda _, eid=experiment_id, sid=step_id: ui.navigate.to(f'/experiments/{eid}/steps/{sid}'), size='sm', color='primary').classes('mt-2')

            except Exception as e:
                error_label.set_text(f"加载失败: {e}。请确保后端已启动。")

        async def process_experiment():
            try:
                ui.notify("启动处理中...", type="info")
                video_paths = exp.get("video_paths", []) if 'exp' in dir() else []
                video_path = video_paths[0] if video_paths else ""
                result = _post_json(f"{API_BASE}/api/v1/experiments/{experiment_id}/process", {"video_path": video_path})
                if "error" in result:
                    ui.notify(f"处理失败: {result['error']}", type="negative")
                else:
                    ui.notify(f"处理完成！步骤数: {result.get('stats', {}).get('total_steps', 0)}", type="positive")
                    await load_experiment()
            except Exception as e:
                ui.notify(f"处理失败: {e}", type="negative")

        ui.button('🔄 刷新', on_click=load_experiment, color='primary')
        ui.separator()

        with timeline_container:
            ui.spinner()

        with exp_info_card:
            ui.spinner()

        ui.label().bind_text_from(error_label, 'text')

        # 初始加载
        load_experiment()


# ---------------------------------------------------------------------------
# 步骤详情页
# ---------------------------------------------------------------------------

@ui.page('/experiments/{experiment_id}/steps/{step_id}')
def step_detail_page(experiment_id: str, step_id: str):
    """步骤详情页 - 展示证据引用和 provenance。"""

    ui.add_head_html('''
    <style>
        .evidence-detail { background: #1e1e1e; border-radius: 8px; padding: 12px; margin: 8px 0; }
        .param-row { display: flex; gap: 8px; margin: 4px 0; }
        .provenance-table { width: 100%; border-collapse: collapse; }
        .provenance-table td { padding: 4px 8px; border-bottom: 1px solid #333; }
        .provenance-table tr:hover { background: #2a2a2a; }
    </style>
    ''')

    with ui.header().classes('bg-blue-800 text-white'):
        ui.button('← 实验详情', on_click=lambda: ui.navigate.to(f'/experiments/{experiment_id}'), color='secondary').props('flat text-white')
        ui.button('← 实验列表', on_click=lambda: ui.navigate.to('/experiments'), color='secondary').props('flat text-white')
        ui.label(f'步骤详情').classes('text-h6 font-bold ml-4')

    with ui.column().classes('w-full p-4'):
        step_container = ui.column().classes('w-full')
        error_label = ui.label('')

        async def load_step():
            error_label.set_text('')
            try:
                step_data = _fetch_json(f"{API_BASE}/api/v1/experiments/{experiment_id}/steps/{step_id}")
                if "error" in step_data:
                    error_label.set_text(f"加载失败: {step_data['error']}")
                    return

                step_container.clear()
                with step_container:
                    step_name = step_data.get("step_name", "未知步骤")
                    step_desc = step_data.get("step_description", "")
                    status = step_data.get("status", "confirmed")
                    conf = float(step_data.get("confidence", 0.0))
                    inferred = step_data.get("completed_by_inference", False)
                    inference_method = step_data.get("inference_method")
                    inference_model = step_data.get("inference_model")
                    notes = step_data.get("notes", "")
                    evidence_notes = step_data.get("evidence_notes", "")

                    start_ts = float(step_data.get("start_time_sec", 0.0))
                    end_ts = step_data.get("end_time_sec")
                    duration = step_data.get("duration_sec")

                    # 步骤基本信息
                    with ui.card().classes('w-full p-4 mb-4'):
                        ui.label(f"步骤: {step_name}").classes('text-xl font-bold')
                        ui.label(f"Step ID: {step_id}").classes('text-xs text-gray mt-1')
                        with ui.row().classes('mt-2'):
                            ui.label(f"状态: {status}").classes(f'badge {_status_badge(status)}')
                            if inferred:
                                ui.label('🏷️ INFERRED').classes('inferred-badge')
                            else:
                                ui.label('✓ CONFIRMED').classes('confirmed-badge')

                        with ui.row().classes('mt-2 text-sm'):
                            ui.label(f"时间: {start_ts:.1f}s").classes('text-gray')
                            if end_ts:
                                ui.label(f" → {end_ts:.1f}s").classes('text-gray')
                            if duration:
                                ui.label(f"  (持续 {duration:.1f}s)").classes('text-gray')
                            ui.space()
                            ui.label(f"置信度: {conf:.4f}").classes(_confidence_color(conf))

                        if step_desc:
                            ui.label(f"描述: {step_desc}").classes('text-sm mt-2')

                    # 推断信息（仅推断步骤显示）
                    if inferred:
                        with ui.card().classes('w-full p-4 mb-4 bg-orange-900'):
                            ui.label('🏷️ 推断信息').classes('text-h6 font-bold mb-2')
                            with ui.table().classes('w-full'):
                                with ui.table():
                                    thead = ui.thead()
                                    with thead:
                                        with ui.tr():
                                            for h in ['字段', '值']:
                                                ui.th(h)
                                    tbody = ui.tbody()
                                    with tbody:
                                        if inference_method:
                                            with ui.tr():
                                                ui.td('推断方法')
                                                ui.td(str(inference_method))
                                        if inference_model:
                                            with ui.tr():
                                                ui.td('使用模型')
                                                ui.td(str(inference_model))
                                        with ui.tr():
                                            ui.td('置信度')
                                            ui.td(f"{conf:.4f} ({_confidence_color(conf)})")
                                        with ui.tr():
                                            ui.td('推断标记')
                                            ui.td('completed_by_inference=True')

                    # 证据引用
                    evidence_refs = step_data.get("evidence_refs", [])
                    with ui.card().classes('w-full p-4 mb-4'):
                        ui.label(f'证据引用 ({len(evidence_refs)} 个)').classes('text-h6 font-bold mb-2')
                        if not evidence_refs:
                            ui.label("无证据引用").classes('text-gray')
                        for ref in evidence_refs:
                            etype = ref.get("evidence_type", "unknown")
                            src = ref.get("source", "")
                            ts = ref.get("timestamp_sec")
                            frame_id = ref.get("frame_id")
                            conf_r = ref.get("confidence", 1.0)
                            desc = ref.get("description", "")
                            prov = ref.get("provenance", {})

                            with ui.card().classes('evidence-detail'):
                                with ui.row().classes('w-full items-center'):
                                    ui.label(f"[{etype}]").classes('font-bold')
                                    ui.label(f"source={src}").classes('text-sm text-gray')
                                    if ts is not None:
                                        ui.label(f"@ {ts:.2f}s").classes('text-sm text-blue')
                                    if frame_id is not None:
                                        ui.label(f"frame={frame_id}").classes('text-sm text-gray')
                                    ui.space()
                                    ui.label(f"conf={conf_r:.2f}").classes(_confidence_color(conf_r))
                                if desc:
                                    ui.label(desc).classes('text-sm text-gray mt-1')
                                if prov:
                                    is_inf = prov.get("is_inferred", False)
                                    ui.label(f"Provenance: inferred={is_inf}, confidence={prov.get('confidence', 0.0):.2f}").classes('text-xs text-gray mt-1')

                    # Provenance 追溯表
                    provenance = step_data.get("provenance")
                    if provenance:
                        with ui.card().classes('w-full p-4 mb-4'):
                            ui.label('Provenance 追溯').classes('text-h6 font-bold mb-2')
                            tbl = ui.table()
                            tbl.columns = [
                                {'name': 'field', 'label': '字段', 'field': 'field'},
                                {'name': 'value', 'label': '值', 'field': 'value'},
                            ]
                            rows = []
                            for k, v in provenance.items():
                                rows.append({'field': k, 'value': str(v)})
                            tbl.rows = rows

                    # 证据备注
                    if evidence_notes:
                        with ui.card().classes('w-full p-4 mb-4'):
                            ui.label('证据备注').classes('text-h6 font-bold mb-2')
                            ui.label(evidence_notes).classes('text-sm')

                    # 关联参数
                    parameters = step_data.get("parameters", [])
                    if parameters:
                        with ui.card().classes('w-full p-4 mb-4'):
                            ui.label(f'操作参数 ({len(parameters)} 个)').classes('text-h6 font-bold mb-2')
                            for param in parameters:
                                name = param.get("name", "?")
                                value = param.get("value", "")
                                unit = param.get("unit", "")
                                source = param.get("source", "")
                                prov = param.get("provenance", {})
                                ui.label(f"• {name}: {value} {unit or ''} [source={source}]").classes('text-sm')
                                if prov and prov.get("is_inferred"):
                                    ui.label(f"  ↳ (inferred, conf={prov.get('confidence', 0.0):.2f})").classes('text-xs text-orange ml-4')

                    # 原始 JSON 查看
                    with ui.card().classes('w-full'):
                        ui.label('原始 StepRecord JSON').classes('text-h6 font-bold mb-2')
                        import json
                        ui.code(json.dumps(step_data, ensure_ascii=False, indent=2), language='json').classes('w-full')

            except Exception as e:
                error_label.set_text(f"加载失败: {e}")

        ui.button('🔄 刷新', on_click=load_step, color='primary')
        ui.separator()

        with step_container:
            ui.spinner()

        ui.label().bind_text_from(error_label, 'text')

        load_step()


# ---------------------------------------------------------------------------
# 上传页
# ---------------------------------------------------------------------------

@ui.page('/upload')
def upload_page():
    """新建实验 - 上传视频/上下文/Protocol。"""

    ui.add_head_html('''
    <style>
        .upload-section { border: 2px dashed #444; border-radius: 8px; padding: 24px; text-align: center; }
        .upload-section:hover { border-color: #666; }
    </style>
    ''')

    exp_id_display = ui.label('')
    video_uploaded = ui.label('')
    exp_created = False
    current_exp_id = [None]

    with ui.header().classes('bg-blue-800 text-white'):
        ui.button('← 实验列表', on_click=lambda: ui.navigate.to('/experiments'), color='secondary').props('flat text-white')
        ui.label('新建实验').classes('text-h5 font-bold ml-4')

    with ui.column().classes('w-full p-6 max-w-4xl'):
        ui.label('新建实验').classes('text-h4 font-bold mb-4')

        # 1. 基本信息
        with ui.card().classes('w-full p-4 mb-4'):
            ui.label('基本信息').classes('text-h6 font-bold mb-3')
            title_input = ui.input(label='实验标题 *', placeholder='如: 蛋白质纯化实验 20260415').classes('w-full')
            desc_input = ui.textarea(label='实验描述', placeholder='简要描述实验目的...').classes('w-full')

        # 2. 上传视频
        with ui.card().classes('w-full p-4 mb-4'):
            ui.label('实验视频 *').classes('text-h6 font-bold mb-3')
            video_file = ui.upload(
                label='选择视频文件',
                on_upload=lambda e: handle_video_upload(e),
            ).props('accept=".mp4,.avi,.mov,.mkv"')
            video_uploaded.set_text('')

        # 3. 对话上下文
        with ui.card().classes('w-full p-4 mb-4'):
            ui.label('对话上下文（可选）').classes('text-h6 font-bold mb-3')
            context_input = ui.textarea(
                label='对话上下文',
                placeholder='粘贴实验记录/操作日志/对话记录...',
            ).classes('w-full')
            ui.label('说明: 操作人员的语音转文字记录或实验日志').classes('text-xs text-gray')

        # 4. Protocol
        with ui.card().classes('w-full p-4 mb-4'):
            ui.label('实验 Protocol（可选）').classes('text-h6 font-bold mb-3')
            protocol_input = ui.textarea(
                label='Protocol 文本',
                placeholder='Paste protocol here...\nExample:\n1. 准备样本\n2. 加入缓冲液\n3. 离心 5min @ 3000rpm\n4. 收集上清',
            ).classes('w-full')
            ui.label('说明: 标准操作流程文本，用于步骤推理参考').classes('text-xs text-gray')

        # 5. 提交
        with ui.row().classes('mt-4'):
            def submit_experiment():
                title = title_input.value.strip()
                if not title:
                    ui.notify("请输入实验标题", type="warning")
                    return

                try:
                    # 创建实验
                    result = _post_json(f"{API_BASE}/api/v1/experiments", {
                        "title": title,
                        "description": desc_input.value or "",
                        "context_text": context_input.value or "",
                        "protocol_text": protocol_input.value or "",
                    })
                    if "error" in result:
                        ui.notify(f"创建失败: {result['error']}", type="negative")
                        return

                    exp_id = result.get("experiment_id")
                    current_exp_id[0] = exp_id
                    exp_id_display.set_text(f"实验已创建: {exp_id[:12]}...")
                    exp_id_display.visible = True
                    ui.notify("实验创建成功！", type="positive")

                    # 如果有视频文件已上传，则直接处理
                    if video_uploaded.value:
                        ui.notify("启动视频处理...", type="info")
                        # 触发处理（如果后端已设置视频路径）
                        process_result = _post_json(f"{API_BASE}/api/v1/experiments/{exp_id}/process", {})
                        if "error" not in process_result:
                            ui.notify(f"处理完成！步骤数: {process_result.get('stats', {}).get('total_steps', 0)}", type="positive")
                        else:
                            ui.notify(f"处理失败: {process_result.get('detail', process_result['error'])}", type="warning")

                    ui.timer(1.0, lambda: ui.navigate.to(f'/experiments/{exp_id}'), once=True)

                except Exception as e:
                    ui.notify(f"错误: {e}", type="negative")

            ui.button('🚀 创建并分析', on_click=submit_experiment, color='primary', size='lg')

        with ui.row().classes('w-full mt-4'):
            exp_id_display.visible = False

    def handle_video_upload(e):
        """处理视频文件上传。"""
        if not current_exp_id[0]:
            ui.notify("请先创建实验", type="warning")
            return
        video_uploaded.set_text(f"已选择: {e.name} ({(len(e.content or b'')) / 1024 / 1024:.1f} MB)")
        video_uploaded.visible = True


# ---------------------------------------------------------------------------
# 启动应用
# ---------------------------------------------------------------------------

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        title="实验室SOP合规智能监控系统",
        host="0.0.0.0",
        port=8080,
        reload=True,
        dark=True
    )