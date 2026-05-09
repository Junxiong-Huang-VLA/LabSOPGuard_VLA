"""
修复版前端服务 - 添加完整的按钮功能和后端连接
"""
import sys
import asyncio
import json
from pathlib import Path
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("启动NiceGUI前端服务...")
print("前端Dashboard地址: http://localhost:8080")

try:
    from nicegui import ui
    import httpx

    # 全局状态
    class AppState:
        def __init__(self):
            self.is_monitoring = False
            self.camera_id = "camera_001"
            self.video_source = "0"
            self.alerts = []
            self.backend_url = "http://localhost:8000"

    app_state = AppState()

    # 后端API客户端
    async def api_call(method: str, endpoint: str, data: dict = None):
        """调用后端API"""
        try:
            async with httpx.AsyncClient() as client:
                url = f"{app_state.backend_url}{endpoint}"
                if method.upper() == "GET":
                    response = await client.get(url, timeout=5.0)
                elif method.upper() == "POST":
                    response = await client.post(url, data=data, timeout=5.0)
                else:
                    return {"error": f"不支持的请求方法: {method}"}

                if response.status_code == 200:
                    return response.json()
                else:
                    return {"error": f"API调用失败: {response.status_code}"}
        except Exception as e:
            return {"error": f"连接失败: {str(e)}"}

    @ui.page('/')
    def dashboard():
        ui.add_head_html('''
        <style>
            .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; }
            .card { background: white; border-radius: 10px; padding: 20px; margin: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            .status-good { color: #28a745; }
            .status-warning { color: #ffc107; }
            .status-danger { color: #dc3545; }
            .btn-margin { margin: 5px; }
            .video-placeholder {
                background: #f8f9fa; border: 2px dashed #dee2e6; border-radius: 8px;
                height: 300px; display: flex; align-items: center; justify-content: center;
            }
            .alert-item {
                border-left: 4px solid #dc3545;
                background: #fff5f5;
                padding: 10px;
                margin: 5px 0;
                border-radius: 4px;
            }
        </style>
        ''')

        # 页面标题
        with ui.header().classes('header'):
            ui.label('实验室SOP合规智能监控系统').classes('text-h4 font-bold')
            ui.space()
            status_label = ui.label('v1.0.0').classes('text-sm')

        # 主内容区域
        with ui.row().classes('p-4 w-full'):
            # 左侧控制面板
            with ui.column().classes('w-1/2'):
                # 系统状态卡片
                with ui.card().classes('w-full'):
                    ui.label('系统状态').classes('text-h6 font-bold mb-4')

                    with ui.row().classes('w-full'):
                        with ui.column().classes('w-1/3 text-center'):
                            ui.label('后端API').classes('text-sm text-gray-600')
                            backend_status = ui.label('检查中...').classes('text-lg font-bold')

                        with ui.column().classes('w-1/3 text-center'):
                            ui.label('监控状态').classes('text-sm text-gray-600')
                            monitor_status = ui.label('未启动').classes('text-lg font-bold')

                        with ui.column().classes('w-1/3 text-center'):
                            ui.label('今日告警').classes('text-sm text-gray-600')
                            alert_count = ui.label('0').classes('text-lg font-bold')

                # 控制面板
                with ui.card().classes('w-full'):
                    ui.label('控制面板').classes('text-h6 font-bold mb-4')

                    # 摄像头配置
                    with ui.row().classes('w-full mb-4'):
                        ui.label('摄像头ID:').classes('font-bold')
                        camera_input = ui.input(value='camera_001', placeholder='输入摄像头ID').classes('w-32')
                        ui.label('视频源:').classes('font-bold ml-4')
                        video_input = ui.input(value='0', placeholder='0=USB, rtsp://...').classes('w-48')

                    # 控制按钮
                    with ui.row():
                        start_btn = ui.button('启动监控', icon='play', color='positive', on_click=start_monitoring).classes('btn-margin')
                        stop_btn = ui.button('停止监控', icon='stop', color='negative', on_click=stop_monitoring).classes('btn-margin')
                        report_btn = ui.button('生成报告', icon='description', on_click=generate_report).classes('btn-margin')
                        refresh_btn = ui.button('刷新状态', icon='refresh', on_click=refresh_status).classes('btn-margin')

                # 操作日志
                with ui.card().classes('w-full'):
                    ui.label('操作日志').classes('text-h6 font-bold mb-4')
                    log_area = ui.column().classes('w-full h-32 overflow-y-auto')

            # 右侧监控区域
            with ui.column().classes('w-1/2'):
                # 实时监控区域
                with ui.card().classes('w-full'):
                    ui.label('实时监控').classes('text-h6 font-bold mb-4')

                    video_placeholder = ui.html('''
                    <div class="video-placeholder">
                        <div style="text-align: center; color: #6c757d;">
                            <div style="font-size: 48px; margin-bottom: 10px;">📹</div>
                            <div style="font-size: 18px;">视频监控画面</div>
                            <div style="font-size: 14px;">点击"启动监控"开始实时监控</div>
                        </div>
                    </div>
                    ''')

                    # 监控信息
                    with ui.row().classes('w-full mt-2'):
                        ui.label('当前帧:').classes('font-bold')
                        frame_info = ui.label('0').classes('ml-2')
                        ui.label('检测延迟:').classes('font-bold ml-4')
                        latency_info = ui.label('0ms').classes('ml-2')

                # 告警列表
                with ui.card().classes('w-full'):
                    ui.label('实时告警').classes('text-h6 font-bold mb-4')

                    alert_list = ui.column().classes('w-full h-48 overflow-y-auto')

                    # 清空告警按钮
                    ui.button('清空告警', icon='clear', on_click=clear_alerts).classes('mt-2')

        # 底部状态栏
        with ui.footer().classes('bg-gray-800 text-white'):
            with ui.row().classes('w-full items-center'):
                ui.label('系统状态: ').classes('font-bold')
                footer_status = ui.label('正常运行').classes('status-good')
                ui.space()
                ui.label('最后更新: ').classes('font-bold')
                last_update = ui.label(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        # 按钮事件处理函数
        async def start_monitoring():
            """启动监控"""
            app_state.camera_id = camera_input.value
            app_state.video_source = video_input.value

            add_log(f"正在启动监控: 摄像头={app_state.camera_id}, 视频源={app_state.video_source}")

            result = await api_call("POST", "/api/v1/streams/start", {
                "camera_id": app_state.camera_id,
                "video_source": app_state.video_source
            })

            if "error" not in result:
                app_state.is_monitoring = True
                monitor_status.set_text('运行中')
                monitor_status.classes('status-good text-lg font-bold')
                start_btn.disable()
                stop_btn.enable()
                add_log("监控启动成功")
                ui.notify('监控启动成功', type='positive')
            else:
                add_log(f"监控启动失败: {result['error']}")
                ui.notify(f"启动失败: {result['error']}", type='negative')

        async def stop_monitoring():
            """停止监控"""
            add_log(f"正在停止监控: 摄像头={app_state.camera_id}")

            result = await api_call("POST", "/api/v1/streams/stop", {
                "camera_id": app_state.camera_id
            })

            if "error" not in result:
                app_state.is_monitoring = False
                monitor_status.set_text('已停止')
                monitor_status.classes('status-warning text-lg font-bold')
                start_btn.enable()
                stop_btn.disable()
                add_log("监控停止成功")
                ui.notify('监控已停止', type='info')
            else:
                add_log(f"停止监控失败: {result['error']}")
                ui.notify(f"停止失败: {result['error']}", type='negative')

        async def generate_report():
            """生成报告"""
            add_log("正在生成报告...")

            if not app_state.is_monitoring:
                ui.notify('请先启动监控', type='warning')
                return

            # 这里可以调用后端的报告生成API
            ui.notify('报告生成功能开发中', type='info')
            add_log("报告生成功能开发中")

        async def refresh_status():
            """刷新状态"""
            add_log("正在刷新系统状态...")

            # 检查后端连接
            result = await api_call("GET", "/")
            if "error" not in result:
                backend_status.set_text('正常')
                backend_status.classes('status-good text-lg font-bold')
                add_log("后端连接正常")
            else:
                backend_status.set_text('异常')
                backend_status.classes('status-danger text-lg font-bold')
                add_log(f"后端连接异常: {result['error']}")

            # 刷新视频流状态
            streams_result = await api_call("GET", "/api/v1/streams/status")
            if "error" not in streams_result:
                active_streams = len(streams_result)
                alert_count.set_text(str(active_streams))
                add_log(f"当前活跃流: {active_streams}")

            last_update.set_text(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        def add_log(message: str):
            """添加日志"""
            timestamp = datetime.now().strftime('%H:%M:%S')
            with log_area:
                ui.label(f"[{timestamp}] {message}").classes('text-sm')

        def clear_alerts():
            """清空告警"""
            alert_list.clear()
            app_state.alerts.clear()
            alert_count.set_text('0')
            add_log("告警列表已清空")

        # 定时刷新状态
        ui.timer(5.0, refresh_status)

        # 初始化
        ui.timer(1.0, refresh_status, once=True)

    if __name__ == "__main__":
        ui.run(
            title="实验室SOP合规智能监控系统",
            host="0.0.0.0",
            port=8080,
            reload=False,
            show=False
        )

except Exception as e:
    print(f"前端启动失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)