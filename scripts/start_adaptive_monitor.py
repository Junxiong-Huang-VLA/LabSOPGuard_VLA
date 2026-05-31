"""
最终修复版自适应监控前端 - 确保所有函数正确定义
"""
import sys
import asyncio
import json
import base64
from pathlib import Path
from datetime import datetime
from typing import Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("启动自适应AI实验室监控系统...")
print("前端Dashboard地址: http://localhost:8080")

try:
    from nicegui import ui
    import cv2
    import numpy as np

    # 导入自适应监控系统
    from src.adaptive_lab_monitor import AdaptiveLabMonitor

    # 全局状态
    class MonitorState:
        def __init__(self):
            self.is_monitoring = False
            self.monitor = AdaptiveLabMonitor("chemistry_lab_001")
            self.frame_count = 0
            self.alerts = []
            self.log_messages = []

    state = MonitorState()

    def add_log(message: str):
        """添加日志"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        state.log_messages.append(f"[{timestamp}] {message}")
        if len(state.log_messages) > 50:
            state.log_messages = state.log_messages[-50:]

    # 定义所有事件处理函数
    async def start_monitoring():
        """开始监控"""
        state.is_monitoring = True
        state.frame_count = 0
        state.alerts.clear()
        add_log("自适应AI监控系统已启动")
        ui.notify('监控系统启动成功', type='positive')

    async def stop_monitoring():
        """停止监控"""
        state.is_monitoring = False
        add_log("自适应AI监控系统已停止")
        ui.notify('监控系统已停止', type='info')

    def reset_stats():
        """重置统计"""
        state.frame_count = 0
        state.alerts.clear()
        state.monitor.violation_history.clear()
        add_log("统计数据已重置")
        ui.notify('统计数据已重置', type='info')

    def clear_alerts():
        """清空告警"""
        state.alerts.clear()
        add_log("告警列表已清空")

    async def process_frame():
        """处理视频帧"""
        if not state.is_monitoring:
            return None

        try:
            # 创建模拟帧
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:] = (50, 50, 50)

            # 模拟实验室环境
            cv2.rectangle(frame, (400, 200), (500, 300), (100, 100, 100), -1)  # 天平
            cv2.rectangle(frame, (200, 100), (350, 400), (80, 80, 80), -1)     # 人员

            # 处理帧
            visual_frame, violations = await state.monitor.process_frame(frame, state.frame_count)
            state.frame_count += 1

            # 处理违规
            for violation in violations:
                alert_data = {
                    "constraint": violation.constraint.description,
                    "severity": violation.constraint.severity.value,
                    "confidence": violation.confidence,
                    "recommendation": violation.recommendation
                }
                state.alerts.insert(0, alert_data)
                add_log(f"违规检测: {violation.constraint.description} [{violation.constraint.severity.value}]")

            # 限制告警数量
            if len(state.alerts) > 20:
                state.alerts = state.alerts[:20]

            # 转换为base64
            _, buffer = cv2.imencode('.jpg', visual_frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')

            return frame_base64

        except Exception as e:
            add_log(f"帧处理错误: {str(e)}")
            return None

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
            .video-container {
                background: #000; border-radius: 8px; overflow: hidden;
                height: 480px; display: flex; align-items: center; justify-content: center;
            }
            .alert-critical { border-left: 4px solid #dc3545; background: #fff5f5; padding: 10px; margin: 5px 0; }
            .alert-major { border-left: 4px solid #fd7e14; background: #fff8f0; padding: 10px; margin: 5px 0; }
            .alert-minor { border-left: 4px solid #ffc107; background: #fffcf0; padding: 10px; margin: 5px 0; }
        </style>
        ''')

        # 页面标题
        with ui.header().classes('header'):
            ui.label('自适应AI实验室监控系统').classes('text-h4 font-bold')
            ui.space()
            ui.label('v2.0 - 智能约束学习').classes('text-sm')

        # 主内容
        with ui.row().classes('p-4 w-full'):
            # 左侧控制面板
            with ui.column().classes('w-1/3'):
                # 系统状态
                with ui.card().classes('w-full'):
                    ui.label('系统状态').classes('text-h6 font-bold mb-4')
                    monitor_status = ui.label('未启动').classes('status-warning text-lg')
                    frame_count_label = ui.label('帧数: 0').classes('')
                    alert_count_label = ui.label('告警: 0').classes('status-danger')

                # 控制按钮
                with ui.card().classes('w-full'):
                    ui.label('控制面板').classes('text-h6 font-bold mb-4')
                    ui.button('开始监控', icon='play', color='positive', on_click=start_monitoring).classes('btn-margin')
                    ui.button('停止监控', icon='stop', color='negative', on_click=stop_monitoring).classes('btn-margin')
                    ui.button('重置统计', icon='refresh', on_click=reset_stats).classes('btn-margin')

                # 操作日志
                with ui.card().classes('w-full'):
                    ui.label('实时日志').classes('text-h6 font-bold mb-4')
                    log_area = ui.column().classes('w-full h-40 overflow-y-auto')

            # 中间视频区域
            with ui.column().classes('w-1/3'):
                with ui.card().classes('w-full'):
                    ui.label('实时监控画面').classes('text-h6 font-bold mb-4')
                    video_display = ui.html('''
                    <div class="video-container">
                        <div style="text-align: center; color: #fff;">
                            <div style="font-size: 48px; margin-bottom: 10px;">🎥</div>
                            <div style="font-size: 18px;">等待开始监控</div>
                        </div>
                    </div>
                    ''')

            # 右侧告警面板
            with ui.column().classes('w-1/3'):
                # 告警统计
                with ui.card().classes('w-full'):
                    ui.label('告警统计').classes('text-h6 font-bold mb-4')
                    with ui.grid(columns=4).classes('w-full gap-2'):
                        ui.label('严重').classes('text-center text-danger')
                        ui.label('重要').classes('text-center text-warning')
                        ui.label('轻微').classes('text-center text-info')
                        ui.label('警告').classes('text-center text-primary')
                        critical_count = ui.label('0').classes('text-center')
                        major_count = ui.label('0').classes('text-center')
                        minor_count = ui.label('0').classes('text-center')
                        warning_count = ui.label('0').classes('text-center')

                # 告警列表
                with ui.card().classes('w-full'):
                    ui.label('实时告警').classes('text-h6 font-bold mb-4')
                    alert_list = ui.column().classes('w-full h-48 overflow-y-auto')
                    ui.button('清空告警', icon='clear', on_click=clear_alerts).classes('mt-2 w-full')

        # 底部状态栏
        with ui.footer().classes('bg-gray-800 text-white'):
            with ui.row().classes('w-full items-center'):
                ui.label('系统状态: ').classes('font-bold')
                footer_status = ui.label('就绪').classes('status-good')
                ui.space()
                ui.label('实验室: chemistry_lab_001').classes('')
                ui.space()
                last_update = ui.label(datetime.now().strftime('%H:%M:%S'))

        # 定时更新函数
        async def auto_update():
            # 处理新帧
            if state.is_monitoring:
                frame_base64 = await process_frame()
                if frame_base64:
                    video_display.set_content(f'''
                    <div class="video-container">
                        <img src="data:image/jpeg;base64,{frame_base64}" style="max-width: 100%; max-height: 100%;" alt="监控画面">
                    </div>
                    ''')

            # 更新状态显示
            if state.is_monitoring:
                monitor_status.set_text('运行中')
                monitor_status.classes('status-good text-lg')
                footer_status.set_text('监控中')
                footer_status.classes('status-good')
            else:
                monitor_status.set_text('未启动')
                monitor_status.classes('status-warning text-lg')
                footer_status.set_text('就绪')
                footer_status.classes('status-good')

            # 更新统计
            frame_count_label.set_text(f'帧数: {state.frame_count}')
            alert_count_label.set_text(f'告警: {len(state.alerts)}')

            # 更新告警统计
            severity_counts = {'Critical': 0, 'Major': 0, 'Minor': 0, 'Warning': 0}
            for alert in state.alerts:
                severity = alert.get('severity', 'Warning')
                severity_counts[severity] = severity_counts.get(severity, 0) + 1

            critical_count.set_text(str(severity_counts['Critical']))
            major_count.set_text(str(severity_counts['Major']))
            minor_count.set_text(str(severity_counts['Minor']))
            warning_count.set_text(str(severity_counts['Warning']))

            # 更新告警列表
            alert_list.clear()
            with alert_list:
                for alert in state.alerts[:10]:  # 显示最近10条
                    severity = alert.get('severity', 'Warning').lower()
                    with ui.card().classes(f'alert-{severity}'):
                        ui.label(alert.get('constraint', '未知违规')).classes('font-bold text-sm')
                        ui.label(f"等级: {alert.get('severity', 'Warning')}").classes('text-xs text-gray-600')
                        ui.label(alert.get('recommendation', '')).classes('text-xs')

            # 更新日志
            log_area.clear()
            with log_area:
                for log_msg in state.log_messages[-10:]:
                    ui.label(log_msg).classes('text-sm')

            # 更新时间
            last_update.set_text(datetime.now().strftime('%H:%M:%S'))

        # 启动定时更新
        ui.timer(2.0, auto_update)  # 每2秒更新一次

    if __name__ == "__main__":
        ui.run(
            title="自适应AI实验室监控系统",
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