"""
集成自适应监控到Web前端 - 增强版Dashboard
支持实时视频流、可视化检测框和智能告警
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

print("启动增强版前端服务...")
print("前端Dashboard地址: http://localhost:8080")

try:
    from nicegui import ui
    import httpx
    import cv2
    import numpy as np

    # 导入自适应监控系统
    from src.adaptive_lab_monitor import AdaptiveLabMonitor, ViolationAlert

    # 全局状态
    class EnhancedAppState:
        def __init__(self):
            self.is_monitoring = False
            self.camera_id = "camera_001"
            self.video_source = "0"
            self.alerts = []
            self.backend_url = "http://localhost:8000"
            self.log_messages = []
            self.monitor: Optional[AdaptiveLabMonitor] = None
            self.frame_count = 0
            self.current_violations = []
            self.statistics = {
                "total_frames": 0,
                "total_violations": 0,
                "detection_rate": 0.0
            }

    app_state = EnhancedAppState()

    # 初始化监控系统
    app_state.monitor = AdaptiveLabMonitor("chemistry_lab_001")

    def add_log(message: str):
        """添加日志"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        app_state.log_messages.append(f"[{timestamp}] {message}")
        if len(app_state.log_messages) > 100:
            app_state.log_messages = app_state.log_messages[-100:]

    async def process_video_frame():
        """处理视频帧"""
        if not app_state.is_monitoring:
            return

        try:
            # 创建模拟帧（实际应用中会从摄像头获取）
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:] = (50, 50, 50)

            # 模拟实验室环境
            cv2.rectangle(frame, (400, 200), (500, 300), (100, 100, 100), -1)  # 天平
            cv2.rectangle(frame, (200, 100), (350, 400), (80, 80, 80), -1)     # 人员

            # 处理帧
            visual_frame, violations = await app_state.monitor.process_frame(
                frame, app_state.frame_count
            )

            # 更新状态
            app_state.frame_count += 1
            app_state.current_violations = violations
            app_state.statistics["total_frames"] = app_state.frame_count
            app_state.statistics["total_violations"] += len(violations)

            # 处理新违规
            for violation in violations:
                alert_data = {
                    "id": violation.alert_id,
                    "timestamp": violation.timestamp.isoformat(),
                    "constraint": violation.constraint.description,
                    "severity": violation.constraint.severity.value,
                    "confidence": violation.confidence,
                    "recommendation": violation.recommendation,
                    "description": violation.description
                }
                app_state.alerts.insert(0, alert_data)

                # 记录日志
                add_log(f"违规检测: {violation.constraint.description} [{violation.constraint.severity.value}]")

            # 限制告警数量
            if len(app_state.alerts) > 50:
                app_state.alerts = app_state.alerts[:50]

            # 将帧转换为base64用于显示
            _, buffer = cv2.imencode('.jpg', visual_frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')

            return frame_base64, violations

        except Exception as e:
            add_log(f"帧处理错误: {str(e)}")
            return None, []

    @ui.page('/')
    def enhanced_dashboard():
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
            .video-frame { max-width: 100%; max-height: 100%; }
            .alert-item {
                border-left: 4px solid; padding: 10px; margin: 5px 0;
                background: #f8f9fa; border-radius: 4px;
            }
            .alert-critical { border-left-color: #dc3545; background: #fff5f5; }
            .alert-major { border-left-color: #fd7e14; background: #fff8f0; }
            .alert-minor { border-left-color: #ffc107; background: #fffcf0; }
            .alert-warning { border-left-color: #17a2b8; background: #f0f9ff; }
            .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
            .stat-item { text-align: center; padding: 10px; background: #f8f9fa; border-radius: 5px; }
            .constraint-list { max-height: 200px; overflow-y: auto; }
            .constraint-item { padding: 5px; margin: 2px 0; background: #e9ecef; border-radius: 3px; }
        </style>
        ''')

        # 页面标题
        with ui.header().classes('header'):
            ui.label('自适应AI实验室监控系统').classes('text-h4 font-bold')
            ui.space()
            ui.label('v2.0 - 智能约束学习').classes('text-sm')

        # 主内容区域
        with ui.row().classes('p-4 w-full'):
            # 左侧控制面板
            with ui.column().classes('w-1/3'):
                # 系统状态
                with ui.card().classes('w-full'):
                    ui.label('系统状态').classes('text-h6 font-bold mb-4')

                    with ui.grid(columns=2).classes('w-full gap-2'):
                        ui.label('监控状态:').classes('font-bold')
                        monitor_status = ui.label('未启动').classes('status-warning')

                        ui.label('帧数:').classes('font-bold')
                        frame_count = ui.label('0').classes('')

                        ui.label('违规次数:').classes('font-bold')
                        violation_count = ui.label('0').classes('status-danger')

                        ui.label('活跃约束:').classes('font-bold')
                        constraint_count = ui.label('5').classes('')

                # 控制面板
                with ui.card().classes('w-full'):
                    ui.label('控制面板').classes('text-h6 font-bold mb-4')

                    # 监控控制
                    with ui.row():
                        start_btn = ui.button('开始监控', icon='play', color='positive',
                                            on_click=start_enhanced_monitoring).classes('btn-margin')
                        stop_btn = ui.button('停止监控', icon='stop', color='negative',
                                           on_click=stop_enhanced_monitoring).classes('btn-margin')

                    # 系统控制
                    with ui.row():
                        reset_btn = ui.button('重置统计', icon='refresh', on_click=reset_statistics).classes('btn-margin')
                        export_btn = ui.button('导出报告', icon='download', on_click=export_report).classes('btn-margin')

                # 约束管理
                with ui.card().classes('w-full'):
                    ui.label('自适应约束').classes('text-h6 font-bold mb-4')

                    constraint_list = ui.column().classes('constraint-list')
                    update_constraint_display(constraint_list)

                # 操作日志
                with ui.card().classes('w-full'):
                    ui.label('实时日志').classes('text-h6 font-bold mb-4')
                    log_area = ui.column().classes('w-full h-32 overflow-y-auto')

            # 中间视频区域
            with ui.column().classes('w-1/3'):
                # 实时视频
                with ui.card().classes('w-full'):
                    ui.label('实时监控画面').classes('text-h6 font-bold mb-4')

                    video_display = ui.html('''
                    <div class="video-container">
                        <div style="text-align: center; color: #fff;">
                            <div style="font-size: 48px; margin-bottom: 10px;">🎥</div>
                            <div style="font-size: 18px;">等待开始监控</div>
                            <div style="font-size: 14px; color: #ccc;">点击"开始监控"启动实时检测</div>
                        </div>
                    </div>
                    ''')

                    # 视频信息
                    with ui.row().classes('w-full mt-2'):
                        ui.label('检测延迟:').classes('font-bold')
                        latency_info = ui.label('0ms').classes('ml-2')
                        ui.label('FPS:').classes('font-bold ml-4')
                        fps_info = ui.label('0').classes('ml-2')

                # 统计图表
                with ui.card().classes('w-full'):
                    ui.label('实时统计').classes('text-h6 font-bold mb-4')

                    with ui.grid(columns=4).classes('w-full gap-2'):
                        with ui.column().classes('stat-item text-center'):
                            ui.label('总帧数').classes('text-sm text-gray-600')
                            stat_frames = ui.label('0').classes('text-h6 font-bold')

                        with ui.column().classes('stat-item text-center'):
                            ui.label('违规次数').classes('text-sm text-gray-600')
                            stat_violations = ui.label('0').classes('text-h6 font-bold text-danger')

                        with ui.column().classes('stat-item text-center'):
                            ui.label('检测率').classes('text-sm text-gray-600')
                            stat_rate = ui.label('0%').classes('text-h6 font-bold')

                        with ui.column().classes('stat-item text-center'):
                            ui.label('学习模式').classes('text-sm text-gray-600')
                            stat_patterns = ui.label('0').classes('text-h6 font-bold')

            # 右侧告警面板
            with ui.column().classes('w-1/3'):
                # 实时告警
                with ui.card().classes('w-full'):
                    ui.label('实时告警').classes('text-h6 font-bold mb-4')

                    # 告警统计
                    with ui.grid(columns=4).classes('w-full gap-1 mb-4'):
                        ui.label('严重').classes('text-center text-danger font-bold')
                        ui.label('重要').classes('text-center text-warning font-bold')
                        ui.label('轻微').classes('text-center text-info font-bold')
                        ui.label('警告').classes('text-center text-primary font-bold')

                        critical_count = ui.label('0').classes('text-center')
                        major_count = ui.label('0').classes('text-center')
                        minor_count = ui.label('0').classes('text-center')
                        warning_count = ui.label('0').classes('text-center')

                    # 告警列表
                    alert_list = ui.column().classes('w-full h-64 overflow-y-auto')

                    # 清空告警
                    ui.button('清空告警', icon='clear', on_click=clear_alerts).classes('mt-2 w-full')

                # 当前违规详情
                with ui.card().classes('w-full'):
                    ui.label('当前违规详情').classes('text-h6 font-bold mb-4')
                    violation_details = ui.column().classes('w-full h-32 overflow-y-auto')

        # 底部状态栏
        with ui.footer().classes('bg-gray-800 text-white'):
            with ui.row().classes('w-full items-center'):
                ui.label('系统状态: ').classes('font-bold')
                footer_status = ui.label('就绪').classes('status-good')
                ui.space()
                ui.label('实验室: ').classes('font-bold')
                ui.label('chemistry_lab_001').classes('')
                ui.space()
                ui.label('最后更新: ').classes('font-bold')
                last_update = ui.label(datetime.now().strftime('%H:%M:%S'))

        # 事件处理函数
        async def start_enhanced_monitoring():
            """启动增强监控"""
            app_state.is_monitoring = True
            app_state.frame_count = 0
            app_state.alerts.clear()

            add_log("启动自适应AI监控系统")
            monitor_status.set_text('运行中')
            monitor_status.classes('status-good')
            start_btn.disable()
            stop_btn.enable()

            ui.notify('自适应监控系统已启动', type='positive')

        async def stop_enhanced_monitoring():
            """停止增强监控"""
            app_state.is_monitoring = False
            add_log("停止自适应AI监控系统")

            monitor_status.set_text('已停止')
            monitor_status.classes('status-warning')
            start_btn.enable()
            stop_btn.disable()

            ui.notify('监控系统已停止', type='info')

        def reset_statistics():
            """重置统计"""
            app_state.frame_count = 0
            app_state.alerts.clear()
            app_state.statistics = {
                "total_frames": 0,
                "total_violations": 0,
                "detection_rate": 0.0
            }
            if app_state.monitor:
                app_state.monitor.violation_history.clear()

            add_log("统计数据已重置")
            ui.notify('统计数据已重置', type='info')

        def export_report():
            """导出报告"""
            add_log("生成监控报告...")
            ui.notify('报告生成功能开发中', type='info')

        def clear_alerts():
            """清空告警"""
            app_state.alerts.clear()
            alert_list.clear()
            add_log("告警列表已清空")

        def update_constraint_display(constraint_list):
            """更新约束显示"""
            constraint_list.clear()
            if app_state.monitor:
                with constraint_list:
                    for constraint in app_state.monitor.current_constraints:
                        severity_color = {
                            'Critical': 'text-danger',
                            'Major': 'text-warning',
                            'Minor': 'text-info',
                            'Warning': 'text-primary'
                        }.get(constraint.severity.value, '')

                        ui.label(f"• {constraint.description}").classes(f'{severity_color} text-sm')

        # 定时更新函数
        async def auto_update():
            """自动更新"""
            if app_state.is_monitoring:
                # 处理新帧
                result = await process_video_frame()
                if result:
                    frame_base64, violations = result

                    # 更新视频显示
                    if frame_base64:
                        video_display.set_content(f'''
                        <div class="video-container">
                            <img src="data:image/jpeg;base64,{frame_base64}" class="video-frame" alt="监控画面">
                        </div>
                        ''')

                    # 更新统计
                    stats = app_state.monitor.get_statistics() if app_state.monitor else {}
                    frame_count.set_text(str(app_state.frame_count))
                    violation_count.set_text(str(stats.get('total_violations', 0)))

                    stat_frames.set_text(str(app_state.statistics['total_frames']))
                    stat_violations.set_text(str(app_state.statistics['total_violations']))
                    stat_patterns.set_text(str(stats.get('learning_patterns', 0)))

                    # 计算检测率
                    if app_state.frame_count > 0:
                        detection_rate = (app_state.statistics['total_violations'] / app_state.frame_count) * 100
                        stat_rate.set_text(f"{detection_rate:.1f}%")

            # 更新告警列表
            update_alert_display()

            # 更新日志
            update_log_display()

            # 更新时间
            last_update.set_text(datetime.now().strftime('%H:%M:%S'))

        def update_alert_display():
            """更新告警显示"""
            alert_list.clear()

            # 统计各等级告警
            severity_counts = {'Critical': 0, 'Major': 0, 'Minor': 0, 'Warning': 0}

            with alert_list:
                for alert in app_state.alerts[:20]:  # 显示最近20条
                    severity = alert.get('severity', 'Warning')
                    severity_counts[severity] = severity_counts.get(severity, 0) + 1

                    severity_class = f"alert-{severity.lower()}"
                    with ui.card().classes(f'{severity_class} p-2'):
                        ui.label(f"{alert.get('constraint', '未知违规')}").classes('font-bold text-sm')
                        ui.label(f"等级: {severity} | 置信度: {alert.get('confidence', 0):.2f}").classes('text-xs text-gray-600')
                        ui.label(alert.get('recommendation', '')).classes('text-xs')

            # 更新统计数字
            critical_count.set_text(str(severity_counts['Critical']))
            major_count.set_text(str(severity_counts['Major']))
            minor_count.set_text(str(severity_counts['Minor']))
            warning_count.set_text(str(severity_counts['Warning']))

        def update_log_display():
            """更新日志显示"""
            log_area.clear()
            with log_area:
                for log_msg in app_state.log_messages[-15:]:  # 显示最近15条
                    ui.label(log_msg).classes('text-sm')

        # 启动定时更新
        ui.timer(1.0, auto_update)  # 每秒更新一次

        # 初始化
        ui.timer(0.5, auto_update, once=True)

    if __name__ == "__main__":
        ui.run(
            title="自适应AI实验室监控系统",
            host="0.0.0.0",
            port=8080,
            reload=False,
            show=False
        )

except Exception as e:
    print(f"增强版前端启动失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)