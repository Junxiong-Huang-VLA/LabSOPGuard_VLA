"""
启动NiceGUI前端服务
"""
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("启动NiceGUI前端服务...")
print("前端Dashboard地址: http://localhost:8080")

try:
    from nicegui import ui
    import uvicorn

    # 简化的前端界面
    @ui.page('/')
    def dashboard():
        ui.add_head_html('''
        <style>
            .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; }
            .card { background: white; border-radius: 10px; padding: 20px; margin: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            .status-good { color: #28a745; }
            .status-warning { color: #ffc107; }
            .status-danger { color: #dc3545; }
        </style>
        ''')

        with ui.header().classes('header'):
            ui.label('实验室SOP合规智能监控系统').classes('text-h4 font-bold')
            ui.space()
            ui.label('v1.0.0').classes('text-sm')

        with ui.row().classes('p-4'):
            # 系统状态卡片
            with ui.card().classes('w-full'):
                ui.label('系统状态').classes('text-h6 font-bold mb-4')

                with ui.row().classes('w-full'):
                    with ui.column().classes('w-1/3 text-center'):
                        ui.label('后端API').classes('text-sm text-gray-600')
                        ui.label('运行中').classes('status-good text-lg font-bold')

                    with ui.column().classes('w-1/3 text-center'):
                        ui.label('视频流').classes('text-sm text-gray-600')
                        ui.label('0 路').classes('text-lg font-bold')

                    with ui.column().classes('w-1/3 text-center'):
                        ui.label('今日告警').classes('text-sm text-gray-600')
                        ui.label('0').classes('text-lg font-bold')

            # 控制面板
            with ui.card().classes('w-full'):
                ui.label('控制面板').classes('text-h6 font-bold mb-4')

                with ui.row():
                    ui.button('启动监控', icon='play', color='positive')
                    ui.button('停止监控', icon='stop', color='negative')
                    ui.button('生成报告', icon='description')
                    ui.button('系统设置', icon='settings')

            # 实时监控区域
            with ui.card().classes('w-full'):
                ui.label('实时监控').classes('text-h6 font-bold mb-4')

                ui.html('''
                <div style="background: #f8f9fa; border: 2px dashed #dee2e6; border-radius: 8px;
                           height: 300px; display: flex; align-items: center; justify-content: center;">
                    <div style="text-align: center; color: #6c757d;">
                        <div style="font-size: 48px; margin-bottom: 10px;">📹</div>
                        <div style="font-size: 18px;">视频监控画面</div>
                        <div style="font-size: 14px;">点击"启动监控"开始实时监控</div>
                    </div>
                </div>
                ''')

            # 告警列表
            with ui.card().classes('w-full'):
                ui.label('最近告警').classes('text-h6 font-bold mb-4')

                ui.label('暂无告警记录').classes('text-gray-500 text-center')

        # 底部状态栏
        with ui.footer().classes('bg-gray-800 text-white'):
            with ui.row().classes('w-full items-center'):
                ui.label('系统状态: ').classes('font-bold')
                ui.label('正常运行').classes('status-good')
                ui.space()
                ui.label('最后更新: ').classes('font-bold')
                ui.label('2026-04-08 11:23')

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
    print("尝试简化启动...")

    # 备用方案：使用uvicorn直接启动
    try:
        from nicegui import ui

        @ui.page('/')
        def simple_dashboard():
            ui.label('实验室SOP合规智能监控系统').classes('text-h4')
            ui.label('前端服务启动中...')

        ui.run(host="0.0.0.0", port=8080, show=False)
    except Exception as e2:
        print(f"备用方案也失败: {e2}")
        sys.exit(1)