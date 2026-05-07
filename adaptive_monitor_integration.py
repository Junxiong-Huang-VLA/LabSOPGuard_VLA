"""Deprecated compatibility helpers for the old adaptive monitor integration spike.

This repository now serves realtime monitoring from the integrated homepage:
  /#realtime-monitor-panel

The authoritative backend implementation lives in:
  integrated_system/adaptive_monitor.py
  integrated_system/app_integrated.py
  integrated_system/templates/integrated_index.html

Keep this module only so stale notes or local experiments that import it still get a
clear pointer to the current entrypoints instead of an outdated parallel implementation.
"""
from __future__ import annotations

HOME_MONITOR_PATH = "/#realtime-monitor-panel"
LEGACY_ROUTE = "/adaptive_monitor"
API_PREFIX = "/api/adaptive_monitor"

ADAPTIVE_MONITOR_JS = """// Deprecated helper.
// Realtime monitor UI now lives on the integrated homepage at /#realtime-monitor-panel.
// Use the server-rendered panel in integrated_system/templates/integrated_index.html.
"""

ADAPTIVE_MONITOR_API = """# Deprecated helper.
# Use the integrated Flask routes in integrated_system/app_integrated.py:
#   POST /api/adaptive_monitor/start
#   POST /api/adaptive_monitor/stop
#   POST /api/adaptive_monitor/process_frame
#   POST /api/adaptive_monitor/reset
#   GET  /api/adaptive_monitor/status
#   GET  /api/adaptive_monitor/report
# Legacy browser path /adaptive_monitor now redirects to /#realtime-monitor-panel.
"""


def get_adaptive_monitor_js() -> str:
    """Return a deprecation notice for stale integration experiments."""
    return ADAPTIVE_MONITOR_JS



def get_adaptive_monitor_api() -> str:
    """Return the current adaptive monitor API entrypoints."""
    return ADAPTIVE_MONITOR_API
