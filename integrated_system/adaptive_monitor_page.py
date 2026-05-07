"""Compatibility helper for the deprecated standalone adaptive monitor page.

The integrated realtime monitor now lives on the homepage at /#realtime-monitor-panel.
Keep this module so any stale imports still render a useful redirect page.
"""
from flask import render_template_string

ADAPTIVE_MONITOR_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="0; url=/#realtime-monitor-panel">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Adaptive Monitor Redirect</title>
  <style>
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      background: #0f172a;
      color: #e2e8f0;
      font-family: Arial, sans-serif;
    }
    main {
      max-width: 560px;
      padding: 24px;
      border: 1px solid rgba(148, 163, 184, 0.35);
      border-radius: 16px;
      background: rgba(15, 23, 42, 0.92);
      box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
    }
    a { color: #38bdf8; }
    p { line-height: 1.5; }
  </style>
</head>
<body>
  <main>
    <h1>Adaptive Monitor Moved</h1>
    <p>The standalone monitor page has been merged into the integrated home page.</p>
    <p>You will be redirected to <a href="/#realtime-monitor-panel">/#realtime-monitor-panel</a>.</p>
  </main>
</body>
</html>
"""


def get_adaptive_monitor_html() -> str:
    return render_template_string(ADAPTIVE_MONITOR_HTML)
