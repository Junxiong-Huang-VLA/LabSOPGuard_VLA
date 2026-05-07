from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import URLError
from urllib.request import urlopen


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LabSOPGuard Camera Monitor</title>
  <style>
    :root { color-scheme: dark; font-family: Arial, Helvetica, sans-serif; background: #111827; color: #f9fafb; }
    body { margin: 0; min-height: 100vh; background: #111827; }
    header { display: flex; align-items: center; justify-content: space-between; padding: 14px 18px; border-bottom: 1px solid #263244; background: #0b1220; }
    h1 { margin: 0; font-size: 18px; letter-spacing: 0; }
    .meta { color: #9ca3af; font-size: 13px; }
    button { border: 1px solid #334155; background: #1f2937; color: #f9fafb; padding: 8px 12px; border-radius: 6px; cursor: pointer; }
    button:hover { background: #374151; }
    main { padding: 12px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 10px; }
    .card { background: #030712; border: 1px solid #263244; border-radius: 6px; overflow: hidden; }
    .card-head { display: flex; align-items: center; justify-content: space-between; padding: 8px 10px; background: #0b1220; }
    .name { display: flex; align-items: center; gap: 8px; font-weight: 700; font-size: 13px; }
    .dot { width: 9px; height: 9px; border-radius: 50%; background: #ef4444; display: inline-block; }
    .dot.online { background: #22c55e; box-shadow: 0 0 10px rgba(34, 197, 94, .8); }
    .sub { color: #9ca3af; font-size: 12px; }
    .video { position: relative; aspect-ratio: 16 / 9; background: #000; display: flex; align-items: center; justify-content: center; }
    img { width: 100%; height: 100%; object-fit: contain; display: block; }
    .empty, .error { padding: 28px; border: 1px solid #334155; border-radius: 6px; color: #cbd5e1; background: #0b1220; }
    .error { color: #fecaca; border-color: #7f1d1d; background: #270f12; }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>LabSOPGuard Camera Monitor</h1>
      <div id="summary" class="meta">Loading cameras...</div>
    </div>
    <button id="refresh">Refresh</button>
  </header>
  <main>
    <div id="error" class="error" style="display:none"></div>
    <div id="grid" class="grid"></div>
  </main>
  <script>
    const API = 'http://127.0.0.1:8000/api/v1/cameras';
    const grid = document.getElementById('grid');
    const summary = document.getElementById('summary');
    const errorBox = document.getElementById('error');

    function streamUrl(id) {
      return `${API}/${encodeURIComponent(id)}/stream?t=${Date.now()}`;
    }

    function render(cameras) {
      const online = cameras.filter(c => c.online).length;
      summary.textContent = `${cameras.length} configured, ${online} online`;
      if (!cameras.length) {
        grid.innerHTML = '<div class="empty">No cameras returned by backend.</div>';
        return;
      }
      grid.innerHTML = cameras.map(c => `
        <section class="card">
          <div class="card-head">
            <div class="name"><span class="dot ${c.online ? 'online' : ''}"></span><span>${c.label || c.camera_id}</span></div>
            <div class="sub">${c.source || 'camera'}${c.listen_port ? ' :' + c.listen_port : ''}</div>
          </div>
          <div class="video"><img alt="${c.camera_id}" src="${streamUrl(c.camera_id)}" /></div>
        </section>
      `).join('');
    }

    async function load() {
      errorBox.style.display = 'none';
      try {
        const response = await fetch(API, { cache: 'no-store' });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        render(data.cameras || []);
      } catch (error) {
        summary.textContent = 'Backend not reachable';
        errorBox.textContent = String(error && error.message ? error.message : error);
        errorBox.style.display = 'block';
      }
    }

    document.getElementById('refresh').addEventListener('click', load);
    load();
    setInterval(load, 5000);
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    backend_url: str = "http://127.0.0.1:8000"

    def do_GET(self) -> None:
        if self.path.startswith("/api/"):
            self._proxy_api()
            return
        self._send_html()

    def do_HEAD(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()

    def log_message(self, fmt: str, *args) -> None:
        return

    def _send_html(self) -> None:
        body = HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _proxy_api(self) -> None:
        try:
            with urlopen(self.backend_url + self.path, timeout=10) as response:
                body = response.read()
                content_type = response.headers.get("Content-Type", "application/json")
                self.send_response(response.status)
                self.send_header("Content-Type", content_type)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)
        except URLError as exc:
            body = json.dumps({"error": str(exc)}).encode("utf-8")
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5173)
    parser.add_argument("--backend", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    Handler.backend_url = args.backend.rstrip("/")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Emergency camera frontend listening on http://{args.host}:{args.port}/cameras")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
