#!/usr/bin/env python3
"""
FlowGenix AI Server — production proxy.

Holds the Anthropic API key server-side. The browser calls this server;
users never see or need a key. Deploy to Railway (free tier works fine).

Local run (PowerShell):
    $env:ANTHROPIC_API_KEY = "sk-ant-..."
    py -3 tools/ai_server.py

Railway deploy: push to GitHub, connect repo in Railway, set
ANTHROPIC_API_KEY in Railway's environment variables panel. Done.
"""
import json, os, sys, time, urllib.request, urllib.error, http.server, socketserver
from collections import defaultdict
from threading import Lock

KEY          = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL        = "claude-sonnet-4-5"
PORT         = int(os.environ.get("PORT", 5181))
# On Railway / Render the HOST env is not set, but we need 0.0.0.0 to accept traffic.
# Locally we bind to 127.0.0.1 only.
HOST         = "0.0.0.0" if os.environ.get("RAILWAY_ENVIRONMENT") or os.environ.get("RENDER") else "127.0.0.1"

# ── Rate limiting: 20 requests / IP / 60 s ───────────────────────────────────
_rate: dict  = defaultdict(list)
_lock        = Lock()
RATE_MAX     = 20
RATE_WINDOW  = 60   # seconds

def allowed(ip: str) -> bool:
    now = time.time()
    with _lock:
        hits = [t for t in _rate[ip] if now - t < RATE_WINDOW]
        _rate[ip] = hits
        if len(hits) >= RATE_MAX:
            return False
        _rate[ip].append(now)
        return True


class Handler(http.server.BaseHTTPRequestHandler):

    # ── CORS headers (required: browser calls from a different origin) ────────
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    # ── Health check — Railway pings this to confirm the server is alive ──────
    def do_GET(self):
        if self.path in ("/", "/health"):
            self._json(200, {"ok": True, "model": MODEL, "key": bool(KEY)})
        else:
            self.send_response(404); self.end_headers()

    # ── Main proxy endpoint ───────────────────────────────────────────────────
    def do_POST(self):
        if self.path != "/complete":
            self.send_response(404); self.end_headers(); return

        ip = self.client_address[0]

        if not KEY:
            self._err(500, "Server has no API key configured — set ANTHROPIC_API_KEY.")
            return

        if not allowed(ip):
            self._err(429, "Rate limit: max 20 requests per minute per IP.")
            return

        try:
            n    = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            self._err(400, "Invalid JSON body."); return

        payload = json.dumps({
            "model":      body.get("model", MODEL),
            "max_tokens": min(int(body.get("maxTokens", 600)), 1000),
            "system":     body.get("system", ""),
            "messages":   [{"role": "user", "content": body.get("prompt", "")}],
        }).encode()

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data    = payload,
            headers = {
                "content-type":      "application/json",
                "x-api-key":         KEY,
                "anthropic-version": "2023-06-01",
            },
            method = "POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read())
            text = "".join(
                b.get("text", "") for b in data.get("content", [])
                if b.get("type") == "text"
            )
            self._json(200, {"text": text})
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:400]
            self._err(e.code, f"Anthropic: {detail}")
        except Exception as ex:
            self._err(502, str(ex))

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _json(self, code: int, obj: dict):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self._cors()
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _err(self, code: int, msg: str):
        self._json(code, {"error": msg})

    def log_message(self, fmt, *args):
        print(f"[{self.client_address[0]}] {fmt % args}", flush=True)


if __name__ == "__main__":
    if not KEY:
        print("WARNING: ANTHROPIC_API_KEY is not set — /complete will return 500.", file=sys.stderr)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((HOST, PORT), Handler) as srv:
        print(f"FlowGenix AI server  {HOST}:{PORT}")
        print(f"Model : {MODEL}")
        print(f"Key   : {'SET' if KEY else 'NOT SET — set ANTHROPIC_API_KEY'}")
        srv.serve_forever()
