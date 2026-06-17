#!/usr/bin/env python3
"""
FlowGenix AI Server — uses Google Gemini (free tier).

FREE: Get a key in 2 minutes at https://aistudio.google.com → "Get API key"
No credit card. No billing. 15 requests/min, 1M tokens/day free.

Local run (PowerShell):
    $env:GEMINI_API_KEY = "AIza..."
    py -3 tools/ai_server.py

Railway deploy: push to GitHub, connect repo in Railway dashboard,
add GEMINI_API_KEY as an environment variable. Done — all users get AI free.
"""
import json, os, sys, time, urllib.request, urllib.error, http.server, socketserver
from collections import defaultdict
from threading import Lock

KEY    = os.environ.get("GEMINI_API_KEY", "")
MODEL  = "gemini-1.5-flash"
PORT   = int(os.environ.get("PORT", 5181))
HOST   = "0.0.0.0"  # must bind to all interfaces on Railway/Render so external traffic reaches it

GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

# ── Rate limiting: 12 requests / IP / 60 s (stay inside Gemini free tier) ────
_rate: dict = defaultdict(list)
_lock       = Lock()
RATE_MAX    = 12
RATE_WINDOW = 60

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

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")

    def do_OPTIONS(self):
        self.send_response(204); self._cors(); self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/health"):
            self._json(200, {"ok": True, "model": MODEL, "key": bool(KEY)})
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        if self.path != "/complete":
            self.send_response(404); self.end_headers(); return

        ip = self.client_address[0]

        if not KEY:
            self._err(500, "Server has no GEMINI_API_KEY configured."); return

        if not allowed(ip):
            self._err(429, "Rate limit: 12 requests per minute."); return

        try:
            n    = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            self._err(400, "Invalid JSON."); return

        system = body.get("system", "")
        prompt = body.get("prompt", "")
        max_tokens = min(int(body.get("maxTokens", 600)), 1000)

        # Gemini API format: system instruction + user turn
        gemini_payload = json.dumps({
            "system_instruction": {"parts": [{"text": system}]} if system else None,
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.7},
        }, default=lambda x: None).encode()

        # Remove null system_instruction if not set
        payload_obj = json.loads(gemini_payload)
        if not payload_obj.get("system_instruction"):
            payload_obj.pop("system_instruction", None)
        gemini_payload = json.dumps(payload_obj).encode()

        req = urllib.request.Request(
            f"{GEMINI_URL}?key={KEY}",
            data    = gemini_payload,
            headers = {"content-type": "application/json"},
            method  = "POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read())
            # Extract text from Gemini response
            text = ""
            for candidate in data.get("candidates", []):
                for part in candidate.get("content", {}).get("parts", []):
                    text += part.get("text", "")
            self._json(200, {"text": text.strip()})
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:400]
            self._err(e.code, f"Gemini API error: {detail}")
        except Exception as ex:
            self._err(502, str(ex))

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
        print("ERROR: GEMINI_API_KEY not set.", file=sys.stderr)
        print("Get a free key at https://aistudio.google.com → Get API key")
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((HOST, PORT), Handler) as srv:
        print(f"FlowGenix AI server  {HOST}:{PORT}")
        print(f"Model : {MODEL}  (Google Gemini — free tier)")
        print(f"Key   : {'SET' if KEY else 'NOT SET'}")
        srv.serve_forever()
