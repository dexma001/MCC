"""로컬 상태·제어 HTTP 엔드포인트 — 상시 실행 브리지를 콘솔 없이 보고 조작한다.

    GET  /health      → 200 {"ok": true, "heartbeat_age_s": ...}  (메인 루프가 멈추면 503)
    GET  /status      → 200 상태 JSON (프레임 수, 지연 p95, 오류, bypass, 체크포인트 …)
    POST /bypass?on=1 → bypass 켜기 (on=0 끄기, on 생략 = 토글)
    POST /reset       → 세션 리셋 (정규화 기준·링버퍼)
    POST /shutdown    → 메인 루프를 정상 종료 (finally 정리까지 탄다)

127.0.0.1 에만 묶는다 — 같은 PC 의 supervisor(server.py)·사람만 접근한다. 인증이 없으므로
0.0.0.0 으로 열지 않는다.

메인 루프와는 '요청 대기열'로만 대화한다: HTTP 스레드는 명령을 넣기만 하고, 실제 bypass/reset 은
메인 루프가 프레임 사이에 적용한다(보정기는 스레드 안전하지 않다).
"""

import json
import queue
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HEARTBEAT_STALE_S = 5.0
"""메인 루프 heartbeat 가 이보다 오래되면 /health 가 503 — supervisor 가 멈춘 프로세스로 본다."""


class StatusServer:
    def __init__(self, port, host="127.0.0.1"):
        self.commands = queue.Queue()
        self._status = {}
        self._lock = threading.Lock()
        self.heartbeat = time.monotonic()
        outer = self

        class _H(BaseHTTPRequestHandler):
            def log_message(self, *a):          # 콘솔을 접근 로그로 덮지 않는다
                pass

            def _reply(self, code, obj):
                body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = urlparse(self.path).path
                age = time.monotonic() - outer.heartbeat
                if path == "/health":
                    ok = age < HEARTBEAT_STALE_S
                    self._reply(200 if ok else 503, {"ok": ok, "heartbeat_age_s": round(age, 3)})
                elif path == "/status":
                    with outer._lock:
                        st = dict(outer._status)
                    st["heartbeat_age_s"] = round(age, 3)
                    self._reply(200, st)
                else:
                    self._reply(404, {"error": "unknown path", "paths": ["/health", "/status"]})

            def do_POST(self):
                u = urlparse(self.path)
                q = parse_qs(u.query)
                if u.path == "/bypass":
                    on = q.get("on", [None])[0]
                    outer.commands.put(("bypass", None if on is None else on not in ("0", "false", "off")))
                elif u.path in ("/reset", "/shutdown"):
                    outer.commands.put((u.path[1:], None))
                else:
                    self._reply(404, {"error": "unknown path", "paths": ["/bypass", "/reset", "/shutdown"]})
                    return
                self._reply(202, {"queued": u.path[1:]})

        self._httpd = ThreadingHTTPServer((host, port), _H)
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._httpd.shutdown()
        self._httpd.server_close()

    def beat(self):
        """메인 루프가 매 반복 호출한다 (프레임이 없어도). 멈춤 감지용."""
        self.heartbeat = time.monotonic()

    def publish(self, status):
        with self._lock:
            self._status = status

    def drain(self):
        """대기 중인 명령 [(name, arg)] 를 모두 꺼낸다 (메인 루프 전용)."""
        out = []
        while True:
            try:
                out.append(self.commands.get_nowait())
            except queue.Empty:
                return out


# ---------------------------------------------------------------------
# 클라이언트 (server.py · 사람)
# ---------------------------------------------------------------------
def request(port, path, method="GET", timeout=2.0, host="127.0.0.1"):
    """(HTTP 코드, JSON) — 연결 실패면 (None, {"error": ...})."""
    req = urllib.request.Request(f"http://{host}:{port}{path}", method=method,
                                 data=b"" if method == "POST" else None)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8"))
        except Exception:
            return e.code, {}
    except Exception as e:
        return None, {"error": f"{type(e).__name__}: {e}"}
