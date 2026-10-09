"""상시 실행 supervisor — vmc_bridge 를 로컬 서버로 계속 돌린다.

    python AI_Streaming/server.py                 # 실행 (전경, Ctrl+C 로 정상 종료)
    python AI_Streaming/server.py --status        # 다른 창에서: 상태 JSON
    python AI_Streaming/server.py --bypass on     # 원본 통과 켜기 (off 로 끄기)
    python AI_Streaming/server.py --reset         # 세션 리셋
    python AI_Streaming/server.py --stop          # supervisor 와 브리지 모두 정상 종료

설정은 server_config.json (같은 폴더). 브리지 자체는 바꾸지 않고 자식 프로세스로 띄운다.

[vmc_bridge 를 그냥 켜 두는 것과 무엇이 다른가 — 2026-10-04 실측으로 드러난 것]
  * 브리지가 죽으면(예외·포트 충돌·체크포인트 오류) 그대로 끝났다 → 자식이 비정상 종료하면
    백오프(1→2→4…60초)로 다시 띄운다. 5분 이상 버틴 뒤의 종료는 백오프를 1초로 되돌린다.
  * 메인 루프가 멈춰도(교착·무한 대기) 프로세스는 살아 있어 아무도 몰랐다 → 브리지의
    /health heartbeat 를 감시해 stall_restart_s 동안 응답이 없으면 죽이고 다시 띄운다.
  * 콘솔 출력만 있었다 → 회전 로그 파일(logs/server.log, 크기 제한)에 남긴다.
  * 체크포인트는 checkpoints/temp.pth 하나다 — 고를 것이 없어 재시작해도 같은 모델이다.
    (예전 설정 파일에 "run" / "epoch" 가 남아 있으면 무시하고 로그에 알린다.)
  * 중복 실행 방지: 잠금 파일(logs/server.lock). 같은 포트를 두 브리지가 잡지 않는다.

자식 종료 코드 0 은 '의도된 종료'(/shutdown)로 보고 supervisor 도 끝낸다. 그 밖은 재시작.
"""

import argparse
import json
import logging
import logging.handlers
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import control  # noqa: E402

DEFAULT_CONFIG = os.path.join(HERE, "server_config.json")


def load_config(path):
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.setdefault("log_dir", "logs")
    cfg.setdefault("log_max_mb", 5)
    cfg.setdefault("log_backups", 5)
    cfg.setdefault("health_interval_s", 2.0)
    cfg.setdefault("startup_grace_s", 90.0)
    cfg.setdefault("stall_restart_s", 15.0)
    cfg.setdefault("stable_after_s", 300.0)
    cfg.setdefault("max_backoff_s", 60.0)
    cfg.setdefault("extra_args", [])
    cfg.setdefault("emit_lag", 2)
    cfg.setdefault("hold_repush", False)
    log_dir = cfg["log_dir"]
    cfg["log_dir"] = log_dir if os.path.isabs(log_dir) else os.path.join(HERE, log_dir)
    return cfg


def bridge_cmd(cfg):
    cmd = [sys.executable, "-u", os.path.join(HERE, "vmc_bridge.py"),
           "--listen-port", str(cfg["listen_port"]),
           "--send-host", cfg["send_host"], "--send-port", str(cfg["send_port"]),
           "--fps", str(cfg["fps"]),
           "--status-port", str(cfg["status_port"]),
           "--threads", str(cfg.get("threads", 0)),
           "--no-keys",
           "--emit-lag", str(cfg.get("emit_lag", 2))]
    if cfg.get("hold_repush"):
        cmd.append("--hold-repush")
    if cfg.get("log_csv"):
        cmd += ["--log-csv", cfg["log_csv"]]
    return cmd + [str(a) for a in cfg["extra_args"]]


def make_logger(cfg, echo=True):
    os.makedirs(cfg["log_dir"], exist_ok=True)
    log = logging.getLogger("mcc_server")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fh = logging.handlers.RotatingFileHandler(
        os.path.join(cfg["log_dir"], "server.log"), maxBytes=int(cfg["log_max_mb"] * 2**20),
        backupCount=int(cfg["log_backups"]), encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s %(message)s")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    if echo:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        log.addHandler(sh)
    return log


class InstanceLock:
    """logs/server.lock 을 배타 잠금. 프로세스가 죽으면 OS 가 잠금을 푼다(오래된 잠금 파일 걱정 없음)."""

    def __init__(self, path):
        self.path = path
        self.f = None

    def acquire(self):
        self.f = open(self.path, "a+")
        try:
            if os.name == "nt":
                import msvcrt
                self.f.seek(0)
                msvcrt.locking(self.f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.f.close()
            self.f = None
            return False
        return True


class Supervisor:
    def __init__(self, cfg, log):
        self.cfg = cfg
        self.log = log
        self.child = None
        self.child_started = 0.0
        self.restarts = 0
        self.backoff = 1.0
        self.stop_file = os.path.join(cfg["log_dir"], "server.stop")

    # --- 자식 ----------------------------------------------------------
    def _pump(self, proc):
        """자식 stdout 을 줄 단위로 로그에 옮긴다 (파이프가 차서 자식이 막히지 않게 항상 읽는다)."""
        for line in iter(proc.stdout.readline, ""):
            self.log.info("[bridge %d] %s", proc.pid, line.rstrip())
        proc.stdout.close()

    def start_child(self):
        cmd = bridge_cmd(self.cfg)
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        self.child = subprocess.Popen(cmd, cwd=os.path.dirname(HERE), env=env, stdin=subprocess.DEVNULL,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, encoding="utf-8", errors="replace", bufsize=1)
        self.child_started = time.monotonic()
        self.child_healthy = False          # 한 번이라도 /health 200 을 받았나 (시작 유예는 그 전까지만)
        threading.Thread(target=self._pump, args=(self.child,), daemon=True).start()
        self.log.info("브리지 시작 pid=%d (재시작 %d회)", self.child.pid, self.restarts)

    def stop_child(self, graceful=True, timeout=8.0):
        p = self.child
        if p is None or p.poll() is not None:
            return
        if graceful:
            control.request(self.cfg["status_port"], "/shutdown", method="POST")
            try:
                p.wait(timeout)
                return
            except subprocess.TimeoutExpired:
                self.log.info("정상 종료 응답 없음 — 강제 종료")
        p.kill()
        p.wait(5)

    # --- 감시 루프 -----------------------------------------------------
    def run(self):
        cfg = self.cfg
        if os.path.exists(self.stop_file):
            os.remove(self.stop_file)
        for old in ("run", "epoch"):
            if old in cfg:
                self.log.info("설정의 \"%s\" 항목은 더 이상 쓰지 않는다 (가중치는 checkpoints/temp.pth 하나) — 무시한다.", old)
        self.log.info("supervisor 시작 pid=%d  설정=%s", os.getpid(), json.dumps(cfg, ensure_ascii=False))
        self.start_child()
        unhealthy_since = None
        try:
            while True:
                time.sleep(cfg["health_interval_s"])
                if os.path.exists(self.stop_file):
                    self.log.info("stop 파일 감지 — 정상 종료")
                    os.remove(self.stop_file)
                    break
                code = self.child.poll()
                if code is not None:
                    up = time.monotonic() - self.child_started
                    if code == 0:
                        self.log.info("브리지가 정상 종료(코드 0, %.0fs) — supervisor 도 끝낸다", up)
                        break
                    self._restart(f"브리지 비정상 종료 코드 {code} (가동 {up:.0f}s)", up)
                    unhealthy_since = None
                    continue
                up = time.monotonic() - self.child_started
                http, body = control.request(cfg["status_port"], "/health", timeout=2.0)
                if http == 200:
                    self.child_healthy = True
                    unhealthy_since = None
                    if up >= cfg["stable_after_s"]:
                        self.backoff = 1.0
                    continue
                if not self.child_healthy and up < cfg["startup_grace_s"]:
                    continue                        # 모델 로드·워밍업 중 (HTTP 가 아직 안 열림)
                    # 한 번 정상 응답한 뒤에는 유예하지 않는다 — 예전에는 가동 90초 안의 멈춤을
                    # '아직 기동 중'으로 보고 93.6초 뒤에야 재시작했다 (장애 주입 실측).
                now = time.monotonic()
                unhealthy_since = unhealthy_since or now
                if now - unhealthy_since >= cfg["stall_restart_s"]:
                    self.log.info("브리지 응답 없음 %.0fs (%s %s) — 강제 재시작",
                                  now - unhealthy_since, http, body)
                    self.stop_child(graceful=False)
                    self._restart("멈춤 감지", up)
                    unhealthy_since = None
        except KeyboardInterrupt:
            self.log.info("Ctrl+C — 정상 종료")
        finally:
            self.stop_child(graceful=True)
            self.log.info("supervisor 종료 (재시작 누계 %d회)", self.restarts)

    def _restart(self, why, uptime):
        if uptime >= self.cfg["stable_after_s"]:
            self.backoff = 1.0
        self.log.info("%s — %.0fs 뒤 재시작", why, self.backoff)
        t_end = time.monotonic() + self.backoff
        while time.monotonic() < t_end:            # 백오프 중에도 stop 파일에 반응
            if os.path.exists(self.stop_file):
                return
            time.sleep(0.2)
        self.backoff = min(self.cfg["max_backoff_s"], self.backoff * 2)
        self.restarts += 1
        self.start_child()


def main(argv=None):
    ap = argparse.ArgumentParser(description="vmc_bridge 상시 실행 supervisor")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--status", action="store_true", help="실행 중인 브리지의 상태 JSON 출력")
    ap.add_argument("--stop", action="store_true", help="실행 중인 supervisor·브리지를 정상 종료")
    ap.add_argument("--bypass", choices=["on", "off", "toggle"])
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--print-cmd", action="store_true", help="띄울 브리지 명령만 출력")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    port = cfg["status_port"]

    if args.print_cmd:
        print(subprocess.list2cmdline(bridge_cmd(cfg)))
        return 0
    if args.status:
        code, body = control.request(port, "/status")
        print(json.dumps(body, ensure_ascii=False, indent=2))
        return 0 if code == 200 else 1
    if args.bypass or args.reset:
        path = "/reset" if args.reset else "/bypass" + {"on": "?on=1", "off": "?on=0", "toggle": ""}[args.bypass]
        code, body = control.request(port, path, method="POST")
        print(code, body)
        return 0 if code == 202 else 1
    if args.stop:
        os.makedirs(cfg["log_dir"], exist_ok=True)
        open(os.path.join(cfg["log_dir"], "server.stop"), "w").close()
        print("stop 요청을 남겼다 — supervisor 가 health 주기 안에 브리지를 정상 종료한다.")
        return 0

    os.makedirs(cfg["log_dir"], exist_ok=True)
    lock = InstanceLock(os.path.join(cfg["log_dir"], "server.lock"))
    if not lock.acquire():
        print("이미 실행 중인 supervisor 가 있다 (logs/server.lock). --status 로 확인하거나 --stop 으로 끈다.")
        return 2
    log = make_logger(cfg)
    Supervisor(cfg, log).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
