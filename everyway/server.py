#!/usr/bin/env python3
"""Every Way local app.  Run from the repo root:  python3 everyway/server.py   then open http://localhost:8787
EVERYWAY_MOCK=1 python3 everyway/server.py   tries the whole flow with local stand-ins (no Texel, no credits)."""
import json, mimetypes, os, re, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline, texel

PORT = int(os.environ.get("EVERYWAY_PORT", "8787"))
OUT = Path(texel.OUT).resolve()
_run = {"thread": None}


def latest_state():
    if pipeline.STATE.get("lesson"):
        return pipeline.STATE
    runs = sorted((OUT / "runs").glob("*/state.json"), key=lambda p: p.stat().st_mtime) if (OUT / "runs").exists() else []
    if runs:
        try:
            s = json.loads(runs[-1].read_text()); s["from_disk"] = True
            return s
        except Exception:
            pass
    return {"status": "idle"}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        b = json.dumps(obj, default=str).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        path = self.path.split("?")[0]
        query = self.path.split("?", 1)[1] if "?" in self.path else ""
        if path in ("/", "/index.html"):
            b = (Path(__file__).parent / "ui.html").read_bytes()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b); return
        if path == "/api/runs":
            out = []
            for f in sorted((OUT / "runs").glob("*/state.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:30] if (OUT / "runs").exists() else []:
                try:
                    d = json.loads(f.read_text())
                    if d.get("status") == "done" and any(L.get("final") for L in d.get("learners", [])):
                        out.append({"run": f.parent.name, "lesson": d.get("lesson"), "for": [L.get("name") for L in d.get("learners", [])]})
                except Exception:
                    pass
            return self._json(200, {"runs": out})
        if path == "/api/state" and query.startswith("run="):
            name = re.sub(r"[^A-Za-z0-9._-]", "", query[4:].split("&")[0])
            files = sorted((OUT / "runs").glob("*/state.json"), key=lambda x: x.stat().st_mtime, reverse=True) if (OUT / "runs").exists() else []
            pick = None
            for f in files:
                if name == "latest" or f.parent.name == name:
                    try:
                        d = json.loads(f.read_text())
                    except Exception:
                        continue
                    if name != "latest" or (d.get("status") == "done" and any(L.get("final") for L in d.get("learners", []))):
                        pick = d; pick["run"] = f.parent.name; break
            if not pick:
                return self._json(404, {"error": "No finished lesson found yet. Make one first."})
            pick.update(mock=texel.MOCK, busy=False, pinned=True, has_texel_key=True)
            return self._json(200, pick)
        if path == "/api/state":
            s = dict(latest_state()); s["mock"] = texel.MOCK; s["busy"] = bool(_run["thread"] and _run["thread"].is_alive())
            s["has_texel_key"] = bool(texel.load_env().get("TEXEL_API_KEY")); s["has_agent_key"] = bool(texel.load_env().get("ANTHROPIC_API_KEY"))
            return self._json(200, s)
        if path.startswith("/media/"):
            f = (OUT / path[len("/media/"):]).resolve()
            if OUT not in f.parents or not f.is_file():
                return self._json(404, {"error": "not found"})
            size = f.stat().st_size
            ctype = mimetypes.guess_type(str(f))[0] or "application/octet-stream"
            start, end = 0, size - 1
            m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    start = int(m.group(1)); end = int(m.group(2)) if m.group(2) else size - 1
                else:
                    start = max(0, size - int(m.group(2)))
                end = min(end, size - 1)
                self.send_response(206); self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
            else:
                self.send_response(200)
            self.send_header("Content-Type", ctype); self.send_header("Accept-Ranges", "bytes"); self.send_header("Content-Length", str(end - start + 1)); self.end_headers()
            try:
                with open(f, "rb") as fh:
                    fh.seek(start); left = end - start + 1
                    while left > 0:
                        chunk = fh.read(min(1 << 16, left))
                        if not chunk:
                            break
                        self.wfile.write(chunk); left -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path not in ("/api/run", "/api/intake", "/api/plan"):
            return self._json(404, {"error": "not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0) or b"{}")
        except Exception:
            return self._json(400, {"error": "bad request"})
        topic = str(body.get("topic", "")).strip()[:300]
        if self.path == "/api/intake":
            if not topic:
                return self._json(400, {"error": "Say or type what you want to learn."})
            return self._json(200, pipeline.intake(topic, str(body.get("who", "")).strip()[:300]))
        if self.path == "/api/plan":
            people = []
            for l in (body.get("learners") or [])[:3]:
                ans = {k: str(v)[:120] for k, v in (l.get("answers") or {}).items() if k in ("level", "angle", "hook") and v}
                people.append({"name": str(l.get("name", "")).strip()[:40], "profile": str(l.get("profile", "")).strip()[:300], "answers": ans})
            if not topic or not people:
                return self._json(400, {"error": "A topic and at least one person are needed."})
            fb = str(body.get("feedback", "")).strip()[:400] or None
            prev = _run.get("pending", {}).get("plans") if fb else None
            plans = pipeline.make_plans(topic, people, fb, prev)
            if not plans:
                return self._json(502, {"error": "The planner agent did not answer. Check ANTHROPIC_API_KEY in .env, or use the saved demo."})
            _run["pending"] = {"topic": topic, "people": people, "plans": plans}
            return self._json(200, {"topic": topic, "plans": plans, "shots": sum(len(p["shots"]) for p in plans)})
        if _run["thread"] and _run["thread"].is_alive():
            return self._json(409, {"error": "A video is already being made. Wait for it to finish."})
        static = bool(body.get("static"))
        pending = _run.get("pending")
        if not static and not pending:
            return self._json(400, {"error": "There is no approved plan yet."})
        lesson = "" if static else pending["topic"]
        learners = [] if static else pending["people"]
        plans = None if static else pending["plans"]

        def work():
            try:
                pipeline.run(lesson, learners, use_static=static, plans=plans)
            except SystemExit as e:
                pipeline.STATE.update({"status": "failed", "error": str(e)}); pipeline.touch()
            except Exception as e:
                pipeline.STATE.update({"status": "failed", "error": repr(e)}); pipeline.touch()

        _run["thread"] = threading.Thread(target=work, daemon=True); _run["thread"].start()
        self._json(200, {"ok": True})


if __name__ == "__main__":
    print("Every Way is running at http://localhost:%d  (%s)" % (PORT, "STAND-IN MODE, not Texel" if texel.MOCK else "Texel API"))
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
