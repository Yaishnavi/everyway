#!/usr/bin/env python3
"""Shared Texel API client for Every Way (used by night.py, pipeline.py and the driver adapter).

Reads TEXEL_API_KEY / TEXEL_EDIT_API_KEY from the repo .env. Never prints keys.
Never more than 2 generation requests in flight (Texel asked for that). Standard library only.
Set EVERYWAY_MOCK=1 to use a local stand-in that draws placeholder media with ffmpeg (no network, no credits).
"""
import base64, json, os, random, ssl, sys, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = Path(os.environ.get("EVERYWAY_OUT", HERE / "out"))
BASE = os.environ.get("TEXEL_BASE", "https://api.prod.texel.ai/v1").rstrip("/")
MAX_CLIPS = int(os.environ.get("EVERYWAY_MAX_CLIPS", "10"))      # stage B cap (credits are not readable via API)
CLIP_SECONDS = float(os.environ.get("EVERYWAY_CLIP_SECONDS", "5"))
POLL = float(os.environ.get("EVERYWAY_POLL", "15"))
VIDEO_TIMEOUT = float(os.environ.get("EVERYWAY_VIDEO_TIMEOUT", "720"))   # 12 min per clip, then give up (or fail over)

IMAGE_MODELS = {"Realistic": "juggernautXL_v8Rundiffusion.safetensors", "Anime": "animagine-xl-3.1.safetensors", "Sketch": "juggernautXL_v8Rundiffusion.safetensors"}
VIDEO_MODELS = {
    "framepack": {"name": "FramePackI2V_HY_fp8_e4m3fn", "type": "FRAMEPACK_VIDEO"},
    "ltxv": {"name": "ltxv-13b-0.9.7-dev-fp8", "type": "LTX_VIDEO"},
    "wan": {"name": "wan2.1_i2v_480p_14B_fp8_scaled", "type": "WAN_VIDEO"},
}
_lock = threading.Lock()
SLOTS = threading.BoundedSemaphore(2)          # hard cap: 2 generation requests in flight
TALLY = {"images": 0, "clips": 0, "image_seconds": 0.0, "clip_seconds": 0.0, "failed": 0, "backup": 0}
MOCK = os.environ.get("EVERYWAY_MOCK") == "1"


def load_env():
    env = {}
    p = REPO / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip().replace("export ", "")] = v.strip().strip('"').strip("'")
    for k in ("TEXEL_API_KEY", "TEXEL_EDIT_API_KEY"):
        if os.environ.get(k):
            env[k] = os.environ[k]
    return env


def log(msg):
    line = "[%s] %s" % (datetime.now().strftime("%H:%M:%S"), msg)
    with _lock:
        print(line, flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        with open(OUT / "night.log", "a") as f:
            f.write(line + "\n")


def trunc(o, n=200):
    if isinstance(o, str):
        return o if len(o) <= n else o[:60] + "...[%d chars]" % len(o)
    if isinstance(o, list):
        return [trunc(x, n) for x in o]
    if isinstance(o, dict):
        return {k: trunc(v, n) for k, v in o.items()}
    return o


def save_raw(name, obj):
    p = OUT / "raw" / (name + ".json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(trunc(obj), indent=1, default=str))


def _ctx():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def api(method, path, key, payload=None, timeout=240):
    """Returns (status, body). body is parsed JSON when possible, else text. status 0 = network error."""
    url = path if path.startswith("http") else BASE + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", "Bearer " + key)
    req.add_header("accept", "application/json")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_ctx()) as r:
            raw, status = r.read(), r.status
    except urllib.error.HTTPError as e:
        raw, status = e.read(), e.code
    except Exception as e:
        return 0, "network error: %r" % (e,)
    try:
        return status, json.loads(raw)
    except Exception:
        return status, raw[:1000].decode("utf-8", "replace")


def download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=300, context=_ctx()) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
    tmp.rename(dest)


def _gen_image_real(key, prompt, negative, model, dest, width=1344, height=768, steps=25, cfg=6.0, seed=None):
    """Synchronous. Returns dict(ok, seconds, error)."""
    if dest.exists() and dest.stat().st_size > 2000:
        return {"ok": True, "seconds": 0, "skipped": True}
    seed = seed or random.randint(1, 2147483647)
    payload = {
        "model": {"name": IMAGE_MODELS.get(model, model), "type": "SDXL"},
        "request": {"prompt": prompt, "negative_prompt": negative, "width": width, "height": height,
                    "steps": steps, "cfg_scale": cfg, "seed": seed},
        "params": {}, "timeout": 120, "callback": {"url": ""},
    }
    last = None
    for attempt in (1, 2):
        t0 = time.time()
        status, body = api("POST", "/sd_server/txt2img", key, payload, timeout=300)
        secs = round(time.time() - t0, 1)
        save_raw("img_%s_a%d" % (dest.stem, attempt), {"status": status, "body": body})
        if status == 200 and isinstance(body, dict) and body.get("images"):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(base64.b64decode(body["images"][0]))
            return {"ok": True, "seconds": secs, "seed": seed, "response_keys": sorted(body.keys())}
        last = {"ok": False, "seconds": secs, "status": status, "error": trunc(body, 400)}
        log("  image attempt %d failed (HTTP %s): %s" % (attempt, status, str(trunc(body, 300))[:300]))
        time.sleep(5)
    return last


def video_payload(model, prompt, negative, image_b64, width, height, seconds, seed):
    m = VIDEO_MODELS[model]
    fps_frames = int(round(seconds * 16)) + 1            # wan: 16 fps, 81 frames = 5 s
    if model == "framepack":
        req = {"cfg_scale": 3.0, "steps": 30, "width": width, "height": height, "length": float(seconds),
               "prompt": prompt, "video_codec": "video/h264-mp4", "seed": seed}
        mod = {"name": m["name"], "type": m["type"], "params": {}}
    elif model == "ltxv":
        req = {"width": width, "height": height, "frames": int(round(seconds * 24)) + 1, "seed": seed,
               "profile": "13b Dynamic", "prompt": prompt, "video_codec": "video/h264-mp4", "negative_prompt": negative}
        mod = {"name": m["name"], "type": m["type"]}
    else:  # wan image to video
        req = {"cfg_scale": 6.0, "steps": 30, "width": width, "height": height, "frames": fps_frames,
               "prompt": prompt, "video_codec": "video/h264-mp4", "seed": seed}
        mod = {"name": m["name"], "type": m["type"], "params": {}}
    req["init_images"] = [image_b64]
    return {"model": mod, "request": req}


def _gen_video_real(key, model, prompt, negative, image_path, dest, seconds=5.0, width=832, height=480, tag=""):
    """Image to video. Submits, polls, downloads. Returns dict(ok, seconds, error, url)."""
    if dest.exists() and dest.stat().st_size > 20000:
        return {"ok": True, "seconds": 0, "skipped": True, "model": model}
    seed = random.randint(1, 2147483647)
    img64 = base64.b64encode(Path(image_path).read_bytes()).decode()
    t0 = time.time()
    status, body = api("POST", "/sd_server/img2vid", key, video_payload(model, prompt, negative, img64, width, height, seconds, seed), timeout=300)
    save_raw("vid_submit_%s" % dest.stem, {"status": status, "body": body})
    job = body.get("id") or body.get("job_id") if isinstance(body, dict) else None
    if status not in (200, 201, 202) or not job:
        return {"ok": False, "model": model, "status": status, "error": trunc(body, 400), "seconds": round(time.time() - t0, 1)}
    log("  %s submitted %s job %s" % (tag or dest.stem, model, job))
    mtype = VIDEO_MODELS[model]["type"]
    last_body, bad = None, 0
    while time.time() - t0 < VIDEO_TIMEOUT:
        time.sleep(POLL)
        status, body = api("GET", "/sd_server/status/%s/%s" % (job, mtype), key, timeout=60)
        if status != 200 or not isinstance(body, dict):
            bad += 1
            if bad >= 8:
                return {"ok": False, "model": model, "job": job, "error": "status endpoint failing: HTTP %s %s" % (status, str(body)[:200]), "seconds": round(time.time() - t0, 1)}
            continue
        bad, last_body = 0, body
        js = str(body.get("job_status", "")).lower()
        urls = body.get("signed_output_urls") or []
        if js == "success" and urls:
            save_raw("vid_done_%s" % dest.stem, body)
            try:
                download(urls[0], dest)
            except Exception as e:
                return {"ok": False, "model": model, "job": job, "error": "download failed: %r" % (e,), "seconds": round(time.time() - t0, 1)}
            return {"ok": True, "model": model, "job": job, "seconds": round(time.time() - t0, 1), "url": urls[0], "status_keys": sorted(body.keys())}
        if js in ("error", "failed", "timeout", "cancelled") or body.get("error_message"):
            save_raw("vid_fail_%s" % dest.stem, body)
            return {"ok": False, "model": model, "job": job, "error": trunc(body, 400), "seconds": round(time.time() - t0, 1)}
    save_raw("vid_timeout_%s" % dest.stem, last_body)
    return {"ok": False, "model": model, "job": job, "error": "no result after %d s" % VIDEO_TIMEOUT, "last": trunc(last_body, 300), "seconds": round(time.time() - t0, 1)}




# ---------- backup engine (fal.ai), used only when Texel fails a request and FAL_KEY is set ----------
FAL_QUEUE = os.environ.get("FAL_QUEUE", "https://queue.fal.run").rstrip("/")
FAL_IMAGE = os.environ.get("EVERYWAY_BACKUP_IMAGE", "fal-ai/flux/dev")
FAL_VIDEO = os.environ.get("EVERYWAY_BACKUP_VIDEO", "fal-ai/kling-video/v3/pro/image-to-video")
BACKUP_ON = os.environ.get("EVERYWAY_FALLBACK", "1") != "0"


def backup_key():
    return load_env().get("FAL_KEY") or os.environ.get("FAL_KEY") or ""


def _fal(method, url, key, payload=None, timeout=120):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", "Key " + key)
    req.add_header("accept", "application/json")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_ctx()) as r:
            raw, status = r.read(), r.status
    except urllib.error.HTTPError as e:
        raw, status = e.read(), e.code
    except Exception as e:
        return 0, "network error: %r" % (e,)
    try:
        return status, json.loads(raw)
    except Exception:
        return status, raw[:600].decode("utf-8", "replace")


def _fal_run(model, args, key, max_wait=900, tag=""):
    """Submit to the fal queue, poll, return (ok, result_or_error)."""
    st, sub = _fal("POST", "%s/%s" % (FAL_QUEUE, model), key, args)
    if st not in (200, 201, 202) or not isinstance(sub, dict) or not sub.get("request_id"):
        return False, "submit HTTP %s %s" % (st, str(trunc(sub, 300))[:300])
    status_url = sub.get("status_url") or "%s/%s/requests/%s/status" % (FAL_QUEUE, model, sub["request_id"])
    response_url = sub.get("response_url") or "%s/%s/requests/%s" % (FAL_QUEUE, model, sub["request_id"])
    log("  %s backup engine job %s on %s" % (tag, sub["request_id"], model))
    t0 = time.time()
    while time.time() - t0 < max_wait:
        time.sleep(min(POLL, 5))
        st, body = _fal("GET", status_url, key)
        state = str(body.get("status", "")).upper() if isinstance(body, dict) else ""
        if state == "COMPLETED":
            st, res = _fal("GET", response_url, key)
            if st == 200 and isinstance(res, dict):
                return True, res
            return False, "result HTTP %s %s" % (st, str(trunc(res, 300))[:300])
        if state in ("FAILED", "ERROR", "CANCELLED") or (isinstance(body, dict) and body.get("error")):
            return False, str(trunc(body, 300))[:300]
    return False, "backup engine gave no result after %d s" % max_wait


def _backup_image(prompt, negative, dest, width, height):
    key = backup_key()
    t0 = time.time()
    ok, res = _fal_run(FAL_IMAGE, {"prompt": prompt, "image_size": {"width": width, "height": height}, "num_images": 1, "enable_safety_checker": True}, key, 300, dest.stem)
    save_raw("backup_img_%s" % dest.stem, {"ok": ok, "res": res})
    try:
        url = res["images"][0]["url"] if ok else None
        if url:
            download(url, dest)
            return {"ok": True, "seconds": round(time.time() - t0, 1), "engine": "backup (%s)" % FAL_IMAGE}
    except Exception as e:
        res = "download failed: %r" % (e,)
    return {"ok": False, "seconds": round(time.time() - t0, 1), "error": str(res)[:300], "engine": "backup"}


def _backup_video(prompt, negative, image_path, dest, seconds):
    key = backup_key()
    t0 = time.time()
    img = "data:image/jpeg;base64," + base64.b64encode(Path(image_path).read_bytes()).decode()
    args = {"prompt": prompt, "start_image_url": img, "duration": str(int(max(3, min(15, round(seconds))))), "generate_audio": False,
            "negative_prompt": negative or "blur, distort, and low quality"}
    ok, res = _fal_run(FAL_VIDEO, args, key, 900, dest.stem)
    save_raw("backup_vid_%s" % dest.stem, {"ok": ok, "res": res})
    try:
        url = res["video"]["url"] if ok else None
        if url:
            download(url, dest)
            return {"ok": True, "seconds": round(time.time() - t0, 1), "engine": "backup (%s)" % FAL_VIDEO, "model": FAL_VIDEO, "url": url}
    except Exception as e:
        res = "download failed: %r" % (e,)
    return {"ok": False, "seconds": round(time.time() - t0, 1), "error": str(res)[:300], "engine": "backup", "model": FAL_VIDEO}


# ---------- local stand-in (EVERYWAY_MOCK=1): placeholder media, clearly labelled, no network ----------
def _ff():
    import shutil
    return shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"


def _mock_image(prompt, dest, width, height):
    import hashlib, subprocess
    h = hashlib.md5(prompt.encode()).hexdigest()
    col = "0x%s" % h[:6]
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([_ff(), "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "gradients=s=%dx%d:c0=%s:c1=0x%s:x0=0:y0=0:x1=%d:y1=%d" % (width, height, col, h[6:12], width, height),
                    "-frames:v", "1", str(dest)], check=False)
    if not dest.exists():
        subprocess.run([_ff(), "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=%s:s=%dx%d" % (col, width, height),
                        "-frames:v", "1", str(dest)], check=True)
    time.sleep(float(os.environ.get("EVERYWAY_MOCK_DELAY", "0.4")))


def _mock_video(image_path, dest, seconds, width, height):
    import subprocess
    dest.parent.mkdir(parents=True, exist_ok=True)
    frames = int(seconds * 24)
    subprocess.run([_ff(), "-y", "-loglevel", "error", "-loop", "1", "-i", str(image_path), "-vf",
                    "scale=%d:%d,zoompan=z='min(zoom+0.0012,1.2)':d=%d:s=%dx%d:fps=24" % (width * 2, height * 2, frames, width, height),
                    "-frames:v", str(frames), "-pix_fmt", "yuv420p", str(dest)], check=True)
    time.sleep(float(os.environ.get("EVERYWAY_MOCK_DELAY", "0.4")) * 3)


def gen_image(key, prompt, negative, model, dest, width=1344, height=768, steps=25, cfg=6.0, seed=None):
    """Text to image on Texel (SDXL). Holds one of the 2 request slots. Returns dict(ok, seconds, ...)."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 2000:
        return {"ok": True, "seconds": 0, "skipped": True}
    with SLOTS:
        if MOCK:
            t0 = time.time(); _mock_image(prompt, dest, width, height)
            r = {"ok": True, "seconds": round(time.time() - t0, 1), "mock": True}
        else:
            r = _gen_image_real(key, prompt, negative, model, dest, width, height, steps, cfg, seed) if key else {"ok": False, "error": "no Texel key"}
            r.setdefault("engine", "Texel")
            if not r.get("ok") and BACKUP_ON and backup_key():
                log("  Texel image failed, switching this picture to the backup engine")
                first = r
                r = _backup_image(prompt, negative, dest, width, height)
                r["texel_error"] = str(first.get("error"))[:200]
    with _lock:
        if r.get("ok"):
            TALLY["images"] += 1; TALLY["image_seconds"] += r.get("seconds") or 0
            if str(r.get("engine", "")).startswith("backup"):
                TALLY["backup"] += 1
        else:
            TALLY["failed"] += 1
    return r


def gen_video(key, model, prompt, negative, image_path, dest, seconds=5.0, width=832, height=480, tag=""):
    """Image to video on Texel. Holds one of the 2 request slots until the clip is downloaded."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 20000:
        return {"ok": True, "seconds": 0, "skipped": True, "model": model}
    with SLOTS:
        if MOCK:
            t0 = time.time(); _mock_video(image_path, dest, seconds, width, height)
            r = {"ok": True, "seconds": round(time.time() - t0, 1), "model": model, "mock": True}
        else:
            if model == "none":
                r = {"ok": False, "error": "Texel video is switched off (EVERYWAY_VIDEO_MODEL=none)", "model": model}
            else:
                r = _gen_video_real(key, model, prompt, negative, image_path, dest, seconds, width, height, tag) if key else {"ok": False, "error": "no Texel key", "model": model}
            r.setdefault("engine", "Texel")
            if not r.get("ok") and BACKUP_ON and backup_key():
                log("  %s Texel clip failed, switching this clip to the backup engine" % tag)
                first = r
                r = _backup_video(prompt, negative, image_path, dest, seconds)
                r["texel_error"] = str(first.get("error"))[:200]
    with _lock:
        if r.get("ok"):
            TALLY["clips"] += 1; TALLY["clip_seconds"] += r.get("seconds") or 0
            if str(r.get("engine", "")).startswith("backup"):
                TALLY["backup"] += 1
        else:
            TALLY["failed"] += 1
    return r
