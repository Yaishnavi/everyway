#!/usr/bin/env python3
"""Every Way: overnight run against the Texel API.

Stage A  smoke test of every endpoint we plan to use (image, 2 video models, billing, edit key).
Stage B  generates the stills and clips for one lesson and two learners (lesson.json).

Run from the repo root on the Mac:   caffeinate -i python3 everyway/night.py
Reads TEXEL_API_KEY (required) and TEXEL_EDIT_API_KEY (optional) from .env. Never prints keys.
Safe to re-run: finished files are skipped. Never more than 2 requests in flight.
Standard library only (Python 3.9+).
"""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import texel
from texel import HERE, REPO, OUT, BASE, MAX_CLIPS, CLIP_SECONDS, load_env, log, trunc, save_raw, api, gen_image, gen_video, _lock

ONLY_SMOKE = "--smoke" in sys.argv
REPORT = {"started": None, "base": BASE, "smoke": {}, "batch": [], "notes": []}


def save_report():
    with _lock:
        (OUT / "report.json").write_text(json.dumps(trunc(REPORT, 400), indent=1, default=str))


def today_range():
    d = datetime.now(timezone.utc).date()
    return "start_date=%s&end_date=%s" % ((d - timedelta(days=2)).isoformat(), (d + timedelta(days=1)).isoformat())


def smoke(env):
    key = env["TEXEL_API_KEY"]
    S = REPORT["smoke"]
    sd = OUT / "smoke"
    log("STAGE A: smoke test against %s" % BASE)

    log("A1 image, Realistic SDXL")
    S["image_realistic"] = gen_image(key, "cinematic photo of a red vintage bicycle leaning on a white wall, morning light, 35mm film look",
                                     "lowres, blurry, text, watermark", "Realistic", sd / "img_realistic.jpg")
    save_report()
    if not S["image_realistic"]["ok"]:
        log("STOP: image generation failed. See everyway/out/report.json. Nothing else was attempted.")
        return False
    log("IMAGE OK in %s s" % S["image_realistic"]["seconds"])

    log("A2 image, Anime SDXL")
    S["image_anime"] = gen_image(key, "masterpiece, best quality, scenery, a small lighthouse on a cliff at sunset, anime style, no humans",
                                 "lowres, text, watermark, worst quality", "Anime", sd / "img_anime.jpg")
    log("  anime image ok=%s" % S["image_anime"]["ok"])
    save_report()

    log("A3 two image-to-video models at once (framepack and ltxv), 2 s each. Expect several minutes.")
    src = sd / "img_realistic.jpg"
    with ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(gen_video, key, "framepack", "the bicycle wheel turns slowly, leaves blow past, soft light flicker", "blurry, distorted", src, sd / "vid_framepack.mp4", 2.0, 832, 480, "smoke")
        f2 = ex.submit(gen_video, key, "ltxv", "the bicycle wheel turns slowly, leaves blow past, soft light flicker", "low quality, worst quality, deformed, distorted, motion smear", src, sd / "vid_ltxv.mp4", 2.0, 832, 480, "smoke")
        S["video_framepack"], S["video_ltxv"] = f1.result(), f2.result()
    for k in ("video_framepack", "video_ltxv"):
        log("  %s ok=%s in %s s %s" % (k, S[k]["ok"], S[k].get("seconds"), "" if S[k]["ok"] else str(S[k].get("error"))[:200]))
    save_report()

    log("A4 billing endpoint with the generation key")
    st, body = api("GET", "/billing/usage?" + today_range(), key, timeout=60)
    S["billing_gen_key"] = {"status": st, "body": trunc(body, 300)}
    save_raw("billing_gen_key", body)

    ek = env.get("TEXEL_EDIT_API_KEY")
    if not ek:
        S["edit_key"] = "TEXEL_EDIT_API_KEY not set in .env: video editing, lip sync and eye contact were not tested"
        log("A5 skipped: no TEXEL_EDIT_API_KEY in .env")
    else:
        log("A5 edit key: auth check, billing, and one encode probe")
        st, body = api("GET", "/video_encoder/status/everyway-auth-check", ek, timeout=60)
        S["edit_auth"] = {"status": st, "body": trunc(body, 300)}
        st, body = api("GET", "/billing/usage?" + today_range(), ek, timeout=60)
        S["billing_edit_key"] = {"status": st, "body": trunc(body, 300)}
        save_raw("billing_edit_key", body)
        vid = S.get("video_framepack") if S.get("video_framepack", {}).get("url") else S.get("video_ltxv")
        if vid and vid.get("url"):
            probe = {"inputs": [{"url": vid["url"], "label": "in", "content_type": "video/mp4"}],
                     "filters": [{"name": "scale", "inputs": ["in.video"], "args": ["1280", "720"], "label": "scaled"}],
                     "client_job_id": "everyway-probe-%d" % int(time.time())}
            st, body = api("POST", "/video_encoder/encode", ek, probe, timeout=120)
            S["encode_probe_no_output"] = {"status": st, "body": trunc(body, 500)}
            save_raw("encode_probe", {"status": st, "body": body})
            jid = body.get("job_id") or body.get("id") if isinstance(body, dict) else None
            if jid:
                for _ in range(12):
                    time.sleep(10)
                    st, b2 = api("GET", "/video_encoder/status/%s" % jid, ek, timeout=60)
                    S["encode_probe_status"] = {"status": st, "body": trunc(b2, 500)}
                    if isinstance(b2, dict) and str(b2.get("job_status", "")).lower() in ("success", "completed", "failed", "error"):
                        break
            log("  encode probe HTTP %s" % S["encode_probe_no_output"]["status"])
    save_report()
    return True


def pick_model():
    S = REPORT["smoke"]
    forced = os.environ.get("EVERYWAY_VIDEO_MODEL")
    if forced:
        return forced
    if S.get("video_framepack", {}).get("ok"):
        return "framepack"
    if S.get("video_ltxv", {}).get("ok"):
        return "ltxv"
    return None


def batch(env):
    key = env["TEXEL_API_KEY"]
    lesson = json.loads((HERE / "lesson.json").read_text())
    model = pick_model()
    log("STAGE B: lesson '%s', %d learners, video model: %s, cap %d clips" % (lesson["lesson"], len(lesson["learners"]), model, MAX_CLIPS))

    jobs = []
    for L in lesson["learners"]:
        for shot in L["shots"]:
            d = OUT / L["id"]
            jobs.append({"learner": L["id"], "shot": shot["id"], "L": L, "S": shot,
                         "img": d / (shot["id"] + ".jpg"), "vid": d / (shot["id"] + ".mp4")})

    log("B1 stills (%d), one at a time" % len(jobs))
    for j in jobs:
        r = gen_image(key, j["S"]["image"], j["L"]["negative"], j["L"]["image_model"], j["img"])
        j["image_result"] = r
        log("  %s/%s image ok=%s %s" % (j["learner"], j["shot"], r["ok"], "(already there)" if r.get("skipped") else "%s s" % r.get("seconds")))
        REPORT["batch"] = [{k: v for k, v in x.items() if k in ("learner", "shot", "image_result", "video_result")} for x in jobs]
        save_report()

    if not model:
        REPORT["notes"].append("No video model passed the smoke test, so no clips were generated. Stills only.")
        log("B2 skipped: no working video model")
        return
    todo = [j for j in jobs if j["image_result"]["ok"]][:MAX_CLIPS]
    log("B2 clips (%d), two at a time, about %.0f s each" % (len(todo), CLIP_SECONDS))

    def work(j):
        r = gen_video(key, model, j["S"]["motion"], j["L"]["negative"], j["img"], j["vid"], CLIP_SECONDS, 832, 480, "%s/%s" % (j["learner"], j["shot"]))
        j["video_result"] = r
        log("  %s/%s clip ok=%s in %s s %s" % (j["learner"], j["shot"], r["ok"], r.get("seconds"), "" if r["ok"] else str(r.get("error"))[:200]))
        REPORT["batch"] = [{k: v for k, v in x.items() if k in ("learner", "shot", "image_result", "video_result")} for x in jobs]
        save_report()

    with ThreadPoolExecutor(max_workers=2) as ex:
        list(ex.map(work, todo))


def write_md():
    S, lines = REPORT["smoke"], ["# Every Way: overnight report", "", "Started %s, finished %s" % (REPORT["started"], datetime.now().isoformat(timespec="seconds")), "", "## Smoke test", ""]
    for k, v in S.items():
        if isinstance(v, dict) and "ok" in v:
            lines.append("- %s: %s, %s s%s" % (k, "OK" if v["ok"] else "FAILED", v.get("seconds"), "" if v["ok"] else " | " + str(v.get("error"))[:300]))
        else:
            lines.append("- %s: %s" % (k, str(v)[:400]))
    lines += ["", "## Lesson assets", ""]
    for b in REPORT["batch"]:
        ir, vr = b.get("image_result", {}), b.get("video_result", {})
        lines.append("- %s/%s: still %s | clip %s%s" % (b["learner"], b["shot"], "OK" if ir.get("ok") else "FAILED",
                     ("OK (%s, %s s)" % (vr.get("model"), vr.get("seconds"))) if vr.get("ok") else ("FAILED" if vr else "not attempted"),
                     "" if vr.get("ok") or not vr else " | " + str(vr.get("error"))[:200]))
    for n in REPORT["notes"]:
        lines += ["", "Note: " + n]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT["started"] = datetime.now().isoformat(timespec="seconds")
    env = load_env()
    if not env.get("TEXEL_API_KEY"):
        print("TEXEL_API_KEY is missing. Add a line TEXEL_API_KEY=... to %s and run again." % (REPO / ".env"))
        sys.exit(2)
    try:
        ok = smoke(env)
        if ok and not ONLY_SMOKE:
            batch(env)
    except KeyboardInterrupt:
        REPORT["notes"].append("Stopped by Ctrl-C.")
    except Exception as e:
        import traceback
        REPORT["notes"].append("Crashed: %r" % (e,))
        log("CRASH: " + traceback.format_exc())
    save_report()
    write_md()
    log("DONE. Report: everyway/out/REPORT.md")


if __name__ == "__main__":
    main()
