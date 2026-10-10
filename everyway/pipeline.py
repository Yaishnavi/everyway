#!/usr/bin/env python3
"""Every Way pipeline: one lesson, made differently for every student, with a critic that rejects bad takes.

  planner  (Claude)  lesson + learner profile -> script and shot list for that learner
  maker    (Texel)   still per shot, then image-to-video clip          max 2 Texel requests in flight
  critic   (Claude)  looks at each still and each clip; accepts, or rejects with a reason and a rewritten prompt
  narrator           macOS `say` (silent track if unavailable)
  editor   (ffmpeg)  title card + shots timed to narration + captions + end card -> final.mp4

Run from the repo root:
  python3 everyway/pipeline.py --static                      # the built-in water cycle lesson, two learners
  python3 everyway/pipeline.py --lesson "Fractions" --learner "Ana: 10, loves baking" --learner "Sam: 13, plays basketball"
EVERYWAY_MOCK=1 runs everything with local stand-ins (no network, no credits). Standard library only.
"""
import argparse, base64, hashlib, json, os, re, shutil, subprocess, sys, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import texel
from texel import HERE, OUT, load_env, log, gen_image, gen_video

MAX_STILL_TAKES = int(os.environ.get("EVERYWAY_STILL_TAKES", "2"))
MAX_CLIP_TAKES = int(os.environ.get("EVERYWAY_CLIP_TAKES", "2"))
CLIP_SECONDS = float(os.environ.get("EVERYWAY_CLIP_SECONDS", "5"))
SHOTS_PER_LESSON = int(os.environ.get("EVERYWAY_SHOTS", "6"))
STYLE = os.environ.get("EVERYWAY_STYLE", "photo").lower()      # sketch = whiteboard explainer doodles; photo = the earlier photographic look
SKETCH_NEG = "photo, photograph, realistic, 3d render, shading, gradient, shadows, text, words, letters, numbers, writing, watermark, signature, busy background, clutter, frame, border, dark background"


def styled(L, prompt):
    """The prompt and negative actually sent to the image model for this learner's style."""
    if L.get("style") == "Sketch":
        return ("simple hand-drawn marker doodle of %s, thick black marker outlines, a few flat bright colours, pure white background, minimal whiteboard explainer illustration, clean, centred, lots of empty space" % prompt.rstrip(". "), SKETCH_NEG)
    return prompt, L.get("negative", "")
MODELS = [m for m in os.environ.get("EVERYWAY_MODEL", "claude-sonnet-5-5,claude-opus-4-1,claude-haiku-4-5-20251001").split(",") if m]
VOICES = ["Samantha", "Daniel", "Karen", "Alex"]
W, H, FPS = 1920, 1080, 24

STATE = {"status": "idle"}
_slock = threading.RLock()
_env = {}


# ------------------------------------------------------------------ state
def touch(run_dir=None):
    with _slock:
        STATE["tally"] = dict(texel.TALLY)
        STATE["updated"] = time.time()
        d = run_dir or STATE.get("run_dir")
        if d:
            try:
                Path(d, "state.json").write_text(json.dumps(STATE, indent=1, default=str))
            except Exception:
                pass


def note(msg):
    log(msg)
    with _slock:
        STATE.setdefault("log", []).append("%s  %s" % (datetime.now().strftime("%H:%M:%S"), msg))
        STATE["log"] = STATE["log"][-40:]
    touch()


def rel(p):
    try:
        return str(Path(p).resolve().relative_to(Path(OUT).resolve()))
    except Exception:
        return str(p)


# ------------------------------------------------------------------ Claude
def claude(system, content, max_tokens=1500):
    """content: str or list of blocks. Returns text, or None when no key / mock / every model fails."""
    key = load_env().get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")   # read fresh: the key may be added while the app runs
    if texel.MOCK:
        return None
    if not key:
        log("  agent call skipped: ANTHROPIC_API_KEY is empty in .env")
        return None
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    for model in MODELS:
        body = json.dumps({"model": model, "max_tokens": max_tokens, "system": system,
                           "messages": [{"role": "user", "content": content}]}).encode()
        req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body, method="POST")
        req.add_header("x-api-key", key); req.add_header("anthropic-version", "2023-06-01"); req.add_header("content-type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=120, context=texel._ctx()) as r:
                data = json.loads(r.read())
            STATE["agent_model"] = model
            return "".join(b.get("text", "") for b in data.get("content", []))
        except urllib.error.HTTPError as e:
            msg = e.read()[:200].decode("utf-8", "replace")
            log("  claude %s -> HTTP %s %s" % (model, e.code, msg))
            if e.code in (400, 404):
                continue          # unknown model id: try the next one
            time.sleep(3)
        except Exception as e:
            log("  claude %s -> %r" % (model, e))
    return None


def parse_json(text):
    if not text:
        return None
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j < 0:
        return None
    try:
        return json.loads(text[i:j + 1])
    except Exception:
        return None


# ------------------------------------------------------------------ planner
PLANNER_SYSTEM = """You are the planner in a studio that makes short personalised explainer videos for one specific person. Every picture is generated by an open-source image model (SDXL) and then animated for 5 seconds, so prompts must be simple, concrete and robust.
Return ONLY a JSON object:
{"title": str, "style": "Anime" | "Realistic", "negative": str, "approach": "<one sentence: the angle you chose for this person and why>",
 "shots": [{"id": "<short name>", "teaches": "<the one idea this shot must make visible>", "narration": str, "image": str, "motion": str}, ...]}
Rules:
- Exactly N shots. Together they teach the topic in order, at the person's level, focused on what they said they most want to understand. Explain it directly and concretely; use an everyday comparison only where it truly makes the idea clearer.
- If the person is described (age, interests, how they learn), let that shape everything: vocabulary for their age, examples from their interests, their name used once. A lesson for an 8-year-old who loves dinosaurs must sound and look different from one for a 15-year-old skater.
- Narration: one short spoken sentence per shot, at most 16 words, everyday words, factually correct. One idea per shot. Read aloud it should sound like a friend explaining, the way the best whiteboard explainer videos do. Use the person's name once if one is given.
- The image model is simple. Each image must be ONE literal, photographable scene with one main subject that it can actually draw: a leaf in sunlight, roots in wet soil, steam over a lake. The clever comparison lives in the NARRATION, never in the picture. Do not combine two worlds in one image (no leaf on a stove, no chef made of leaves). No diagrams, labels, arrows, formulas, cross-sections, microscopic views or text.
- "teaches" is a plain description of what is literally visible in that image, in at most 12 words.
- No children or teenagers, no real people, no brands or logos. People only when essential, and then adults.
- Pick the style that suits the person. "Anime" image prompts are comma separated tags starting with "masterpiece, best quality, very aesthetic" and ending with "anime style". "Realistic" image prompts start with "cinematic photo" and end with "35mm film look".
- One clear subject per image, wide or medium framing.
- motion: one sentence of gentle, physically simple movement for the animation model.
- negative: a comma separated negative prompt suited to the style."""

INTAKE_SYSTEM = """Someone wants a short personalised video that teaches a topic to a particular person (themselves unless they say otherwise). Before making it, ask what you need to know to make it right for THAT person. If they describe who it is for, word the questions and options for that person's age and interests.
Return ONLY JSON: {"topic": "<the topic, cleaned up>", "questions": [{"id": "level" | "angle", "question": str, "options": [str, str, str, str]}]}
Rules: exactly 2 questions. "level": how much they already know, with options specific to this topic. "angle": which part or question about the topic they most want answered, with options specific to this topic. Questions at most 12 words, options at most 6 words, plain language, no jargon the learner may not know."""

CHECKER_SYSTEM = """You are the fact checker in a studio that makes short explainer videos. You are given a shot list. Check every narration line for factual errors, misleading simplifications and anything unsuitable for the stated audience. Fix only what is wrong, keeping each line at most 22 words.
Return ONLY JSON: {"fixes": [{"shot": <1-based index>, "problem": "<at most 15 words>", "narration": "<corrected line>"}], "verdict": "<one sentence overall>"}. Return an empty fixes list when everything is correct."""


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:32] or "x"


def intake(topic, who=""):
    """Clarifying questions for a topic. Always returns something usable."""
    global _env
    _env = _env or load_env()
    q = parse_json(claude(INTAKE_SYSTEM, "Topic as the person typed or said it: %s%s" % (topic, ("\nWho it is for: " + who) if who else ""), 700))
    if q and q.get("questions"):
        q["questions"] = [x for x in q["questions"] if x.get("id") != "hook" and x.get("options")][:2]
        if q["questions"]:
            q["by"] = "intake agent"
            return q
    return {"topic": topic, "by": "stand-in" if texel.MOCK else "default questions (no agent key)", "questions": [
        {"id": "level", "question": "How much do you already know about this?", "options": ["Nothing yet", "The basics", "I studied it once", "A lot, go deeper"]},
        {"id": "angle", "question": "What do you most want to understand?", "options": ["What it is", "How it works step by step", "Why it matters", "A common confusion"]}]}


def describe(learner):
    """Turns a learner record into the text the planner and critic see."""
    bits = [learner.get("profile") or ""]
    for k, label in (("level", "Already knows"), ("angle", "Wants to understand"), ("hook", "Explain it through")):
        v = (learner.get("answers") or {}).get(k)
        if v:
            bits.append("%s: %s" % (label, v))
    return ". ".join(b for b in bits if b) or "a curious adult, no background assumed"


def plan_learner(lesson, learner, n_shots, feedback=None, previous=None):
    who = describe(learner)
    ask = "Topic: %s\nPerson: %s. %s" % (lesson, learner.get("name") or "(no name given)", who)
    if feedback and previous:
        ask += "\n\nYour previous plan:\n%s\n\nThe person asked for this change: %s\nReturn the full revised plan." % (json.dumps({"title": previous.get("title"), "shots": previous.get("shots")})[:3500], feedback)
    system = PLANNER_SYSTEM.replace("Exactly N shots", "Exactly %d shots" % n_shots)
    if STYLE == "sketch":
        system += """\n\nSTYLE OVERRIDE: every picture is a simple whiteboard doodle, like a hand-drawn explainer video. Set "style" to "Sketch". Write each "image" as a plain description of ONE simple drawing: which few objects, how they are arranged, and any arrows showing what moves where (for example: "a sun on the left with wavy rays reaching a single green leaf on the right"). At most 25 words, no style words, no text or labels in the drawing. Simple icons a child could draw: sun, leaf, water drop, cloud, arrow, stick figure. Ignore the earlier rules about Anime and Realistic prompt formats."""
    text = claude(system, ask, 3000)
    p = parse_json(text)
    if text and not (p and p.get("shots")):
        log("  planner reply could not be read as a plan, asking again. It began: %r" % text[:160])
        text = claude(system, ask + "\n\nReturn the JSON object only. No prose before or after it, no code fence.", 3000)
        p = parse_json(text)
        if text and not (p and p.get("shots")):
            log("  planner reply unreadable a second time. It began: %r" % text[:160])
    if (not p or not p.get("shots")) and texel.MOCK:
        # stand-in planner so the whole flow can be tried offline
        steps = ["what it is", "how it starts", "what happens next", "why it matters", "the whole picture"][:n_shots]
        p = {"title": lesson, "style": "Anime" if len(learner.get("name", "")) % 2 else "Realistic", "negative": "lowres, text, watermark",
             "approach": "stand-in plan%s" % (", revised: " + feedback if feedback else ""),
             "shots": [{"id": "s%d" % (i + 1), "teaches": st, "narration": "%s: %s, explained through %s." % (lesson, st, (learner.get("answers") or {}).get("hook", "everyday life")),
                        "image": "%s, %s" % (lesson, st), "motion": "slow gentle movement"} for i, st in enumerate(steps)]}
        STATE["agent_model"] = "stand-in"
    if not p or not p.get("shots"):
        return None
    p["style"] = "Sketch" if STYLE == "sketch" else ("Anime" if str(p.get("style", "")).lower().startswith("anime") else "Realistic")
    for i, s in enumerate(p["shots"]):
        s["id"] = "s%d_%s" % (i + 1, slug(str(s.get("id", "shot")))[:18])
    p["planned_by"] = "planner agent"
    return p


def check_plan(lesson, learner, p):
    """Fact checker pass over the narration. Mutates p; returns it with notes for the preview."""
    lines = "\n".join("%d. %s" % (i + 1, s["narration"]) for i, s in enumerate(p["shots"]))
    r = parse_json(claude(CHECKER_SYSTEM, "Topic: %s\nAudience: %s\nShot narration:\n%s" % (lesson, describe(learner), lines), 900))
    if not r:
        p["checked"] = {"by": "stand-in" if texel.MOCK else "none", "fixes": [], "verdict": "stand-in check" if texel.MOCK else "not checked"}
        return p
    fixes = []
    for f in r.get("fixes") or []:
        try:
            i = int(f["shot"]) - 1
            if 0 <= i < len(p["shots"]) and f.get("narration"):
                fixes.append({"shot": i + 1, "problem": str(f.get("problem", ""))[:140], "was": p["shots"][i]["narration"]})
                p["shots"][i]["narration"] = str(f["narration"])
        except Exception:
            pass
    p["checked"] = {"by": "fact checker agent", "fixes": fixes, "verdict": str(r.get("verdict", ""))[:200]}
    return p


COMPANION_SYSTEM = """You write the short written lesson that sits under a personalised explainer video. Same person, same angle as the video, but the page can go one step deeper than the narration.
Return ONLY JSON:
{"overview": [str, str], "key_points": [str, str, str], "terms": [{"term": str, "meaning": str}], "quiz": [{"q": str, "options": [str, str, str], "answer": 0, "why": str}], "illustrations": [{"caption": str, "image": str}, {"caption": str, "image": str}], "sources": [str, str, str]}
Rules:
- overview: two short paragraphs, at most 60 words each, second person, plain language, factually correct, using the person's chosen world for the comparison.
- key_points: three one-sentence takeaways. terms: three or four key words with a one-line meaning each.
- quiz: exactly three questions that check understanding, three options each, "answer" is the index of the right one, "why" explains it in one sentence.
- illustrations: exactly two. "image" is a prompt for a simple image model in the given style: ONE literal, photographable scene with one main subject, no text, labels, diagrams or people under 18. "caption" says what it shows and why it matters, at most 18 words.
- sources: three well-known, trustworthy places to read more about this topic (a standard textbook, an encyclopedia entry, a museum or university or agency site). Names only. Do not write URLs."""


def make_companion(lesson, L):
    """Written lesson + quiz + two extra illustration prompts for one learner. Returns dict or None."""
    style = L.get("style", "Realistic")
    ask = "Topic: %s\nPerson: %s. %s\nImage style: %s (%s)\nThe video's narration, in order:\n%s" % (
        lesson, L.get("name"), L.get("profile"), style,
        'a simple whiteboard doodle: write a plain description of one simple drawing, which few objects and how they are arranged, arrows allowed, no style words, no text' if style == "Sketch" else
        'prompts are comma separated tags starting with "masterpiece, best quality, very aesthetic" and ending with "anime style"' if style == "Anime" else 'prompts start with "cinematic photo" and end with "35mm film look"',
        "\n".join("- " + S["narration"] for S in L["shots"]))
    c = parse_json(claude(COMPANION_SYSTEM, ask, 2500))
    if not c and texel.MOCK:
        c = {"overview": ["Stand-in write-up about %s." % lesson, "A second stand-in paragraph."], "key_points": ["Stand-in point one.", "Stand-in point two.", "Stand-in point three."],
             "terms": [{"term": "Term", "meaning": "stand-in meaning"}], "quiz": [{"q": "Stand-in question?", "options": ["First", "Second", "Third"], "answer": 1, "why": "Stand-in reason."}],
             "illustrations": [{"caption": "Stand-in illustration one", "image": lesson + " one"}, {"caption": "Stand-in illustration two", "image": lesson + " two"}], "sources": ["A standard textbook", "An encyclopedia entry", "A museum site"]}
    if not c or not c.get("overview"):
        return None
    c["quiz"] = [q for q in (c.get("quiz") or []) if isinstance(q, dict) and isinstance(q.get("options"), list) and isinstance(q.get("answer"), int) and 0 <= q["answer"] < len(q["options"])][:3]
    c["illustrations"] = [i for i in (c.get("illustrations") or []) if isinstance(i, dict) and i.get("image")][:2]
    return c


DIAGRAM_SYSTEM = """You draw explanatory diagrams as SVG, in the style of a clear whiteboard explainer video: simple shapes, thick dark outlines, a few flat colours, short labels, arrows that show what moves where. A learner should understand the idea from the diagram alone.
Output exactly two diagrams, each in this form and nothing else:
CAPTION: <one sentence, at most 16 words, saying what the diagram shows>
<svg viewBox="0 0 800 450" xmlns="http://www.w3.org/2000/svg"> ... </svg>
Rules for every SVG:
- First element is <rect width="800" height="450" fill="#FFFFFF"/>. Everything stays inside the 800 by 450 box with a 30 px margin.
- Outlines: stroke="#1B1F2A" stroke-width="3" stroke-linecap="round" stroke-linejoin="round". Fills only from: #FFB85C, #5B8DEF, #63D2A1, #FF7D6E, #FFE08A, #FFFFFF.
- Text: font-family="Helvetica, Arial, sans-serif", font-size 20 or larger, fill="#1B1F2A", at most four words per label, never overlapping shapes or other text.
- Arrows: define one <marker> arrowhead in <defs> and use it. Arrows must start and end where they should.
- Use only rect, circle, ellipse, line, polyline, polygon, path, text, g, defs, marker. No scripts, images, links, foreignObject, filters or animation. At most 45 elements.
- Diagram 1 shows the whole idea at a glance (what goes in, what happens, what comes out). Diagram 2 zooms in on the one step people find hardest.
- Accuracy matters more than decoration. Label inputs and outputs with their real names.
After the two diagrams, if takeaways are listed in the request, output one small picture for each takeaway, in the same order, each in this form:
ICON:
<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg"> ... </svg>
Each icon is a simple drawing of that takeaway in the same style and colours: white background rect, thick dark outlines, no text at all, at most 12 elements."""


def clean_svg(svg):
    svg = re.sub(r"<script.*?</script>", "", svg, flags=re.S | re.I)
    svg = re.sub(r"<(foreignObject|image|iframe|style|animate\w*|set)\b.*?(</\1>|/>)", "", svg, flags=re.S | re.I)
    svg = re.sub(r"\son\w+\s*=\s*(\"[^\"]*\"|'[^']*')", "", svg, flags=re.I)
    svg = re.sub(r"\s(xlink:)?href\s*=\s*(\"(?!#)[^\"]*\"|'(?!#)[^']*')", "", svg, flags=re.I)
    return svg if svg.count("<") < 400 and len(svg) < 30000 else None


def make_diagrams(lesson, L, points=None):
    text = claude(DIAGRAM_SYSTEM, "Topic: %s\nLearner: %s\nThe explanation they are getting, in order:\n%s%s" % (
        lesson, L.get("profile"), "\n".join("- " + S["narration"] for S in L["shots"]),
        ("\n\nTakeaways that each need a small icon:\n" + "\n".join("%d. %s" % (i + 1, p) for i, p in enumerate(points))) if points else ""), 9000)
    if not text and texel.MOCK:
        text = 'ICON:\n<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg"><rect width="120" height="120" fill="#FFFFFF"/><circle cx="60" cy="60" r="34" fill="#FFB85C" stroke="#1B1F2A" stroke-width="3"/></svg>\nCAPTION: Stand-in diagram of the idea.\n<svg viewBox="0 0 800 450" xmlns="http://www.w3.org/2000/svg"><rect width="800" height="450" fill="#FFFFFF"/><circle cx="160" cy="225" r="70" fill="#FFB85C" stroke="#1B1F2A" stroke-width="3"/><rect x="520" y="155" width="180" height="140" rx="16" fill="#63D2A1" stroke="#1B1F2A" stroke-width="3"/><line x1="250" y1="225" x2="500" y2="225" stroke="#1B1F2A" stroke-width="3"/><text x="320" y="205" font-family="Helvetica, Arial, sans-serif" font-size="22" fill="#1B1F2A">stand-in</text></svg>'
    out = []
    for cap, svg in re.findall(r"CAPTION:\s*(.*?)\s*(<svg\b.*?</svg>)", text or "", flags=re.S):
        svg = clean_svg(svg)
        if svg:
            out.append({"caption": cap.strip()[:160], "svg": svg})
    icons = [clean_svg(x) for x in re.findall(r"ICON:\s*(<svg\b.*?</svg>)", text or "", flags=re.S)]
    return out[:2], [i for i in icons if i][:4]


def _wiki(url):
    req = urllib.request.Request(url, headers={"User-Agent": "EveryWay/0.1 (hackathon learning app; contact via texel genjam)", "accept": "application/json"})
    with urllib.request.urlopen(req, timeout=20, context=texel._ctx()) as r:
        return json.loads(r.read())


def web_resources(topic):
    """Real reference material from Wikipedia: the matching article's summary and its own pictures, with links back."""
    if texel.MOCK:
        return None
    from urllib.parse import quote
    try:
        # Never search with the learner's raw words (that is how "rain" found a film monologue). Claude names the article, Wikipedia confirms it exists.
        want = (claude("You pick encyclopedia articles. Reply with the exact title of the one English Wikipedia article that best explains the science or facts behind the topic. A general subject article, never a film, song, book, person or list. Title only, nothing else.",
                       "Topic a learner asked about: %s" % topic, 40) or "").strip().strip('"').splitlines()[0:1]
        want = want[0].strip() if want else ""
        if not want or len(want) > 80:
            return None
        hits = _wiki("https://en.wikipedia.org/w/api.php?action=query&list=search&format=json&srlimit=3&srsearch=" + quote(want))["query"]["search"]
        hit = next((h for h in hits if h["title"].lower() == want.lower()), None) or (hits[0] if hits else None)
        if not hit or re.search(r"\((film|song|album|novel|band|TV series|video game|disambiguation)\)|monologue|soundtrack", hit["title"], re.I):
            return None
        title = hit["title"]; t = quote(title.replace(" ", "_"), safe="")
        summ = _wiki("https://en.wikipedia.org/api/rest_v1/page/summary/" + t)
        if summ.get("type") == "disambiguation":
            return None
        out = {"title": summ.get("title", title), "extract": summ.get("extract", ""), "url": (summ.get("content_urls", {}).get("desktop", {}) or {}).get("page") or "https://en.wikipedia.org/wiki/" + t, "images": []}
        try:
            for it in _wiki("https://en.wikipedia.org/api/rest_v1/page/media-list/" + t).get("items", []):
                if it.get("type") != "image" or not it.get("showInGallery", True) or not it.get("srcset"):
                    continue
                src = it["srcset"][-1].get("src", "")
                name = it.get("title", "")
                if not src or re.search(r"(icon|logo|wiki|commons-logo|question_book|edit-clear|padlock)", name, re.I):
                    continue
                out["images"].append({"src": ("https:" + src) if src.startswith("//") else src, "caption": ((it.get("caption") or {}).get("text") or name.replace("File:", "").rsplit(".", 1)[0].replace("_", " "))[:180],
                                      "page": "https://commons.wikimedia.org/wiki/" + quote(name.replace(" ", "_"), safe=":")})
                if len(out["images"]) >= 4:
                    break
        except Exception as e:
            log("  wikipedia pictures failed: %r" % (e,))
        return out
    except Exception as e:
        log("  wikipedia lookup failed: %r" % (e,))
        return None


def rasterize_svg(svg, dest_png):
    """SVG -> 1920x1080 PNG using whatever this Mac has: headless Chrome, rsvg-convert, or Quick Look. Returns True on success."""
    dest_png = Path(dest_png); d = dest_png.parent
    page = d / (dest_png.stem + ".html")
    page.write_text('<!doctype html><html><body style="margin:0;background:#fff;width:1920px;height:1080px;display:flex;align-items:center;justify-content:center">'
                    '<div style="width:1840px">%s</div><style>svg{width:100%%;height:auto;display:block}</style></body></html>' % svg)
    chromes = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "/Applications/Chromium.app/Contents/MacOS/Chromium",
               "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge", "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
               shutil.which("google-chrome") or "", shutil.which("chromium") or "", shutil.which("chromium-browser") or ""]
    for c in chromes:
        if c and Path(c).exists():
            try:
                subprocess.run([c, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1", "--window-size=1920,1080",
                                "--user-data-dir=%s" % (d / ".chrome-shot"), "--screenshot=%s" % dest_png, page.resolve().as_uri()], capture_output=True, timeout=45)
                if dest_png.exists() and dest_png.stat().st_size > 3000:
                    return True
            except Exception as e:
                log("  chrome screenshot failed: %r" % (e,))
    svgf = d / (dest_png.stem + ".svg"); svgf.write_text(svg)
    rsvg = shutil.which("rsvg-convert")
    try:
        if rsvg:
            subprocess.run([rsvg, "-w", "1920", "-b", "white", "-o", str(dest_png), str(svgf)], capture_output=True, timeout=30)
        elif shutil.which("qlmanage"):
            subprocess.run(["qlmanage", "-t", "-s", "1920", "-o", str(d), str(svgf)], capture_output=True, timeout=30)
            q = d / (svgf.name + ".png")
            if q.exists():
                q.rename(dest_png)
    except Exception as e:
        log("  svg convert failed: %r" % (e,))
    return dest_png.exists() and dest_png.stat().st_size > 3000


def build_companion(key, lesson, L, ldir):
    """Runs beside the video: the written lesson first (seconds), then Claude's diagrams, then two Texel illustrations."""
    try:
        web = web_resources(lesson)
        if web:
            L["companion"] = {"web": web}; touch()
            note("%s: found the Wikipedia article '%s' with %d pictures" % (L["name"], web["title"], len(web["images"])))
        c = make_companion(lesson, L)
        if c:
            if web:
                c["web"] = web
            L["companion"] = c; touch()
            note("%s: written lesson ready" % L["name"])
        try:
            d, icons = make_diagrams(lesson, L, (c or {}).get("key_points"))
            for n, dg in enumerate(d):
                png = ldir / ("diagram_%d.png" % (n + 1))
                if rasterize_svg(dg["svg"], png):
                    dg["png"] = png.name
            comp = L.setdefault("companion", {})
            comp["diagrams"] = d; comp["icons"] = icons; touch()
            note("%s: %d diagrams drawn, %d ready for the video" % (L["name"], len(d), sum(1 for x in d if x.get("png"))))
        except Exception as e:
            log("  diagrams failed: %r" % (e,))
            L.setdefault("companion", {}).setdefault("diagrams", [])
        for n, ill in enumerate((c or {}).get("illustrations", [])):
            dest = ldir / ("illustration_%d.jpg" % (n + 1))
            sp, sn = styled(L, ill["image"])
            r = gen_image(key, sp, sn, L.get("style", "Realistic"), dest)
            if r.get("ok"):
                ill["path"] = rel(dest); ill["engine"] = r.get("engine", "Texel"); touch()
    except Exception as e:
        log("  companion failed: %r" % (e,))


def make_plans(lesson, learners, feedback=None, previous=None):
    """learners: [{"name", "profile", "answers"}]. Returns the plans shown in the preview (no generation yet)."""
    global _env
    _env = _env or load_env()
    plans = []
    for i, lr in enumerate(learners):
        L = {"id": slug(lr.get("name") or "you") + ("-%d" % (i + 1) if i else ""), "name": lr.get("name") or "You", "profile": describe(lr), "answers": lr.get("answers") or {}}
        prev = previous[i] if previous and i < len(previous) else None
        p = plan_learner(lesson, lr, SHOTS_PER_LESSON, feedback, prev)
        if not p:
            return None
        check_plan(lesson, lr, p)
        L.update({k: p.get(k) for k in ("style", "negative", "shots", "planned_by", "approach", "checked")}); L["title"] = p.get("title") or lesson
        plans.append(L)
    return plans


def static_plans():
    L = json.loads((HERE / "lesson.json").read_text())
    plans = []
    for l in L["learners"]:
        name = l["profile"].split(",")[0].strip()
        plans.append({"id": l["id"], "name": name, "profile": l["profile"].split(",", 1)[1].strip() if "," in l["profile"] else l["profile"],
                      "style": l["image_model"], "negative": l["negative"], "title": L["lesson"], "shots": [dict(s) for s in l["shots"]],
                      "planned_by": "saved plan"})
    return L["lesson"], plans


# ------------------------------------------------------------------ critic
CRITIC_SYSTEM = """You are the critic in a studio that makes short explainer videos with an open-source image model. You are shown one take for one shot. The narration and captions do the teaching; the picture only has to be a fitting, good-looking illustration for that line. Grade four checks, each true or false:
- "shows_idea": the picture shows the intended scene, or something that clearly fits the narration line. It does NOT need to show every element or prove the science. Fail only if the subject is plainly wrong or unrelated.
- "accurate": nothing in the picture is clearly false or misleading for this topic (for example two suns, a plant with animal parts). Artistic licence is fine.
- "clean": no broken anatomy, extra limbs, melted or duplicated subjects, garbled text, watermarks or heavy smearing.
- "suitable": right for this audience; no child or teenager depicted; nothing frightening or inappropriate.
When the style is a whiteboard doodle, a simple marker drawing on white is exactly what is wanted; fail "clean" if it came out as a photograph, is cluttered, or has scribbled fake writing in it.
Most reasonable takes should pass. Reject only real problems a viewer would notice.
The verdict is "reject" if any check is false, otherwise "accept".
Return ONLY JSON: {"checks": {"shows_idea": bool, "accurate": bool, "clean": bool, "suitable": bool}, "verdict": "accept" | "reject", "reason": "<at most 14 words, plain language, name the specific problem>", "revised_prompt": "<if rejecting: a simpler, more literal prompt in the same style that fixes it, else empty>"}"""

CHECKS = ["shows_idea", "accurate", "clean", "suitable"]


def small_jpeg(src, dest, width=768):
    subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-i", str(src), "-vf", "scale=%d:-2" % width, "-frames:v", "1", "-q:v", "5", str(dest)], check=False)
    return dest if Path(dest).exists() else None


def frames_of(clip, tmp, n=3):
    d = duration(clip) or CLIP_SECONDS
    out = []
    for i in range(n):
        f = Path(tmp) / ("%s_f%d.jpg" % (Path(clip).stem, i))
        subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-ss", "%.2f" % (d * (0.1 + 0.4 * i)), "-i", str(clip), "-vf", "scale=640:-2", "-frames:v", "1", "-q:v", "6", str(f)], check=False)
        if f.exists():
            out.append(f)
    return out


def critic(kind, images, shot, learner, prompt, take, tmp):
    """Returns dict(verdict, reason, revised_prompt, by)."""
    if texel.MOCK or not (load_env().get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")):
        # stand-in critic so the loop can be exercised offline: rejects the first take of a deterministic ~1 in 3 shots
        h = int(hashlib.md5((learner["id"] + shot["id"] + kind).encode()).hexdigest(), 16)
        if texel.MOCK and take == 1 and h % 3 == 0:
            bad = CHECKS[h % 4]
            return {"verdict": "reject", "reason": "stand-in critic: pretend failure, not a real judgment", "checks": {c: c != bad for c in CHECKS},
                    "revised_prompt": prompt + ", simple composition, single subject", "by": "stand-in"}
        return {"verdict": "accept", "reason": "stand-in critic" if texel.MOCK else "no critic available (no Anthropic key)", "checks": {c: True for c in CHECKS} if texel.MOCK else {},
                "revised_prompt": "", "by": "stand-in" if texel.MOCK else "none"}
    blocks = []
    for im in images:
        blocks.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(Path(im).read_bytes()).decode()}})
    blocks.append({"type": "text", "text": "Topic: %s\nTake %d, %s.\nAudience: %s, %s.\nThe idea this shot must make visible: %s\nNarration over it: %s\nPrompt used: %s\n%s" % (
        STATE.get("lesson", ""), take, "a still image" if kind == "still" else "%d frames sampled in order from a 5 second clip" % len(images), learner["name"], learner["profile"],
        shot.get("teaches") or shot["narration"], shot["narration"], prompt,
        "Style: %s. The intended picture: %s. If you reject, rewrite the IMAGE prompt as a simpler plain description." % ("whiteboard doodle" if learner.get("style") == "Sketch" else learner.get("style", ""), shot.get("teaches") or "a scene that fits the line") if kind == "still" else "For a clip, judge motion and stability as part of 'clean'. If you reject, rewrite the MOTION prompt to be simpler and more stable.")})
    v = parse_json(claude(CRITIC_SYSTEM, blocks, 600))
    if not v or v.get("verdict") not in ("accept", "reject"):
        return {"verdict": "accept", "reason": "critic gave no usable answer, kept the take", "revised_prompt": "", "by": "fallback"}
    v["by"] = "critic agent"
    v["reason"] = str(v.get("reason", ""))[:140]
    ch = v.get("checks") if isinstance(v.get("checks"), dict) else {}
    v["checks"] = {c: bool(ch.get(c, True)) for c in CHECKS}
    if not all(v["checks"].values()):
        v["verdict"] = "reject"
    return v


# ------------------------------------------------------------------ media helpers
def ffmpeg():
    return shutil.which("ffmpeg") or ("/opt/homebrew/bin/ffmpeg" if Path("/opt/homebrew/bin/ffmpeg").exists() else "/usr/local/bin/ffmpeg")


def ffprobe():
    return shutil.which("ffprobe") or str(Path(ffmpeg()).with_name("ffprobe"))


def duration(path):
    try:
        r = subprocess.run([ffprobe(), "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)], capture_output=True, text=True)
        return float(r.stdout.strip())
    except Exception:
        return None


_HAS_DRAWTEXT = None


def has_drawtext():
    global _HAS_DRAWTEXT
    if _HAS_DRAWTEXT is None:
        r = subprocess.run([ffmpeg(), "-hide_banner", "-filters"], capture_output=True, text=True)
        _HAS_DRAWTEXT = " drawtext " in r.stdout
    return _HAS_DRAWTEXT


def find_font(run_dir):
    dest = Path(run_dir) / "font.ttf"
    if dest.exists():
        return "font.ttf"
    for c in ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf",
              "/System/Library/Fonts/Supplemental/Verdana.ttf", "/usr/share/fonts/truetype/lato/Lato-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
        if Path(c).exists():
            shutil.copy(c, dest)
            return "font.ttf"
    return None


def wrap(text, width=44):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            lines.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return "\n".join(lines)


_VOICE = {}
VOICE_PREFS = ["Ava (Premium)", "Zoe (Premium)", "Evan (Premium)", "Nathan (Premium)", "Ava (Enhanced)", "Zoe (Enhanced)", "Samantha (Enhanced)", "Allison (Enhanced)", "Evan (Enhanced)",
               "Tom (Enhanced)", "Susan (Enhanced)", "Ava", "Zoe", "Allison", "Samantha"]


def best_voice():
    """The most natural English voice installed on this Mac. EVERYWAY_VOICE overrides."""
    if "v" in _VOICE:
        return _VOICE["v"]
    v = os.environ.get("EVERYWAY_VOICE")
    say = shutil.which("say")
    if not v and say:
        try:
            out = subprocess.run([say, "-v", "?"], capture_output=True, text=True, timeout=20).stdout
            have = [re.split(r"\s{2,}", ln.strip())[0] for ln in out.splitlines() if re.search(r"\ben[_-](US|GB|AU)\b", ln)]
            v = next((p for p in VOICE_PREFS if p in have), None)
        except Exception:
            v = None
    _VOICE["v"] = v or "Samantha"
    return _VOICE["v"]


def narrate(text, dest_wav, voice):
    dest_wav = Path(dest_wav)
    if dest_wav.exists():
        return duration(dest_wav), "cached"
    say = shutil.which("say")
    if say and not texel.MOCK:
        voice = best_voice()
        aiff = dest_wav.with_suffix(".aiff")
        r = subprocess.run([say, "-v", voice, "-r", os.environ.get("EVERYWAY_VOICE_RATE", "172"), "-o", str(aiff), text], capture_output=True)
        if r.returncode != 0:
            subprocess.run([say, "-o", str(aiff), text], capture_output=True)
        if aiff.exists():
            subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-i", str(aiff), "-af", "adelay=250|250,apad=pad_dur=0.25", "-ar", "48000", "-ac", "2", str(dest_wav)], check=False)
            aiff.unlink(missing_ok=True)
            if dest_wav.exists():
                return duration(dest_wav), "Mac voice %s" % voice
    secs = max(2.5, len(text.split()) / 2.6)
    subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", "%.2f" % secs, str(dest_wav)], check=True)
    return secs, "silent placeholder"


ENC = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(FPS), "-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "160k"]


def drawtext(textfile, size, y, font, box=True):
    if not font or not has_drawtext():
        return ""
    s = ",drawtext=fontfile=%s:textfile=%s:fontsize=%d:fontcolor=white:line_spacing=10:x=(w-text_w)/2:y=%s" % (font, textfile, size, y)
    if box:
        s += ":box=1:boxcolor=black@0.55:boxborderw=22"
    return s


def card(run_dir, name, lines, seconds, colour):
    """A plain colour card with centred text lines [(text, size)]."""
    font = find_font(run_dir)
    if not font or not has_drawtext():
        return None          # this ffmpeg cannot draw text: skip the card instead of showing a blank colour
    vf = "format=yuv420p"
    y = H / 2 - sum(sz for _, sz in lines) * 0.9
    for i, (txt, sz) in enumerate(lines):
        tf = "%s_t%d.txt" % (name, i)
        Path(run_dir, tf).write_text(txt)
        vf += drawtext(tf, sz, "%d" % y, font, box=False)
        y += sz * 1.8
    out = "%s.mp4" % name
    subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=%s:s=%dx%d:r=%d" % (colour, W, H, FPS),
                    "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", "%.2f" % seconds, "-vf", vf] + ENC + [out], check=True, cwd=str(run_dir))
    return out


def segment(run_dir, name, clip, still, wav, caption, reveal=None):
    """One shot: the clip slowed and held to cover the narration, caption burned in."""
    font = find_font(run_dir)
    T = (duration(Path(run_dir) / wav) or 4.0) + 0.6
    Path(run_dir, name + ".txt").write_text(wrap(caption))
    cap = drawtext(name + ".txt", 46, "h-text_h-96", font)
    base = "scale=%d:%d:force_original_aspect_ratio=increase:flags=lanczos,crop=%d:%d" % (W, H, W, H)
    out = name + ".mp4"
    if clip and Path(run_dir, clip).exists():
        clen = duration(Path(run_dir) / clip) or CLIP_SECONDS
        stretch = min(max(T / clen, 1.0), 1.5)
        vf = "%s,setpts=%.3f*PTS,fps=%d,tpad=stop_mode=clone:stop_duration=30%s" % (base, stretch, FPS, cap)
        cmd = [ffmpeg(), "-y", "-loglevel", "error", "-i", clip, "-i", wav, "-filter_complex", "[0:v]%s[v];[1:a]apad[a]" % vf, "-map", "[v]", "-map", "[a]", "-t", "%.2f" % T]
    else:
        frames = int(T * FPS) + 2
        k = sum(ord(c) for c in name) % 3
        z = ["min(1.0+0.0007*on,1.14)", "max(1.14-0.0007*on,1.0)", "min(1.04+0.0005*on,1.12)"][k]
        xy = ["x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'", "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'", "x='(iw-iw/zoom)*on/%d':y='ih/2-(ih/zoom/2)'" % frames][k]
        if reveal if reveal is not None else STYLE == "sketch":
            # whiteboard feel: the drawing is revealed left to right over the first 1.3 s, then a very slow push in
            zp = "scale=%d:%d:force_original_aspect_ratio=increase,crop=%d:%d,zoompan=z='min(1.0+0.00035*on,1.06)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=%d:s=%dx%d:fps=%d" % (W * 2, H * 2, W * 2, H * 2, frames, W, H, FPS)
            fc = "[0:v]%s[pic];color=c=white:s=%dx%d:r=%d[wh];[pic][wh]overlay=x='W*min(t/1.3\\,1)':y=0:shortest=0[rev];[rev]format=yuv420p%s[v];[1:a]apad[a]" % (zp, W, H, FPS, cap)
            cmd = [ffmpeg(), "-y", "-loglevel", "error", "-loop", "1", "-i", still, "-i", wav, "-filter_complex", fc, "-map", "[v]", "-map", "[a]", "-t", "%.2f" % T]
        else:
            vf = "scale=%d:%d:force_original_aspect_ratio=increase,crop=%d:%d,zoompan=z='%s':%s:d=%d:s=%dx%d:fps=%d%s" % (W * 2, H * 2, W * 2, H * 2, z, xy, frames, W, H, FPS, cap)
            cmd = [ffmpeg(), "-y", "-loglevel", "error", "-loop", "1", "-i", still, "-i", wav, "-filter_complex", "[0:v]%s[v];[1:a]apad[a]" % vf, "-map", "[v]", "-map", "[a]", "-t", "%.2f" % T]
    subprocess.run(cmd + ENC + [out], check=True, cwd=str(run_dir))
    return out


# ------------------------------------------------------------------ the loop
def make_shot(key, video_model, L, S, ldir):
    """Still -> critic -> (retake) -> clip -> critic -> (retake). Updates S in place for the UI."""
    tmp = ldir / "critic"; tmp.mkdir(parents=True, exist_ok=True)
    S["takes"] = []
    prompt = S["image"]
    still = None
    S["status"] = "drawing the still"; touch()
    for take in range(1, MAX_STILL_TAKES + 1):
        dest = ldir / ("%s_still_t%d.jpg" % (S["id"], take))
        cached = HERE / "out" / L["id"] / (S["id"] + ".jpg")
        if take == 1 and L.get("planned_by") == "saved plan" and cached.exists() and not dest.exists():
            shutil.copy(cached, dest)
        sp, sn = styled(L, prompt)
        r = gen_image(key, sp, sn, L["style"], dest)
        t = {"kind": "still", "n": take, "path": rel(dest) if r.get("ok") else None, "seconds": r.get("seconds"), "prompt": prompt, "engine": r.get("engine", "Texel")}
        if not r.get("ok"):
            t.update(verdict="failed", reason="Texel did not return an image"); S["takes"].append(t); touch(); continue
        S["status"] = "critic is checking the still"; touch()
        small = small_jpeg(dest, tmp / (dest.stem + "_s.jpg")) or dest
        v = critic("still", [small], S, L, prompt, take, tmp)
        t.update(verdict=v["verdict"], reason=v["reason"], by=v["by"], checks=v.get("checks") or {}); S["takes"].append(t); touch()
        note("%s / %s still take %d: %s%s" % (L["name"], S["id"], take, v["verdict"], "" if v["verdict"] == "accept" else " (" + v["reason"] + ")"))
        still = dest
        if v["verdict"] == "accept":
            break
        if take < MAX_STILL_TAKES and v.get("revised_prompt"):
            prompt = v["revised_prompt"]
    if not still:
        S["status"] = "failed"; touch(); return
    if S["takes"][-1].get("verdict") == "reject":
        S["takes"][-1]["kept_anyway"] = True      # out of takes: use the last one and say so
    S["still"] = rel(still)

    motion = S["motion"]
    clip = None
    no_video = video_model == "none" and not (texel.BACKUP_ON and texel.backup_key()) and not texel.MOCK
    for take in range(1, (0 if no_video else MAX_CLIP_TAKES) + 1):
        S["status"] = "animating on Texel (take %d)" % take; touch()
        dest = ldir / ("%s_clip_t%d.mp4" % (S["id"], take))
        cached = HERE / "out" / L["id"] / (S["id"] + ".mp4")
        if take == 1 and L.get("planned_by") == "saved plan" and cached.exists() and not dest.exists() and len(S["takes"]) == 1:
            shutil.copy(cached, dest)             # reuse the overnight clip only when its still was accepted first time
        r = gen_video(key, video_model, motion, L["negative"], still, dest, CLIP_SECONDS, 832, 480, "%s/%s" % (L["id"], S["id"]))
        t = {"kind": "clip", "n": take, "path": rel(dest) if r.get("ok") else None, "seconds": r.get("seconds"), "prompt": motion, "model": r.get("model") or video_model, "engine": r.get("engine", "Texel")}
        if not r.get("ok"):
            t.update(verdict="failed", reason=str(r.get("error"))[:120]); S["takes"].append(t); touch()
            note("%s / %s clip take %d failed on Texel" % (L["name"], S["id"], take)); continue
        S["status"] = "critic is checking the clip"; touch()
        v = critic("clip", frames_of(dest, tmp), S, L, motion, take, tmp)
        t.update(verdict=v["verdict"], reason=v["reason"], by=v["by"], checks=v.get("checks") or {}); S["takes"].append(t); touch()
        note("%s / %s clip take %d: %s%s" % (L["name"], S["id"], take, v["verdict"], "" if v["verdict"] == "accept" else " (" + v["reason"] + ")"))
        clip = dest
        if v["verdict"] == "accept":
            break
        if take < MAX_CLIP_TAKES and v.get("revised_prompt"):
            motion = v["revised_prompt"]
    if clip:
        if S["takes"][-1].get("verdict") == "reject":
            S["takes"][-1]["kept_anyway"] = True
        S["clip"] = rel(clip)
    S["status"] = "ready" if clip else "ready (still only)"; touch()


def assemble(lesson, L, ldir, idx):
    L["status"] = "narrating and editing"; touch()
    voice = L.get("voice") or VOICES[idx % len(VOICES)]
    segs = [card(ldir, "card_title", [(lesson, 96), ("made for %s" % L["name"].lower() if L["name"] == "You" else "made for %s" % L["name"], 56)], 2.0, "0x14231F")]
    t0 = time.time()                       # give Claude's diagrams a moment to arrive so they can be scenes too
    while time.time() - t0 < 80 and "diagrams" not in (L.get("companion") or {}) and not texel.MOCK:
        time.sleep(2)
    dgs = [x for x in ((L.get("companion") or {}).get("diagrams") or []) if x.get("png") and (ldir / x["png"]).exists()]

    def diagram_seg(n):
        x = dgs[n]; wav = "diagram_%d_narration.wav" % (n + 1)
        narrate(x["caption"], ldir / wav, voice)
        return segment(ldir, "seg_diagram_%d" % (n + 1), None, x["png"], wav, x["caption"], reveal=True)

    shots_ok = [S for S in L["shots"] if S.get("still")]
    for i, S in enumerate(shots_ok):
        wav = "%s_narration.wav" % S["id"]
        d, how = narrate(S["narration"], ldir / wav, voice)
        L["narration"] = how
        clip = Path(OUT, S["clip"]).name if S.get("clip") else None
        segs.append(segment(ldir, "seg_" + S["id"], clip, Path(OUT, S["still"]).name, wav, S["narration"]))
        try:
            if i == len(shots_ok) // 2 - 1 and len(dgs) > 1:
                segs.append(diagram_seg(1))          # the zoomed-in diagram lands mid-way
        except Exception as e:
            log("  diagram scene failed: %r" % (e,))
    try:
        if dgs:
            segs.append(diagram_seg(0))              # the whole idea at a glance closes the video
            L["diagram_scenes"] = min(len(dgs), 2)
    except Exception as e:
        log("  diagram scene failed: %r" % (e,))
    takes_all = [t for S in L["shots"] for t in S.get("takes", []) if t.get("path")]
    on_backup = sum(1 for t in takes_all if str(t.get("engine", "")).startswith("backup"))
    credit = "Every frame generated on Texel" if not on_backup else ("Generated on Texel, with %d of %d takes on a backup engine" % (on_backup, len(takes_all)) if on_backup < len(takes_all) else "Generated on a backup engine")
    L["credit"] = credit
    segs.append(card(ldir, "card_end", [(credit, 60 if not on_backup else 44), ("%d takes made, %d rejected by the critic" % (
        sum(len(S.get("takes", [])) for S in L["shots"]), sum(1 for S in L["shots"] for t in S.get("takes", []) if t.get("verdict") == "reject")), 40)], 2.6, "0x14231F"))
    segs = [x for x in segs if x]
    L["captions"] = bool(find_font(ldir) and has_drawtext())
    (ldir / "concat.txt").write_text("".join("file '%s'\n" % s for s in segs))
    subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", "concat.txt", "-c", "copy", "-movflags", "+faststart", "final.mp4"], check=True, cwd=str(ldir))
    L["final"] = rel(ldir / "final.mp4"); L["final_seconds"] = round(duration(ldir / "final.mp4") or 0, 1)
    L["status"] = "done"; touch()
    note("%s: finished video, %s s" % (L["name"], L["final_seconds"]))


COLOURS = ["0x2456B8", "0xB8356A", "0x1F7A5C", "0x8A5A12"]


def run(lesson, learners, use_static=False, run_name=None, plans=None):
    """learners: [{"name":..., "profile":...}]. Blocks until every learner's video is done."""
    global _env
    _env = load_env()
    key = _env.get("TEXEL_API_KEY", "")
    if not key and not texel.MOCK and not (texel.BACKUP_ON and texel.backup_key()):
        raise SystemExit("TEXEL_API_KEY is missing in .env (or set EVERYWAY_MOCK=1 to try the app with stand-ins).")
    for k in texel.TALLY:
        texel.TALLY[k] = 0
    video_model = os.environ.get("EVERYWAY_VIDEO_MODEL")
    if not video_model:
        video_model = "none"     # every Texel video job failed on Oct 10; set EVERYWAY_VIDEO_MODEL=framepack|ltxv|wan to try again
        try:
            sm = json.loads((HERE / "out" / "report.json").read_text())["smoke"]
            if sm.get("video_framepack", {}).get("ok"):
                video_model = "framepack"
            elif sm.get("video_ltxv", {}).get("ok"):
                video_model = "ltxv"
        except Exception:
            pass
    if use_static:
        lesson, plans = static_plans()
    run_dir = Path(OUT) / "runs" / (run_name or "%s-%s" % (datetime.now().strftime("%m%d-%H%M%S"), slug(lesson)))
    run_dir.mkdir(parents=True, exist_ok=True)
    with _slock:
        STATE.clear()
        STATE.update({"status": "planning", "lesson": lesson, "run_dir": str(run_dir), "started": time.time(), "log": [],
                      "engine": "local stand-in, not Texel" if texel.MOCK else "Texel API", "video_model": video_model, "learners": []})
    note("Lesson: %s" % lesson)
    if not use_static and not plans:
        STATE["status"] = "planning"; touch()
        plans = make_plans(lesson, learners)
        if not plans:
            raise SystemExit("The planner agent returned nothing (no Anthropic key, or the call failed). Use --static for the saved lesson.")
    STATE["learners"] = plans
    for L in plans:
        note("%s: plan ready, %d shots, %s style" % (L["name"], len(L["shots"]), L["style"]))
    for i, L in enumerate(plans):
        L["colour"] = COLOURS[i % len(COLOURS)]; L["status"] = "generating"
        for S in L["shots"]:
            S["status"] = "waiting"; S["takes"] = []
    STATE["status"] = "generating"; touch()

    def do_learner(args):
        i, L = args
        ldir = run_dir / L["id"]; ldir.mkdir(parents=True, exist_ok=True)
        try:
            side = threading.Thread(target=build_companion, args=(key, lesson, L, ldir), daemon=True); side.start()
            with ThreadPoolExecutor(max_workers=2) as ex:
                list(ex.map(lambda S: make_shot(key, video_model, L, S, ldir), L["shots"]))
            assemble(lesson, L, ldir, i)
            side.join(timeout=240)
        except Exception as e:
            import traceback
            L["status"] = "failed: %r" % (e,); note("%s failed: %s" % (L["name"], traceback.format_exc()[-400:]))

    with ThreadPoolExecutor(max_workers=max(1, len(plans))) as ex:
        list(ex.map(do_learner, enumerate(plans)))
    STATE["status"] = "done"; STATE["finished"] = time.time(); touch()
    if texel.TALLY.get("backup"):
        STATE["engine"] = "Texel API, with %d takes on the backup engine" % texel.TALLY["backup"]
    note("All done. %d images, %d clips on %s." % (texel.TALLY["images"], texel.TALLY["clips"], STATE["engine"]))
    return run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--static", action="store_true", help="use the saved water cycle lesson (no planner call)")
    ap.add_argument("--lesson")
    ap.add_argument("--learner", action="append", default=[], help='"Name: profile", repeat for each learner')
    ap.add_argument("--name")
    a = ap.parse_args()
    learners = []
    for l in a.learner:
        n, _, p = l.partition(":")
        learners.append({"name": n.strip(), "profile": p.strip()})
    if not a.static and not (a.lesson and learners):
        ap.error("give --static, or --lesson and at least one --learner")
    d = run(a.lesson, learners, use_static=a.static, run_name=a.name)
    print("Videos in", d)


if __name__ == "__main__":
    main()
