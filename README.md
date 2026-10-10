# Every Way

**Anyone can learn anything, explained their way.**

[![Watch the Every Way demo](https://img.youtube.com/vi/Ik9C9eY9Sws/hqdefault.jpg)](https://youtu.be/Ik9C9eY9Sws)

**[Watch the 2 minute demo on YouTube](https://youtu.be/Ik9C9eY9Sws)**

Type or speak a topic, say who it is for, and Every Way builds a personal lesson: a written explanation, labelled diagrams, real references, a quiz, and a narrated video whose every scene is generated with the [Texel API](https://api.prod.texel.ai/docs).

Built solo in 48 hours for the AAIFF Global GenJam (Austin AI Film Festival, October 2026), engineering track.

## What it does

1. **Ask.** Enter a topic ("Why is the sky blue") and who it is for ("Maya, 8, loves dinosaurs"). You can also speak the topic.
2. **Two quick questions.** The app asks about level and angle, inline, so the lesson is pitched at the right person.
3. **Lesson appears right away.** Overview, diagrams, key takeaways, key terms, a quiz and read-more links show up while the video is still being made.
4. **Video arrives when ready.** A short narrated explainer plays at the top of the same page.

The same topic produces a different lesson for a different learner. An eight year old who loves dinosaurs and a college student get different words, different examples and different pictures.

## How it works

| Step | What happens | Powered by |
|---|---|---|
| Intake | Turns the topic and learner into two clarifying questions | Claude |
| Planner | Writes a shot list: one literal image and one short narration line per scene | Claude |
| Checker | Reviews the plan for accuracy and fit before any credits are spent | Claude |
| Scene images | Generates every scene image | **Texel** `txt2img` (SDXL) |
| Critic | Looks at each image and rejects any that does not teach the idea, then Texel regenerates | Claude vision + Texel |
| Written lesson | Overview, takeaways, terms, quiz | Claude |
| Diagrams | Labelled SVG diagrams and icons, also inserted into the video as scenes | Claude |
| References | Finds the right article and pulls a summary and links | Wikipedia REST API |
| Narration | Text to speech | macOS `say` |
| Assembly | Camera moves on stills, diagram scenes, audio mix | ffmpeg |

Design choices worth knowing:

- **Never more than 2 Texel requests in flight**, as Texel asked of jam participants.
- **Silent fallback.** If a Texel call fails and a fal.ai key is present, the app retries on a backup model without bothering the user. With no backup key, it skips that take and carries on.
- **Texel image to video is built in and optional** (`EVERYWAY_VIDEO_MODEL=framepack`, `ltxv` or `wan`). The default video uses Texel stills with camera moves, which keeps a lesson fast and light on credits.
- **Standard library only.** No pip install. The server, the Texel client and the Claude client are plain Python.

## Run it

Requirements: macOS, Python 3.9 or newer, `ffmpeg` on your PATH, and Google Chrome (optional, used to turn diagrams into video scenes).

```bash
git clone <this repo>
cd everyway
cp .env.example .env        # then add your own keys to .env
python3 everyway/server.py
```

Open http://localhost:8787

- `http://localhost:8787` is the start page.
- `http://localhost:8787/#latest` opens the most recent finished lesson.
- `http://localhost:8787/#run=<folder name>` opens a specific finished lesson from `everyway/out/runs/`.

### Try it without keys or credits

```bash
EVERYWAY_MOCK=1 python3 everyway/server.py
```

Mock mode draws placeholder media with ffmpeg so you can click through the whole flow offline.

### Keys

| Variable | Required | Used for |
|---|---|---|
| `ANTHROPIC_API_KEY` | yes | Planner, checker, critic, written lesson, diagrams |
| `TEXEL_API_KEY` | yes | All scene images (and video clips if enabled) |
| `FAL_KEY` | no | Backup provider when a Texel call fails |

Keys live only in your local `.env`, which is git ignored. The app never prints or logs key values.

### Settings (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `EVERYWAY_PORT` | `8787` | Local port |
| `EVERYWAY_SHOTS` | `6` | Scenes per lesson |
| `EVERYWAY_STILL_TAKES` | `2` | Max image attempts per scene before moving on |
| `EVERYWAY_VIDEO_MODEL` | stills | `framepack`, `ltxv` or `wan` to animate scenes with Texel |
| `EVERYWAY_VOICE` | best installed | macOS voice name, for example `Ava (Premium)` |
| `EVERYWAY_MODEL` | see `pipeline.py` | Comma separated list of Claude models, tried in order |
| `EVERYWAY_FALLBACK` | `1` | Set to `0` to disable the backup provider |

## Files

```
everyway/
  server.py     Local web server and JSON API
  pipeline.py   The agents (intake, planner, checker, critic), lesson writer, diagrams, narration, assembly
  texel.py      Texel API client with the 2 request cap, mock mode and optional backup provider
  ui.html       The whole front end in one file
  night.py      Smoke test for a Texel account: python3 everyway/night.py --smoke
  lesson.json   A hand written sample lesson plan (two learners, one topic)
```

## Known limits

- macOS only for now, because narration uses the built in `say` command.
- Captions need an ffmpeg build with `drawtext`. Without it the video is made with no captions.
- Texel SDXL makes good photographic and anime scenes but cannot draw labelled diagrams, which is why diagrams come from Claude as SVG.

## Credits

Scene images by Texel. Writing, checking and diagrams by Claude (Anthropic). References from Wikipedia. Built by Vaishnavi Yeruva.

## License

MIT
