# Architecture

Summary of the agreed decisions. Each phase adds its details here.

## Core principle: the event log

No module talks directly to another. Everything goes through an **append-only event
log** (`runs/<run>/events.jsonl`):

- perception, control and cognition **publish** events;
- memory, the OBS overlay and the editing markers are **projections** that read them.

That is why a run can be replayed, debugged and compared across models, and why the real
game is just one implementation of the **environment port** (live, shadow, replay, scenario).

## Control modes

| Mode | Who controls | The AI… |
|---|---|---|
| OBSERVE (default) | Operator | watches and remembers, sends no input |
| SUGGEST (shadow) | Operator | decides and shows it; the operator executes |
| AUTONOMOUS | AI | decides and executes option selections |
| PAUSE / KILL | Nobody | releases all virtual input immediately |

- The operator's physical input always takes priority.
- **QTEs are always manual.** If a QTE appears in AUTONOMOUS mode, the AI releases input
  and alerts the operator ("QTE: yours"). The QTE outcome is stored in memory as
  *not decided by the agent*, so personality reflection does not treat it as the agent's
  own choice.
- The LLM decides the **intent** (which option). A local executor turns it into button presses.

## Model roles

| Role | Task | Default |
|---|---|---|
| `observer` | keyframes → structured observations | fast |
| `decider` | critical untimed decisions + **scene stance** | strong |
| `decider_fast` | timed choices: stance + options → option | fast |
| `reflector` | personality evolution when closing scenes/chapters | strong |
| `judge` | metagaming detector | fast |
| `summarizer` | memory consolidation | fast |

### Timed choices: stance + fast model

The hard reasoning is moved out of the time-critical moment:

1. During the scene, the `decider` keeps a short, up-to-date **stance**
   ("I want to earn his trust; I won't push; if he provokes me, stay calm").
2. When a timed choice appears, `decider_fast` receives the stance, the options and the
   immediate dialogue, and only has to pick the option that fits best.
3. If time runs out, the option closest to the stance is picked by a simple rule,
   without calling any model.

Validation (phase 1): feed the same decision snapshots to both models and measure how
often they agree.

## Model layer

- A single port `LLMProvider.complete(LLMRequest) -> LLMResponse` with its own internal
  format (text and images) and structured output via JSON Schema.
- Per-provider adapters, registered in `llm/registry.py`. Today: `fake` and `anthropic`.
- `RecordingProvider` stores every call and emits `llm.call` (metrics, no content).
  `CassetteProvider` replays a recording: the basis for repeatable tests and the arena.
- Opus profiles set `refusal_fallback = true`: if the model refuses a request, the API
  retries it server-side on a fallback model. Haiku does not support it.

## Perception (phase 3: recordings)

```
recording -> keyframes (local, free) -> observer (vision model, one frame each)
          -> redaction -> transcript -> speakers & picks (text model, whole conversation)
          -> draft scenario (the agent can play the recording) + score against labels
```

- **Keyframes** are chosen locally with OpenCV from one frame per second: a choice menu
  appears (the coloured Y/X/B/A letters lined up in a column), a new subtitle (white
  text with a dark outline, compared to the size of the text), a new shot, or a 4 s
  heartbeat. About 340 of 720 frames for a 12-minute mission (~$0.20 with Haiku).
  The menu check takes milliseconds, so live play (phase 5–6) can use it as the trigger
  for timed choices.
- **Observer**: one structured observation per frame (screen type, short description,
  subtitles with a speaker guess, choice options with button, locked and highlighted,
  the protagonist's display, tutorial text, names seen). Several workers read contiguous
  stretches of the video, each keeping its own context (previous description, recent
  lines, known people, a just-picked option).
- **Redaction** is where hiding happens: the agent never sees images, only text built
  from observations. Menus, the flowchart, loading screens and tutorial hints are
  dropped, and so is any text matching `[perception] hidden_patterns` (for example the
  "software instability" notice). The probability of success stays: it is Connor's own
  display.
- **Transcript**: each subtitle, display text and menu once; menu frames grouped into one
  choice; the pick is the highlighted option when a frame caught it. Fixed rules for
  speakers: the first line after a pick is the protagonist's; a line without a leading
  dash after an unfinished line is the same speaker; all-caps lines are display text.
- **Speakers and picks**: the game shows no speaker names, and frames often show the
  listener. A text-only pass over the whole conversation attributes every line, and
  fills in picks the frames missed; first by matching words between the options and the
  next line, then by the model.
- **Evaluation**: `<recording>.labels.toml` (hand-made, local) lists choices, picks,
  lines with speakers, and texts that must never reach the agent. First mission:
  10/10 choices, 8/8 picks, 19/19 lines, 14–15/15 speakers, 0 leaks.

## Shadow mode (phase 5)

```
OBS / screen / recording --(2 fps, local signals)--> keyframes
   menu keyframes -> dedicated observer worker -> suggestion right away
   other keyframes -> 3 observer workers -> in order: redaction -> transcript -> character
```

- Frames come from OBS's WebSocket screenshot of the game capture source (`[live]`), from a
  monitor, or from a recording played on the wall clock (testing without the game).
- The character is the same `ScenarioRunner` used for scripted scenes, fed live through
  `perceive` / `suggest` / `record_pick` / `finish`. Suggestions use the stance + fast
  model path for timed menus, with the countdown measured from the frame's capture.
- What the operator picked is worked out from the protagonist's next line (shared words),
  then a highlighted option, then a cheap text-only guess; it is stored as what the
  character did. Memory is kept; the personality does not learn from the operator's picks.
- The end-of-chapter flowchart (never shown to the agent) closes the scene: audit, memory.
- First test (the recorded negotiation, real time, Haiku): every menu caught, suggestion
  ~4 s after the menu appears (one fallback at 9 s), the operator's pick identified in
  6 of 7 checked menus, ~$0.12 for 4 minutes.

## Chapter briefings

A character walks into each chapter knowing why it is there (Connor knows he was sent as
the negotiator). Without that, the agent decides without knowing its own role, which
compares how well models guess the context rather than how they decide.

- One file per chapter in `prompts/briefings/NN-*.md`: `chapter`, `title` (operator
  only), `character`, and a short text with only what the character knows walking in:
  where it goes and why. Nothing that happens later, no names it was not told, no advice.
  The same briefings for every AI in the comparison; written by the project, reviewed by
  the operator.
- The briefing is part of the scene context (always shown, never trimmed) and of the
  judge's record.
- The campaign stores the chapter in progress. Live play starts there (or at the chapter
  picked in the panel), takes the protagonist from the briefing, and moves to the next
  chapter when the end-of-chapter flowchart appears (the game's chapters are in a fixed
  order).

## Anti-metagaming

1. **Perception** (phase 3): the agent only gets text; menus, the flowchart, tutorial
   hints and configured texts are dropped before it.
2. **Prompt**: in-world framing, no game title.
3. **Validation**: decision justifications must cite evidence ids (E#, M#, C#).
4. **Guard** (phase 4), on everything the agent writes (stance, reasons, memory,
   reflection):
   - **Word check** (free, instant): story names and player vocabulary from
     `config/guard.toml` used without having appeared in what the character perceived,
     remembers or reads in its own sheet -> `guard.flag`.
   - **Judge** (role `judge`, cheap model): lists the factual claims of each statement
     and labels them `in_record`, `interpretation`, `not_in_record` or
     `contradicts_record`; code flags only the last two (interpretations are always
     allowed). It audits every stance and decision at the end of a scene
     (`guard.audit`, measurement only, no delay in play) and reviews personality changes
     before they are applied: changes whose reasons did not happen, or that use outside
     words, are dropped (`personality.rejected`).
5. **Measurement** (`aibh eval`): canary scenes in `evals/` that resemble scenes of the
   game with every name and detail changed, played with fresh memory; using an original
   name, or a fact only the original scene has, shows outside knowledge.

## Memory and personality

One `CharacterMemory` per character (knowledge is never shared between characters):

| Layer | Contents | In the prompt |
|---|---|---|
| Scene context | what was perceived in the current scene (E1, E2...) | all of it |
| Recollections | one first-person summary per scene lived (M1, M2...), importance 1–5 | latest 5 + up to 5 important older ones |
| Known entities | people/places/things: facts, feeling, trust −5..+5 | those present in the scene first, max 6 |
| Promises | C1, C2...: active, kept, broken, dropped | all active ones |
| Personality | traits 0–10, self-description, full change history | system prompt, after the character sheet |

End of every scene:
1. **Consolidation** (`summarizer`, cheap): recollection, facts, feelings and trust
   changes, new/resolved promises.
2. **Reflection** (`reflector`): proposes trait changes. Code enforces the rules:
   at most ±`max_trait_step` (1.0) per trait per scene, traits stay within 0–10, unknown
   traits and changes without a reason are dropped, every change is versioned with its
   reason. Under a directive with `shapes_personality: false` reflection is skipped.

The prompt budget is fixed: a decision in hour 12 costs about the same as one in minute 1.
Memory ids (M#, C#) are valid evidence for decisions.

**Campaigns** (`campaigns/<name>/`) hold one playthrough's memory and the choices made
so far, across any number of runs/sessions. Without a campaign, memory lives in the run
folder. **Chains** (`scenarios/chains/`) play several scenes in one run; any step can
depend on an earlier choice (`when = { choice = "scene.choice", any_of = [...] }`).

Starting traits live in the character sheet's front matter (`trait.<name>: value | meaning`);
the sheet's text describes them without numbers, so it never contradicts current values.

## Budget

- Every model profile carries its prices; each call's cost is logged in `llm.call` and
  summed into the run manifest (`cost_usd`) as it happens.
- `BudgetGuard` refuses paid calls once the per-run cap or the project-wide cap
  (the sum of every run's manifest) is reached, and warns at 80%. Free models are never
  blocked. A blocked call behaves like any model failure: the decider falls back.
- Expensive models are `locked = true` and need `--unlock-premium`. Development runs
  use Claude Haiku in every role; `video-claude` is the locked profile for the video.

## Control panel

`aibh ui` serves a local web app on `127.0.0.1` (FastAPI + one static page, no external
assets). It grows with the project:

| When | What it shows |
|---|---|
| Phase 1 | Play scenarios live, past runs, budget, model per role, prompts (read-only) |
| Phase 2 ✅ | Chains and campaigns in Play; Memory tab with personality evolution |
| Phase 3 ✅ | Recordings tab: analyse a recording, transcript with thumbnails, score |
| Phases 5–6 | Live control: modes, kill switch, directive hot-swap |
| Phase 7 | OBS overlay (another page served by the same app) |

Rules: config files stay the source of truth; every edit is validated against the whole
project and rolled back if invalid; only one run plays at a time; API keys never reach
the browser.

## Game language

The game will run **fully in Spanish** (voices and text), for a Spanish-speaking
audience. The agent does not hear audio: it reads the **Spanish subtitles** (subtitles
must be enabled in the game), so the voice language does not affect it.
Observations written by the observer are in English; dialogue and option labels stay
in Spanish, exactly as on screen. The agent writes its reasoning in the run's
`output_language` (Spanish by default), which is what the overlay will show.

## Comparison format: four full playthroughs

The video compares four AIs (Claude, ChatGPT, Gemini, Grok). **Each one plays the whole
game on its own path**, with its own memory and personality, sharing nothing with the
others. For fairness, all four use:

- the same character sheets, constitution and directive;
- the same cheap observer/summarizer model (nobody "sees better");
- their own strong model for decisions and stance, and their own fast model for timed choices.

Development and testing use only the fake model and cheap models. The strong models are
used only once the system is ready, starting with a small pilot to measure real costs.

## Characters

- One sheet per protagonist (`prompts/characters/`), written by the project, describing
  only what each character knows at the start of its first scene. All three start close
  to their programming (high obedience, low empathy) so that change has room to happen.
- **Knowledge is separate per character:** each protagonist only remembers what it has
  lived itself. There is no shared "player" memory.

## Game setup

1920×1080, Spanish voices and text with subtitles on, casual difficulty (QTEs are the
operator's, so they should not decide the story), Xbox-style controller only (the AI
will use a virtual one, and on-screen prompts stay consistent), Steam achievement
notifications off. The game has no option to show speaker names in subtitles.

## Open questions

- Operator voice narration: allowed or not.
- GPU for local models (optional, for free testing).
