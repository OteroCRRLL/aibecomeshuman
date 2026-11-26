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

## Anti-metagaming (summary)

1. **Perception**: hide the end-of-chapter flowchart, menus, trophies and statistics.
2. **Prompt**: in-world framing, no game title. Experimental anonymized mode.
3. **Validation**: justifications must reference evidence; known-entities list; player
   vocabulary filter; optional judge.
4. **Measurement**: canary scenarios and synthetic control scenarios.

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
| Phase 3 | What the AI "saw" in recordings |
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

## Open questions

- Game resolution.
- Controller or keyboard for manual control (the AI will use a virtual controller).
- Operator voice narration: allowed or not.
- GPU for local models (optional, for free testing).
