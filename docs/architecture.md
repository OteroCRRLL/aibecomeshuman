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

- Game language and resolution.
- Controller or keyboard for manual control (the AI will use a virtual controller).
- Operator voice narration: allowed or not.
- GPU for local models (optional, for free testing).
