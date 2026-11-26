# aibecomeshuman

An AI agent that plays *Detroit: Become Human* on PC, documented for YouTube.

- **Hybrid control**: you explore while the AI watches and remembers. The AI takes over
  in dialogue, decisions and deductions. QTEs are always manual.
- **Evolving memory and personality** shaped by what the character lives through, kept
  coherent without relying on a giant history.
- **Swappable run directives**, with no code changes.
- **No metagaming**: the agent decides only from what it has seen, its personality and
  the directive.
- **Any model**: one provider per role, configurable, with model-vs-model comparison.
- **Testing without the game**: simulator, recordings and a fake model.

Architecture and decisions: [docs/architecture.md](docs/architecture.md).

## Status

**Phase 0, foundations.** Configuration, event log, model layer (fake + Anthropic),
call recording and replay, run manifest, CLI, tests and CI.

| Phase | Scope | Status |
|---|---|---|
| 0 | Foundations | ✅ (real-model ping pending) |
| 1 | Scenario simulator + decider + directives | ⏳ next |
| 2 | Memory and personality | |
| 3 | Offline perception (recordings) | |
| 4 | Anti-metagaming guard and evaluations | |
| 5 | Shadow-mode control (the AI suggests, you execute) | |
| 6 | Live autonomy | |
| 7 | Overlay and OBS | |
| 8 | Multi-model arena | |

## Getting started

Requirements: Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

```powershell
uv sync                     # create .venv and install dependencies
uv run aibh check           # validate the configuration
uv run aibh run -p offline  # empty run with the fake model
uv run aibh ping -p offline # one test call (fake model)
uv run pytest               # tests (no game, no APIs)
```

With a real model (needs `ANTHROPIC_API_KEY` or `ant auth login`):

```powershell
uv run aibh ping -p default -r decider        # Opus
uv run aibh ping -p default -r decider_fast   # Haiku
uv run pytest -m live                         # contract tests against the API
```

Each run creates a folder in `runs/` with `manifest.json` (models, prompts, cost),
`events.jsonl` (everything that happened) and `llm_calls.jsonl` (every call, replayable).

```powershell
uv run aibh events runs/<folder>
uv run aibh events runs/<folder> -t llm.call
```

## Layout

```
config/            settings, model profiles and run profiles (TOML)
prompts/           constitution, characters, directives (Markdown)
src/aibecomeshuman/
  core/            event log and clock
  llm/             model port, adapters, recording/cassettes
  run/             run session and manifest
  cli/             the aibh command
tests/             unit/ · integration/ · contract/ (live)
docs/              architecture and dev log
data/              local captures (not committed)
runs/              run outputs (not committed)
```

## Changing the model or the directive

- **Model**: edit `config/runs/<run>.toml` and assign another profile to a role, or
  create a new profile in `config/models/`.
- **Directive**: create `prompts/directives/<name>.md` and set `directive = "<name>"` in the run.

## License

[MIT](LICENSE): the code in this repository can be freely used, copied and modified,
as long as the copyright notice is kept.

## Disclaimer

Fan project, not affiliated with the rights holders of *Detroit: Become Human*. The
repository contains no game assets: game captures, recordings and text stay local
(`data/`, ignored by git).
