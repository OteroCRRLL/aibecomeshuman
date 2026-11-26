# scenarios/

Scripted scenes for testing the agent **without the game**. Every scene here is
**invented for this project**. They are not scenes from the game, so they can be shared
freely and the agent cannot "recognize" them.

They mimic what perception will produce from the real game:
- `observation` steps are written in **English**, as the observer model will write them.
- `dialogue` lines and choice `label`s are in **Spanish**, as the game's subtitles and
  on-screen options will appear.

## Format

```toml
id = "unique-id"
title = "Readable title"
character = "markus"          # must be one of the run's characters
location = "Where it happens"
synthetic = true              # invented scene (always true here)

[[steps]]
kind = "observation"
text = "What the character sees."
significant = true            # optional: update the stance after this step

[[steps]]
kind = "dialogue"
speaker = "Woman"
text = "¿Puedes ayudarme?"

[[steps]]
kind = "choice"
id = "help"
timed = true                  # timed choices go to the fast decider
seconds = 6
  [[steps.options]]
  id = "A"
  label = "Ayudarla"
  outcome = "What happens if this option is chosen."
  [[steps.options]]
  id = "B"
  label = "Seguir caminando"
  outcome = "..."
  default = true              # optional: what happens if nobody answers in time
```

Play one with:

```powershell
uv run aibh play scenarios/tram-stop.toml -p offline     # fake model, free
uv run aibh play scenarios/tram-stop.toml                # Claude Haiku, cents
uv run aibh play scenarios/tram-stop.toml -d bad_ideas   # another directive
```
