# aibecomeshuman

Un agente de IA que juega a *Detroit: Become Human* en PC. Proyecto documentado para YouTube.

- **Control híbrido**: tú exploras, la IA observa y memoriza. La IA toma el mando en
  diálogos, decisiones y deducciones. Los QTE son siempre manuales.
- **Memoria y personalidad que evolucionan** con lo que vive el personaje, con coherencia
  y sin depender de un historial gigante.
- **Directivas de run** intercambiables sin tocar código.
- **Sin metagaming**: el agente decide solo con lo que ha visto, su personalidad y la directiva.
- **Cualquier modelo**: proveedor por rol, configurable, y comparación entre modelos.
- **Pruebas sin el juego**: simulador, grabaciones y modelo falso.

Arquitectura y decisiones: [docs/architecture.md](docs/architecture.md).

## Estado

**Fase 0, cimientos.** Configuración, registro de eventos, capa de modelos (falso y
Anthropic), grabación y reproducción de llamadas, manifiesto de run, CLI, pruebas y CI.

| Fase | Contenido | Estado |
|---|---|---|
| 0 | Cimientos | ✅ |
| 1 | Simulador de escenarios + decisor + directivas | ⏳ |
| 2 | Memoria y personalidad | |
| 3 | Percepción offline (grabaciones) | |
| 4 | Guardia anti-metagaming y evaluaciones | |
| 5 | Control en modo sombra (la IA propone, tú ejecutas) | |
| 6 | Autonomía en vivo | |
| 7 | Overlay y OBS | |
| 8 | Arena multi-modelo | |

## Puesta en marcha

Requisitos: Python ≥ 3.12 y [uv](https://docs.astral.sh/uv/).

```powershell
uv sync                     # crea .venv e instala dependencias
uv run aibh check           # valida la configuración
uv run aibh run -p offline  # run vacía con el modelo falso
uv run aibh ping -p offline # una llamada de prueba (modelo falso)
uv run pytest               # pruebas (sin juego ni APIs)
```

Con un modelo real (necesita `ANTHROPIC_API_KEY` o `ant auth login`):

```powershell
uv run aibh ping -p default -r decider        # Opus
uv run aibh ping -p default -r decider_fast   # Haiku
uv run pytest -m live                         # pruebas de contrato contra la API
```

Cada run deja una carpeta en `runs/` con `manifest.json` (modelos, prompts, coste),
`events.jsonl` (todo lo que pasó) y `llm_calls.jsonl` (cada llamada, reproducible).

```powershell
uv run aibh events runs/<carpeta>
uv run aibh events runs/<carpeta> -t llm.call
```

## Organización

```
config/            ajustes, perfiles de modelo y perfiles de run (TOML)
prompts/           constitución, personajes, directivas (Markdown)
src/aibecomeshuman/
  core/            registro de eventos y reloj
  llm/             puerto de modelos, adaptadores, grabación/cassettes
  run/             sesión de run y manifiesto
  cli/             comando aibh
tests/             unit/ · integration/ · contract/ (live)
docs/              arquitectura y diario de desarrollo
data/              capturas locales (no se suben a git)
runs/              salidas de partidas (no se suben a git)
```

## Cambiar de modelo o de directiva

- **Modelo**: edita `config/runs/<run>.toml` y asigna otro perfil a un rol, o crea un
  perfil nuevo en `config/models/`.
- **Directiva**: crea `prompts/directives/<nombre>.md` y pon `directive = "<nombre>"` en el run.

## Aviso

Proyecto de fans sin relación con los titulares de *Detroit: Become Human*. El
repositorio no contiene recursos del juego: capturas, grabaciones y textos del juego
se quedan en local (`data/`, ignorado por git).
