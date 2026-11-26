# Arquitectura

Resumen de las decisiones acordadas. Cada fase añade su detalle aquí.

## Principio central: registro de eventos

Ningún módulo habla directamente con otro. Todo pasa por un **registro de eventos
append-only** (`runs/<run>/events.jsonl`):

- percepción, control y cognición **publican** eventos;
- memoria, overlay de OBS y marcadores de montaje son **proyecciones** que los leen.

Por eso una partida se puede reproducir, depurar y comparar entre modelos, y el juego real
es solo una de las implementaciones del **puerto de entorno** (vivo, sombra, replay, escenario).

## Modos de control

| Modo | Quién controla | La IA… |
|---|---|---|
| OBSERVAR (por defecto) | Operador | mira y memoriza, sin enviar input |
| PROPONER (sombra) | Operador | decide y lo enseña; el operador ejecuta |
| AUTÓNOMO | IA | decide y ejecuta selecciones de opciones |
| PAUSA / KILL | Nadie | suelta todo el input virtual al momento |

- El input físico del operador siempre tiene prioridad.
- **Los QTE son siempre manuales.** Si aparece un QTE en modo AUTÓNOMO, la IA suelta el
  input y avisa ("QTE: tuyo"). El resultado del QTE se guarda en memoria como
  *no decidido por el agente*, para que la reflexión de personalidad no lo trate como
  una elección suya.
- El LLM decide la **intención** (qué opción). Un ejecutor local la traduce a pulsaciones.

## Roles de modelo

| Rol | Tarea | Por defecto |
|---|---|---|
| `observer` | fotogramas clave → observaciones estructuradas | rápido |
| `decider` | decisiones críticas sin tiempo + **postura de escena** | potente |
| `decider_fast` | elecciones con temporizador: postura + opciones → opción | rápido |
| `reflector` | evolución de personalidad al cerrar escenas/capítulos | potente |
| `judge` | detector de metagaming | rápido |
| `summarizer` | consolidación de memoria | rápido |

### Elecciones con temporizador: postura + modelo rápido

El razonamiento difícil se saca del momento con prisa:

1. Durante la escena, el `decider` mantiene una **postura** corta y actualizada
   ("quiero ganarme su confianza; no presiono; si me provoca, calma").
2. Cuando aparece una elección con tiempo, el `decider_fast` recibe la postura, las
   opciones y el diálogo inmediato, y solo tiene que elegir la opción que mejor encaja.
3. Si se agota el tiempo, se usa la opción más afín a la postura según un criterio
   simple, sin llamar a ningún modelo.

Validación (fase 1): pasar las mismas instantáneas de decisión a ambos modelos y medir
cuántas veces coinciden.

## Capa de modelos

- Puerto único `LLMProvider.complete(LLMRequest) -> LLMResponse` con formato interno propio
  (texto e imágenes) y salida estructurada por JSON Schema.
- Adaptadores por proveedor, registrados en `llm/registry.py`. Hoy: `fake` y `anthropic`.
- `RecordingProvider` guarda cada llamada y emite `llm.call` (métricas, sin contenido).
  `CassetteProvider` reproduce una grabación: base de pruebas repetibles y de la arena.
- En los perfiles de Opus, `refusal_fallback = true`: si el modelo rechaza una petición,
  la API la reintenta en servidor con un modelo de respaldo. Haiku no lo admite.

## Anti-metagaming (resumen)

1. **Percepción**: se oculta el diagrama de flujo de fin de capítulo, los menús, trofeos y estadísticas.
2. **Prompt**: en el mundo, sin el título del juego. Modo anonimizado experimental.
3. **Validación**: justificaciones con referencias obligatorias a la evidencia, lista de
   entidades conocidas, léxico de jugador y juez opcional.
4. **Medición**: escenarios canario y escenarios sintéticos de control.

## Pendiente de decidir

- Idioma y resolución del juego.
- Mando o teclado para el control manual (la IA usará mando virtual).
- Conocimiento entre protagonistas: separado o con una capa común de "jugador".
- Narración por voz del operador: permitida o no.
- Presupuesto de API y GPU para modelos locales.
