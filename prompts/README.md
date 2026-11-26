# prompts/

Todos los textos que recibe el modelo viven aquí, separados del código. Cada run guarda
en su manifiesto el hash de cada fichero, así que siempre se sabe con qué prompts se jugó.

Pila de capas (de más fija a más variable), tal como se montará en la fase 1:

1. `constitution/`: reglas fijas: dentro del mundo, solo evidencia, formato.
2. `roles/`: instrucciones por rol técnico (observador, decisor, reflexor, juez). *Fase 1.*
3. `characters/`: ficha base del personaje (escrita por nosotros, nunca texto del juego).
4. *Personalidad actual*: la genera la memoria. *Fase 2.*
5. `directives/`: directiva de la run (`default.md` o cualquier override).
6. *Contexto compuesto y tarea*: los monta el compositor de contexto.

Para cambiar la directiva de una run basta con crear `directives/<nombre>.md` y poner
`directive = "<nombre>"` en `config/runs/<run>.toml`. No hay que tocar código.
