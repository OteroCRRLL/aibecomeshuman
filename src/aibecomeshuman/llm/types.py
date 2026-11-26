"""Formato interno de mensajes, independiente del proveedor."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImagePart(BaseModel):
    type: Literal["image"] = "image"
    media_type: str = "image/png"
    data_b64: str


ContentPart = Annotated[TextPart | ImagePart, Field(discriminator="type")]


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: list[ContentPart]

    @classmethod
    def user(cls, text: str) -> Message:
        return cls(role="user", content=[TextPart(text=text)])

    @classmethod
    def assistant(cls, text: str) -> Message:
        return cls(role="assistant", content=[TextPart(text=text)])


class LLMRequest(BaseModel):
    purpose: str = Field(description="Etiqueta de la tarea, para logs (p. ej. 'decide.dialogue')")
    system: str = ""
    messages: list[Message]
    json_schema: dict[str, Any] | None = None
    max_tokens: int | None = None

    def fingerprint(self) -> str:
        """Huella estable del contenido: misma petición -> misma huella (para cassettes)."""
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def redacted(self) -> dict[str, Any]:
        """Versión para logs: las imágenes se sustituyen por su hash (no inflan el fichero)."""
        data = self.model_dump(mode="json")
        for message in data["messages"]:
            for part in message["content"]:
                if part["type"] == "image":
                    raw = part.pop("data_b64").encode("ascii")
                    part["sha256"] = hashlib.sha256(raw).hexdigest()
        return data

    def last_user_text(self) -> str:
        for message in reversed(self.messages):
            if message.role == "user":
                return "\n".join(p.text for p in message.content if isinstance(p, TextPart))
        return ""


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0


class LLMResponse(BaseModel):
    text: str
    parsed: dict[str, Any] | None = None
    stop_reason: str | None = None
    usage: Usage = Field(default_factory=Usage)
    provider: str
    model: str
    latency_ms: float = 0.0
    request_id: str | None = None


class LLMError(Exception):
    """Fallo de la capa de modelos (red, formato, cassette sin entrada...)."""


class LLMRefusal(LLMError):
    """El modelo rechazó la petición."""
