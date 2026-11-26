"""Internal, provider-independent message format."""

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
    purpose: str = Field(description="Task label for logs (e.g. 'decide.dialogue')")
    system: str = ""
    messages: list[Message]
    json_schema: dict[str, Any] | None = None
    max_tokens: int | None = None

    def fingerprint(self) -> str:
        """Stable content hash: same request -> same fingerprint (used by cassettes)."""
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def redacted(self) -> dict[str, Any]:
        """Log-friendly version: images are replaced by their hash (keeps files small)."""
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
    """Model layer failure (network, format, missing cassette entry...)."""


class LLMRefusal(LLMError):
    """The model refused the request."""


class BudgetExceeded(LLMError):
    """A paid call was blocked because a budget cap has been reached."""
