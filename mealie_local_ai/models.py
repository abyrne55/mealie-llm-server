from __future__ import annotations

import time
from uuid import uuid4

from pydantic import BaseModel, ConfigDict


class Message(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: str | list | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[Message]
    temperature: float | None = None
    stream: bool = False
    response_format: dict | None = None

    @property
    def system_message(self) -> str | None:
        for msg in self.messages:
            if msg.role == "system" and isinstance(msg.content, str):
                return msg.content
        return None


class ResponseMessage(BaseModel):
    role: str = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: ResponseMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


def build_chat_completion_response(
    content: str,
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid4().hex[:24]}",
        created=int(time.time()),
        model=model,
        choices=[Choice(message=ResponseMessage(content=content))],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )
