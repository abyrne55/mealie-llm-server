from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from mealie_local_ai.models import ChatCompletionRequest, ChatCompletionResponse

if TYPE_CHECKING:
    from llama_cpp import Llama
    from mealie_local_ai.mealie_client import MealieClient


class Handler(ABC):
    reference_prompt: str
    model_key: str

    @abstractmethod
    async def handle(
        self,
        request: ChatCompletionRequest,
        model: Llama,
        mealie_client: MealieClient,
    ) -> ChatCompletionResponse: ...
