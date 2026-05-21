from importlib.resources import files

from mealie_local_ai.handlers.base import Handler
from mealie_local_ai.models import ChatCompletionResponse


class GenericTextHandler(Handler):
    model_key = "general"

    def __init__(self):
        self.reference_prompt = files("mealie_local_ai.prompts").joinpath("scrape-recipe.txt").read_text()

    async def handle(self, request, model, mealie_client) -> ChatCompletionResponse:
        raise NotImplementedError("GenericTextHandler is not yet implemented")
