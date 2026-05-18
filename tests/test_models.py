from mealie_llm_server.models import (
    ChatCompletionRequest,
    build_chat_completion_response,
)


class TestChatCompletionRequest:
    def test_parse_minimal_request(self):
        data = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
        }
        req = ChatCompletionRequest.model_validate(data)
        assert req.model == "gpt-4o"
        assert len(req.messages) == 2
        assert req.temperature is None
        assert req.stream is False

    def test_extract_system_message(self):
        data = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "Parse ingredients."},
                {"role": "user", "content": '["1 cup flour"]'},
            ],
        }
        req = ChatCompletionRequest.model_validate(data)
        assert req.system_message == "Parse ingredients."

    def test_no_system_message_returns_none(self):
        data = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        req = ChatCompletionRequest.model_validate(data)
        assert req.system_message is None

    def test_response_format_preserved(self):
        data = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "test"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "Test", "strict": True, "schema": {}},
            },
        }
        req = ChatCompletionRequest.model_validate(data)
        assert req.response_format["type"] == "json_schema"

    def test_extra_fields_preserved(self):
        """Mealie may send fields we don't explicitly model. They must survive for proxy."""
        data = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "test"}],
            "top_p": 0.9,
            "frequency_penalty": 0.5,
        }
        req = ChatCompletionRequest.model_validate(data)
        assert req.model_extra["top_p"] == 0.9


class TestChatCompletionResponse:
    def test_build_response(self):
        resp = build_chat_completion_response(
            content='{"ingredients": []}',
            model="nuextract-2.0-2b",
        )
        assert resp.object == "chat.completion"
        assert resp.choices[0].message.content == '{"ingredients": []}'
        assert resp.choices[0].message.role == "assistant"
        assert resp.model == "nuextract-2.0-2b"
        assert resp.id.startswith("chatcmpl-")
