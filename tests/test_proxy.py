import json
import pytest
import httpx
import respx
from mealie_llm_server.proxy import ProxyHandler


@pytest.fixture
def proxy():
    return ProxyHandler(upstream_url="http://upstream.test/v1", upstream_api_key="upstream-key", timeout=30)


@pytest.fixture
def proxy_no_upstream():
    return ProxyHandler(upstream_url=None, upstream_api_key=None)


class TestForwardChatCompletion:
    @respx.mock
    @pytest.mark.asyncio
    async def test_forwards_request_body(self, proxy):
        body = json.dumps({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}).encode()
        upstream_route = respx.post("http://upstream.test/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": "hello"}}]})
        )
        resp = await proxy.forward_chat_completion(body, "application/json", stream=False)
        assert upstream_route.called
        sent_body = json.loads(upstream_route.calls[0].request.content)
        assert sent_body["model"] == "gpt-4o"

    @respx.mock
    @pytest.mark.asyncio
    async def test_replaces_auth_header(self, proxy):
        body = json.dumps({"model": "gpt-4o", "messages": []}).encode()
        upstream_route = respx.post("http://upstream.test/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={})
        )
        await proxy.forward_chat_completion(body, "application/json", stream=False)
        auth = upstream_route.calls[0].request.headers["authorization"]
        assert auth == "Bearer upstream-key"

    @respx.mock
    @pytest.mark.asyncio
    async def test_forwards_content_type(self, proxy):
        body = json.dumps({"model": "gpt-4o", "messages": []}).encode()
        upstream_route = respx.post("http://upstream.test/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={})
        )
        await proxy.forward_chat_completion(body, "application/json", stream=False)
        ct = upstream_route.calls[0].request.headers["content-type"]
        assert "application/json" in ct

    @pytest.mark.asyncio
    async def test_returns_501_when_no_upstream_configured(self, proxy_no_upstream):
        body = json.dumps({"model": "gpt-4o", "messages": []}).encode()
        resp = await proxy_no_upstream.forward_chat_completion(body, "application/json", stream=False)
        assert resp.status_code == 501

    @respx.mock
    @pytest.mark.asyncio
    async def test_streaming_passthrough(self, proxy):
        body = json.dumps({"model": "gpt-4o", "messages": [], "stream": True}).encode()
        respx.post("http://upstream.test/v1/chat/completions").mock(
            return_value=httpx.Response(200, content=b"data: {}\n\n", headers={"content-type": "text/event-stream"})
        )
        resp = await proxy.forward_chat_completion(body, "application/json", stream=True)
        assert resp.media_type == "text/event-stream"


class TestForwardAudioTranscription:
    @respx.mock
    @pytest.mark.asyncio
    async def test_audio_transcription_proxy(self, proxy):
        body = b"audio-data"
        upstream_route = respx.post("http://upstream.test/v1/audio/transcriptions").mock(
            return_value=httpx.Response(200, json={"text": "hello world"})
        )
        resp = await proxy.forward_audio_transcription(body, "multipart/form-data")
        assert upstream_route.called
        assert resp.status_code == 200
