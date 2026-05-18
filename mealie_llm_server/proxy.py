from __future__ import annotations

import httpx
from starlette.responses import JSONResponse, Response, StreamingResponse


class ProxyHandler:
    def __init__(self, upstream_url: str | None, upstream_api_key: str | None, timeout: int = 300):
        self._upstream_url = upstream_url.rstrip("/") if upstream_url else None
        self._upstream_api_key = upstream_api_key
        self._client = httpx.AsyncClient(timeout=timeout) if upstream_url else None

    async def forward_chat_completion(self, request_body: bytes, content_type: str, stream: bool) -> Response:
        if not self._upstream_url or not self._client:
            return JSONResponse(
                {"error": "No upstream API configured. Set UPSTREAM_URL to enable proxying."},
                status_code=501,
            )
        headers = {
            "Content-Type": content_type,
            "Authorization": f"Bearer {self._upstream_api_key}",
        }
        if stream:
            resp = await self._client.post(
                f"{self._upstream_url}/chat/completions",
                content=request_body,
                headers=headers,
            )
            return StreamingResponse(
                content=resp.aiter_bytes(),
                media_type="text/event-stream",
                status_code=resp.status_code,
            )
        resp = await self._client.post(
            f"{self._upstream_url}/chat/completions",
            content=request_body,
            headers=headers,
        )
        return JSONResponse(content=resp.json(), status_code=resp.status_code)

    async def forward_audio_transcription(self, request_body: bytes, content_type: str) -> Response:
        if not self._upstream_url or not self._client:
            return JSONResponse(
                {"error": "No upstream API configured. Set UPSTREAM_URL to enable proxying."},
                status_code=501,
            )
        headers = {
            "Content-Type": content_type,
            "Authorization": f"Bearer {self._upstream_api_key}",
        }
        resp = await self._client.post(
            f"{self._upstream_url}/audio/transcriptions",
            content=request_body,
            headers=headers,
        )
        return JSONResponse(content=resp.json(), status_code=resp.status_code)
