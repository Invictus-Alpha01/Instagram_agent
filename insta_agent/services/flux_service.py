"""Flux text-to-image generation via the Black Forest Labs API."""

from __future__ import annotations

import logging
import time

import httpx

from insta_agent.config import Settings
from insta_agent.errors import AgentError

logger = logging.getLogger(__name__)

_FAILURE_STATUSES = {"Error", "Failed", "Request Moderated", "Content Moderated"}


class FluxService:
    """Submit a prompt, poll the returned URL, and download the image bytes."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = httpx.Client(
            timeout=httpx.Timeout(60.0, connect=20.0),
            headers={"accept": "application/json", "x-key": settings.flux_api_key},
            follow_redirects=True,
        )
        # Signed delivery URLs are fetched without the API key so the signature stays valid.
        self._downloads = httpx.Client(
            timeout=httpx.Timeout(60.0, connect=20.0),
            follow_redirects=True,
        )

    def close(self) -> None:
        self._client.close()
        self._downloads.close()

    def __enter__(self) -> FluxService:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def generate(self, prompt: str, *, slide_index: int) -> bytes:
        """Generate one JPEG. `slide_index` is 1-based and offsets an optional seed."""

        url = f"{self._settings.flux_api_base.rstrip('/')}/{self._settings.flux_model.lstrip('/')}"
        payload: dict = {
            "prompt": prompt,
            "width": self._settings.flux_width,
            "height": self._settings.flux_height,
            "output_format": "jpeg",
        }
        if self._settings.flux_seed is not None:
            payload["seed"] = self._settings.flux_seed + slide_index
        response = self._post(url, payload)
        polling_url = response.get("polling_url")
        if not polling_url:
            raise AgentError("Flux did not return a polling_url. Check FLUX_API_BASE and FLUX_MODEL.")
        sample_url = self._poll(polling_url)
        return self._download(sample_url)

    def _post(self, url: str, payload: dict) -> dict:
        try:
            response = self._client.post(url, json=payload)
        except httpx.HTTPError as exc:
            raise AgentError(f"Could not reach Flux: {exc}") from exc
        if response.status_code >= 400:
            raise AgentError(f"Flux rejected the request ({response.status_code}): {_short(response.text)}")
        return response.json()

    def _poll(self, polling_url: str) -> str:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            try:
                response = self._client.get(polling_url)
            except httpx.HTTPError as exc:
                raise AgentError(f"Flux polling failed: {exc}") from exc
            if response.status_code >= 400:
                raise AgentError(f"Flux polling failed ({response.status_code}): {_short(response.text)}")
            body = response.json()
            status = body.get("status", "")
            logger.debug("Flux status: %s", status)
            if status == "Ready":
                sample = (body.get("result") or {}).get("sample")
                if not sample:
                    raise AgentError("Flux reported Ready without an image URL.")
                return sample
            if status in _FAILURE_STATUSES:
                raise AgentError(f"Flux generation {status}.")
            time.sleep(0.8)
        raise AgentError("Flux did not finish within 3 minutes.")

    def _download(self, sample_url: str) -> bytes:
        try:
            with self._downloads.stream("GET", sample_url) as response:
                response.raise_for_status()
                chunks: list[bytes] = []
                total = 0
                for chunk in response.iter_bytes():
                    total += len(chunk)
                    if total > 15 * 1024 * 1024:
                        raise AgentError("Flux image was larger than 15MB.")
                    chunks.append(chunk)
        except httpx.HTTPError as exc:
            raise AgentError(f"Could not download the Flux image: {exc}") from exc
        if not chunks:
            raise AgentError("Flux returned an empty image.")
        return b"".join(chunks)


def _short(text: str) -> str:
    collapsed = " ".join(text.split())
    return collapsed[:300]
