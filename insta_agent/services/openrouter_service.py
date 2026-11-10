"""OpenRouter calls for the three agents and for photograph generation.

Chat completions use the OpenAI-compatible endpoint. Photographs use the
Image API. Both authenticate with OPENROUTER_API_KEY.
"""

from __future__ import annotations

import base64
import binascii
import logging
import time
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from insta_agent.config import Settings
from insta_agent.errors import AgentError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_RETRY_STATUSES = {429, 500, 502, 503}
_IMAGE_LIMIT = 15 * 1024 * 1024


class OpenRouterService:
    """Structured chat, including vision, and text-to-image generation."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = httpx.Client(
            base_url=settings.openrouter_api_base.rstrip("/"),
            timeout=httpx.Timeout(180.0, connect=20.0),
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                "Content-Type": "application/json",
                "X-Title": "instagram-carousel-agent",
            },
        )

    def close(self) -> None:
        self._client.close()

    def parse(
        self,
        *,
        model: str,
        system: str,
        user: str,
        response_format: type[T],
        temperature: float,
        images: list[tuple[str, str]] | None = None,
    ) -> T:
        """Return one structured object. `images` are (label, data-url) pairs."""

        if images:
            content: str | list[dict] = [{"type": "text", "text": user}]
            for label, data_url in images:
                content.append({"type": "text", "text": label})
                content.append({"type": "image_url", "image_url": {"url": data_url}})
        else:
            content = user
        payload: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "temperature": temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.__name__,
                    "strict": True,
                    "schema": _strict_schema(response_format),
                },
            },
            "provider": {"require_parameters": True},
        }

        last_error: Exception | None = None
        body: dict | None = None
        for attempt in range(3):
            try:
                response = self._client.post("/chat/completions", json=payload)
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt < 2:
                    logger.warning("OpenRouter connection failed, retrying.")
                    time.sleep(2 * (attempt + 1))
                    continue
                raise AgentError("Could not reach OpenRouter.") from exc
            if response.status_code == 400 and "temperature" in payload:
                message = _error_message(response).lower()
                if "temperature" in message:
                    logger.info("OpenRouter model %s ignored temperature; retrying without it.", model)
                    payload.pop("temperature")
                    continue
            if response.status_code in _RETRY_STATUSES and attempt < 2:
                logger.warning("OpenRouter %s returned %s, retrying.", model, response.status_code)
                time.sleep(2 * (attempt + 1))
                continue
            if response.status_code >= 400:
                raise AgentError(_failure("OpenRouter request failed", response))
            try:
                body = response.json()
            except ValueError as exc:
                raise AgentError("OpenRouter returned a response that was not JSON.") from exc
            break
        if body is None:
            raise AgentError(f"Could not reach OpenRouter: {last_error}")

        try:
            message = body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise AgentError("OpenRouter returned a completion without a message.") from exc
        refusal = message.get("refusal")
        if refusal:
            raise AgentError(str(refusal))
        text = _message_text(message.get("content"))
        if not text:
            raise AgentError("The model returned no structured result.")
        try:
            return response_format.model_validate_json(text)
        except ValidationError as exc:
            raise AgentError(f"{model} returned JSON that did not match the schema: {exc}") from exc

    def generate_image(self, prompt: str, *, slide_index: int) -> bytes:
        """Generate one photograph and return its bytes."""

        payload: dict = {
            "model": self._settings.image_model,
            "prompt": prompt,
            "n": 1,
            "aspect_ratio": self._settings.image_aspect_ratio,
            "resolution": self._settings.image_resolution,
        }
        if self._settings.image_seed is not None:
            payload["seed"] = self._settings.image_seed + slide_index

        last_error: Exception | None = None
        body: dict | None = None
        for attempt in range(3):
            try:
                response = self._client.post("/images", json=payload)
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt < 2:
                    logger.warning("OpenRouter image connection failed, retrying.")
                    time.sleep(2 * (attempt + 1))
                    continue
                raise AgentError("Could not reach OpenRouter.") from exc
            if response.status_code == 400 and "seed" in payload:
                message = _error_message(response).lower()
                if "seed" in message:
                    logger.info("OpenRouter image model ignored the seed; retrying without it.")
                    payload.pop("seed")
                    continue
            if response.status_code in _RETRY_STATUSES and attempt < 2:
                logger.warning(
                    "OpenRouter image model returned %s, retrying.",
                    response.status_code,
                )
                time.sleep(2 * (attempt + 1))
                continue
            if response.status_code >= 400:
                raise AgentError(_image_http_error(response))
            try:
                body = response.json()
            except ValueError as exc:
                raise AgentError("OpenRouter returned an image response that was not JSON.") from exc
            break
        if body is None:
            raise AgentError(f"Could not reach OpenRouter: {last_error}")

        encoded = _first_image(body)
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise AgentError("OpenRouter returned image data that could not be decoded.") from exc
        if not raw:
            raise AgentError("OpenRouter returned an empty image.")
        if len(raw) > _IMAGE_LIMIT:
            raise AgentError("OpenRouter image was larger than 15MB.")
        return raw


def _first_image(body: dict) -> str:
    data = body.get("data")
    if not isinstance(data, list) or not data:
        raise AgentError("OpenRouter returned no image.")
    item = data[0]
    if not isinstance(item, dict):
        raise AgentError("OpenRouter returned no image.")
    encoded = item.get("b64_json")
    if not isinstance(encoded, str) or not encoded.strip():
        raise AgentError("OpenRouter returned no image.")
    return encoded


def _image_http_error(response: httpx.Response) -> str:
    message = _error_message(response)
    lowered = message.lower()
    if response.status_code == 400 and any(
        marker in lowered for marker in ("moderat", "safety", "content policy", "flagged")
    ):
        detail = f" {message[:300]}" if message else ""
        return f"Image generation refused this prompt.{detail}"
    return _failure("OpenRouter image request failed", response)


def _strict_schema(model: type[BaseModel]) -> dict:
    """Pydantic's schema plus the flags strict JSON mode requires."""

    schema = _strict(model.model_json_schema())
    if not isinstance(schema, dict):
        raise AgentError(f"{model.__name__} did not produce an object schema.")
    return schema


def _strict(node: object) -> object:
    if isinstance(node, list):
        return [_strict(item) for item in node]
    if not isinstance(node, dict):
        return node
    cleaned: dict = {}
    for key, value in node.items():
        if key == "title":
            continue
        cleaned[key] = _strict(value)
    if cleaned.get("type") == "object":
        cleaned["additionalProperties"] = False
        properties = cleaned.get("properties")
        if isinstance(properties, dict):
            cleaned["required"] = list(properties.keys())
    return cleaned


def _message_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in {None, "text"}:
                parts.append(str(item.get("text", "")))
        return "".join(parts).strip()
    return ""


def _failure(prefix: str, response: httpx.Response) -> str:
    message = _error_message(response)
    if message:
        return f"{prefix} ({response.status_code}). {message[:300]}"
    return f"{prefix} ({response.status_code})."


def _error_message(response: httpx.Response) -> str:
    try:
        error = response.json().get("error", {})
    except ValueError:
        return " ".join(response.text.split())[:300]
    if isinstance(error, str):
        return " ".join(error.split())
    if isinstance(error, dict):
        return " ".join(str(error.get("message", "")).split())
    return ""
