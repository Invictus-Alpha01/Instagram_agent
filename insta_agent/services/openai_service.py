"""OpenAI chat calls shared by the three agents.

The HTTP API is used directly so the program does not depend on the OpenAI SDK's
native parser, which some Windows machines block.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from insta_agent.config import Settings
from insta_agent.errors import AgentError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class OpenAIService:
    """Structured chat completions, including vision inputs for the visual editor."""

    def __init__(self, settings: Settings):
        self._client = httpx.Client(
            base_url="https://api.openai.com/v1",
            timeout=httpx.Timeout(120.0, connect=20.0),
            headers={
                "Authorization": f"Bearer {settings.openai_api_key}",
                "Content-Type": "application/json",
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
                content.append(
                    {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}}
                )
        else:
            content = user
        payload: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.__name__,
                    "strict": True,
                    "schema": _strict_schema(response_format),
                },
            },
        }
        if _supports_temperature(model):
            payload["temperature"] = temperature

        last_error: Exception | None = None
        body: dict | None = None
        for attempt in range(3):
            try:
                response = self._client.post("/chat/completions", json=payload)
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt < 2:
                    logger.warning("OpenAI connection failed, retrying.")
                    time.sleep(2 * (attempt + 1))
                    continue
                raise AgentError("Could not reach the OpenAI API.") from exc
            if response.status_code in {429, 500, 502, 503} and attempt < 2:
                logger.warning("OpenAI %s returned %s, retrying.", model, response.status_code)
                time.sleep(2 * (attempt + 1))
                continue
            if response.status_code >= 400:
                raise AgentError(_error_text(response))
            try:
                body = response.json()
            except ValueError as exc:
                raise AgentError("OpenAI returned a response that was not JSON.") from exc
            break
        if body is None:
            raise AgentError(f"Could not reach the OpenAI API: {last_error}")

        try:
            message = body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise AgentError("OpenAI returned a completion without a message.") from exc
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


def _strict_schema(model: type[BaseModel]) -> dict:
    """Pydantic's schema plus the flags OpenAI strict mode requires."""

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


def _error_text(response: httpx.Response) -> str:
    message = ""
    try:
        message = str(response.json().get("error", {}).get("message", ""))
    except ValueError:
        message = ""
    collapsed = " ".join(message.split())
    if collapsed:
        return f"OpenAI request failed ({response.status_code}). {collapsed[:300]}"
    return f"OpenAI request failed ({response.status_code})."


def _supports_temperature(model: str) -> bool:
    name = model.lower()
    return name.startswith(("gpt-4o", "gpt-4.1", "gpt-3.5"))
