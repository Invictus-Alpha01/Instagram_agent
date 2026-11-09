"""Ask a model for a structured object, and retry when the shape is wrong."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel

from insta_agent.errors import AgentError
from insta_agent.services.openai_service import OpenAIService

T = TypeVar("T", bound=BaseModel)


def request_model(
    service: OpenAIService,
    *,
    model: str,
    system: str,
    user: str,
    response_format: type[T],
    temperature: float,
    validate: Callable[[T], None],
    images: list[tuple[str, str]] | None = None,
    attempts: int = 3,
) -> T:
    """Call the model and apply domain checks. Schema mistakes are sent back once or twice."""

    feedback = ""
    last_error: Exception | None = None
    for _ in range(attempts):
        prompt = user
        if feedback:
            prompt = (
                f"{user}\n\nYour previous response was rejected: {feedback}\n"
                "Return a corrected object that satisfies every constraint."
            )
        try:
            result = service.parse(
                model=model,
                system=system,
                user=prompt,
                response_format=response_format,
                temperature=temperature,
                images=images,
            )
            validate(result)
            return result
        except ValueError as exc:
            last_error = exc
            feedback = str(exc)[:1500]
        except AgentError as exc:
            text = str(exc)
            if text.startswith("OpenAI request failed") or text.startswith("Could not reach"):
                raise
            last_error = exc
            feedback = text[:1500]
    raise AgentError(f"The model did not return a usable result: {last_error}")
