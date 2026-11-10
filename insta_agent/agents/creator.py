"""Agent 2. Turns the analyst's lines into caption, slide copy, and image prompts."""

from __future__ import annotations

import json
import logging

from insta_agent.agents.support import request_model
from insta_agent.config import Settings
from insta_agent.models import CarouselDraft, ContentBrief, VisualReview
from insta_agent.services.openrouter_service import OpenRouterService

logger = logging.getLogger(__name__)

_SYSTEM = """You are the carousel creator for one Instagram account. You turn an approved content brief into a publishable carousel.

Preserve the meaning of each brief line. Write a short headline and one supporting sentence per slide. Headlines are at most 8 words. Supporting lines are at most 18 words. Slide 1 must be understandable with no context. The last slide closes the idea without begging for a follow.

Write the caption in the brand voice, with line breaks, under 1800 characters, and at most 8 relevant hashtags at the end. No markdown and no label such as "Caption:".

Write one visual_style sentence that locks lighting, lens, color, and era for the whole carousel. For each slide, write an image_prompt for one photograph that supports that line. Describe a scene only. Do not request text, letters, numbers, logos, watermarks, collages, or frames. Keep the same kind of subject and treatment so the slides feel like one series."""


class Creator:
    """Writes the post, then revises only the slides the visual editor rejected."""
    print("Creator is initializing_1")

    def __init__(self, service: OpenRouterService, settings: Settings):
        self._service = service
        self._settings = settings
        print("Creator is initialized_2")   

    def draft(self, brief: ContentBrief) -> CarouselDraft:
        logger.info("Creator is writing %s slides.", self._settings.carousel_slides)
        print(f"Creator is writing slides_3{self._settings.carousel_slides}{brief.model_dump()}")
        return self._write("Write the carousel for this brief.", brief, current=None, review=None)

    def revise(self, brief: ContentBrief, current: CarouselDraft, review: VisualReview) -> CarouselDraft:
        logger.info("Creator is revising after the visual check: %s", review.summary)
        instruction = (
            "The visual editor rejected this carousel. Change only the slides and prompts the review "
            "calls out. Keep approved slides' headlines, bodies, and image prompts identical unless a "
            "series issue requires a new visual_style. If you change visual_style, update every "
            "image_prompt so the set still matches, and keep approved headlines and bodies the same."
        )
        return self._write(instruction, brief, current=current, review=review)

    def _write(
        self,
        instruction: str,
        brief: ContentBrief,
        *,
        current: CarouselDraft | None,
        review: VisualReview | None,
    ) -> CarouselDraft:
        payload: dict = {
            "brand": self._settings.brand_name,
            "voice": self._settings.brand_voice,
            "niche": self._settings.account_niche,
            "brief": brief.model_dump(),
        }
        if current is not None and review is not None:
            payload["previous_draft"] = current.model_dump()
            payload["visual_review"] = review.model_dump()
        user = f"{instruction}\n{json.dumps(payload, indent=2, ensure_ascii=False)}"
        draft = request_model(
            self._service,
            model=self._settings.text_model,
            system=_SYSTEM,
            user=user,
            response_format=CarouselDraft,
            temperature=0.7,
            validate=lambda result: _validate(result, self._settings.carousel_slides),
        )
        logger.info("Creator finished the caption and %s slides.", len(draft.slides))
        return draft


def _validate(draft: CarouselDraft, slide_count: int) -> None:
    if len(draft.slides) != slide_count:
        raise ValueError(f"Return exactly {slide_count} slides, not {len(draft.slides)}.")
    if not draft.visual_style.strip():
        raise ValueError("visual_style must describe the shared look of the photographs.")
    caption = draft.caption.strip()
    if not caption:
        raise ValueError("The caption is empty.")
    if len(caption) > 2200:
        raise ValueError("The caption is over Instagram's 2200 character limit. Shorten it.")
    hashtags = [word for word in caption.split() if word.startswith("#")]
    if len(hashtags) > 8:
        raise ValueError("Use at most 8 hashtags.")
    for index, slide in enumerate(draft.slides, start=1):
        if not slide.headline.strip():
            raise ValueError(f"Slide {index} needs a headline.")
        if len(slide.headline.split()) > 10:
            raise ValueError(f"Slide {index} headline is longer than 10 words. Shorten it.")
        if len(slide.body.split()) > 22:
            raise ValueError(f"Slide {index} body is longer than 22 words. Shorten it.")
        if len(slide.image_prompt.split()) < 8:
            raise ValueError(f"Slide {index} image_prompt is too vague. Describe one photograph.")
        if not slide.alt_text.strip():
            raise ValueError(f"Slide {index} needs alt text.")
