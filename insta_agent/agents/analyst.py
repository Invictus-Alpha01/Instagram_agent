"""Agent 1. Reads past performance and decides the lines for the next carousel."""

from __future__ import annotations

import json
import logging

from insta_agent.agents.support import request_model
from insta_agent.config import Settings
from insta_agent.models import ContentBrief, PastPost
from insta_agent.services.openai_service import OpenAIService

logger = logging.getLogger(__name__)

_SYSTEM = """You are the performance analyst for one Instagram account. You decide the content lines for the next carousel. You do not write the final caption and you do not describe photographs.

Study the ranked past posts. Prefer themes and structures that earned saves, shares, and comments. A like with no saves is a weak signal. Do not repeat a hook from the last few posts. When history is empty, pick one specific useful angle inside the niche instead of a generic introduction.

Return exactly the requested number of slides. Slide 1 is the hook. The last slide is a payoff or a soft next step, not a hard sell. Every middle slide carries one concrete line a reader could repeat later. Lines are claims or steps, not labels like "Tip 2". Stay inside the niche and the brand voice."""


class Analyst:
    """Chooses the next carousel's theme and the line each slide must deliver."""

    def __init__(self, service: OpenAIService, settings: Settings):
        self._service = service
        self._settings = settings

    def decide(self, posts: list[PastPost]) -> ContentBrief:
        ranked = sorted(posts, key=lambda post: post.interaction_weight, reverse=True)[:20]
        logger.info("Analyst is reading %s past post(s).", len(ranked))
        payload = {
            "niche": self._settings.account_niche,
            "brand": self._settings.brand_name,
            "voice": self._settings.brand_voice,
            "slide_count": self._settings.carousel_slides,
            "posts_best_first": [_post_view(post) for post in ranked],
        }
        if not ranked:
            payload["note"] = "No published posts yet. Choose a strong first angle inside the niche."
        user = (
            "Decide the next carousel from this performance history.\n"
            f"{json.dumps(payload, indent=2, ensure_ascii=False)}"
        )
        brief = request_model(
            self._service,
            model=self._settings.openai_text_model,
            system=_SYSTEM,
            user=user,
            response_format=ContentBrief,
            temperature=0.4,
            validate=lambda result: _validate(result, self._settings.carousel_slides),
        )
        logger.info("Analyst chose theme: %s", brief.theme)
        logger.info("Analyst rationale: %s", brief.rationale)
        return brief


def _post_view(post: PastPost) -> dict:
    view = {
        "published_at": post.published_at,
        "media_type": post.media_type,
        "theme": post.theme,
        "caption_excerpt": post.caption[:500],
        "likes": post.like_count,
        "comments": post.comments_count,
        "interaction_weight": post.interaction_weight,
    }
    if post.slides:
        view["slide_lines"] = [slide.headline for slide in post.slides]
    for name in ("reach", "saved", "shares", "views"):
        value = getattr(post, name)
        if value is not None:
            view[name] = value
    return view


def _validate(brief: ContentBrief, slide_count: int) -> None:
    if len(brief.slides) != slide_count:
        raise ValueError(f"Return exactly {slide_count} slides, not {len(brief.slides)}.")
    if not brief.theme.strip() or not brief.rationale.strip():
        raise ValueError("Theme and rationale must both be filled in.")
    for index, slide in enumerate(brief.slides, start=1):
        if not slide.line.strip():
            raise ValueError(f"Slide {index} is missing its content line.")
    if brief.slides[0].role.strip().lower() != "hook":
        raise ValueError("Slide 1 role must be hook.")
    if brief.slides[-1].role.strip().lower() != "cta":
        raise ValueError("The last slide role must be cta.")
