"""Agent 3. Looks at the rendered slides and blocks publishing when they fail."""

from __future__ import annotations

import base64
import logging
from io import BytesIO
from pathlib import Path

from PIL import Image

from insta_agent.agents.support import request_model
from insta_agent.config import Settings
from insta_agent.models import CarouselDraft, ContentBrief, SlideReview, VisualReview
from insta_agent.services.openrouter_service import OpenRouterService

logger = logging.getLogger(__name__)

_SYSTEM = """You are the visual editor for an Instagram carousel. You are the last check before it can be published. You are shown the rendered slides, with the headline and body already laid over each photograph.

Approve a slide only when all of these are true:
- The headline and body match the expected copy, are spelled correctly, and are fully visible.
- Text is not crowded, clipped, or sitting on a busy part of the photo.
- The photograph supports that slide's message.
- The photograph contains no text, watermark, logo, or obvious warped objects.
- Important details stay off the extreme edges.

Approve the carousel only when every slide passes and the slides look like one series. When you reject something, name the slide and the specific change a creator can make. Do not reject a clear, on-brief slide over taste alone. If an image is missing or you cannot see it, reject that slide."""


class VisualQA:
    """Reviews the finished JPEGs. Publishing happens only after this agent approves."""

    def __init__(self, service: OpenRouterService, settings: Settings):
        self._service = service
        self._settings = settings

    def review(
        self,
        brief: ContentBrief,
        draft: CarouselDraft,
        slide_paths: list[Path],
    ) -> VisualReview:
        logger.info("Visual editor is checking %s slides.", len(slide_paths))
        expected = _expected_copy(brief, draft)
        images = [
            (f"Rendered slide {index}", _data_url(path))
            for index, path in enumerate(slide_paths, start=1)
        ]
        user = (
            "Review this carousel before it is published.\n"
            f"Shared visual style requested: {draft.visual_style}\n\n"
            f"{expected}"
        )
        result = request_model(
            self._service,
            model=self._settings.vision_model,
            system=_SYSTEM,
            user=user,
            response_format=VisualReview,
            temperature=0.1,
            images=images,
            validate=lambda review: _validate_indexes(review, len(slide_paths)),
        )
        normalized = _normalize(result, len(slide_paths))
        if normalized.approved:
            logger.info("Visual editor approved the carousel.")
        else:
            logger.info("Visual editor rejected the carousel: %s", normalized.summary)
        return normalized


def _expected_copy(brief: ContentBrief, draft: CarouselDraft) -> str:
    lines = []
    for index, slide in enumerate(draft.slides, start=1):
        claim = brief.slides[index - 1].line if index - 1 < len(brief.slides) else ""
        lines.append(
            f"Slide {index} must show headline '{slide.headline}' and body '{slide.body}'. "
            f"It must communicate: {claim}"
        )
    return "\n".join(lines)


def _data_url(path: Path) -> str:
    with Image.open(path) as image:
        preview = image.convert("RGB")
        preview.thumbnail((1024, 1280), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        preview.save(buffer, format="JPEG", quality=80)
    encoded = base64.standard_b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _validate_indexes(review: VisualReview, slide_count: int) -> None:
    indexes = [slide.index for slide in review.slides]
    expected = list(range(1, slide_count + 1))
    if sorted(indexes) != expected:
        raise ValueError(f"Return one review for each slide index {expected}.")


def _normalize(review: VisualReview, slide_count: int) -> VisualReview:
    """An approved carousel requires every slide to pass, plus no series-level issue."""

    slides: list[SlideReview] = []
    for slide in sorted(review.slides, key=lambda item: item.index):
        issues = [issue.strip() for issue in slide.issues if issue.strip()]
        if slide.approved:
            issues = []
        elif not issues:
            issues = ["Rejected without a specific fix. Adjust the photograph or shorten the copy."]
        slides.append(SlideReview(index=slide.index, approved=slide.approved, issues=issues))
    series = [issue.strip() for issue in review.series_issues if issue.strip()]
    summary = review.summary.strip() or "Rejected. See the slide issues."
    slides_ok = all(slide.approved for slide in slides) and len(slides) == slide_count
    if not review.approved and slides_ok and not series:
        series = [summary]
    approved = slides_ok and not series
    if approved:
        summary = review.summary.strip() or "Approved."
    return VisualReview(approved=approved, summary=summary, series_issues=series, slides=slides)
