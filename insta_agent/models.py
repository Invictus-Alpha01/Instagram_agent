"""Data passed between the three agents and the local performance store."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SlideRecord(BaseModel):
    """Copy that was actually published on one slide."""

    model_config = ConfigDict(extra="ignore")

    headline: str
    body: str = ""


class PastPost(BaseModel):
    """One historical post, merged from Instagram and the local store."""

    model_config = ConfigDict(extra="ignore")

    id: str
    caption: str = ""
    published_at: str = ""
    permalink: str = ""
    media_type: str = ""
    like_count: int = 0
    comments_count: int = 0
    reach: int | None = None
    saved: int | None = None
    shares: int | None = None
    views: int | None = None
    theme: str = ""
    slides: list[SlideRecord] = Field(default_factory=list)

    @property
    def interaction_weight(self) -> int:
        """A single comparable score. Saves and shares count more than likes."""

        return (
            self.like_count
            + self.comments_count * 3
            + (self.saved or 0) * 5
            + (self.shares or 0) * 4
        )


class SlideOutline(BaseModel):
    """One content line chosen by the analyst."""

    role: str = Field(description="hook for slide 1, value for middle slides, cta for the last slide")
    line: str = Field(description="The specific claim or step this slide must teach. One sentence.")


class ContentBrief(BaseModel):
    """What the analyst decided the next carousel should say."""

    theme: str = Field(description="Short name for this carousel, under 8 words")
    rationale: str = Field(description="Why this angle should outperform recent posts, based on the data")
    tone: str = Field(description="How the copy should sound for this post")
    slides: list[SlideOutline]
    caption_direction: str = Field(description="What the caption should add beyond the slides")
    avoid: list[str] = Field(description="Topics, formats, or phrases to stay away from")


class SlideCopy(BaseModel):
    """Finished words and the scene prompt for one slide."""

    headline: str = Field(description="At most 8 words. This is drawn on the image.")
    body: str = Field(description="At most 18 words. One supporting sentence. May be empty.")
    image_prompt: str = Field(description="A single photograph that supports the line. No text in the scene.")
    alt_text: str = Field(description="One sentence describing the photograph for someone who cannot see it")


class CarouselDraft(BaseModel):
    """The creator's finished carousel, before images exist."""

    caption: str = Field(description="Instagram caption, plain text, hashtags only at the end")
    visual_style: str = Field(description="One sentence locking lighting, lens, color, and era for every slide")
    slides: list[SlideCopy]


class SlideReview(BaseModel):
    """The visual editor's verdict on one rendered slide."""

    index: int = Field(description="1-based slide number")
    approved: bool
    issues: list[str] = Field(description="Specific problems. Empty only when approved is true.")


class VisualReview(BaseModel):
    """The visual editor's verdict on the whole carousel."""

    approved: bool
    summary: str
    series_issues: list[str] = Field(
        description="Problems that apply to the set, such as an inconsistent look. Empty when there are none."
    )
    slides: list[SlideReview]
