"""Run the analyst, creator, and visual editor, then publish when the editor approves."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from insta_agent.agents.analyst import Analyst
from insta_agent.agents.creator import Creator
from insta_agent.agents.visual_qa import VisualQA
from insta_agent.config import Settings
from insta_agent.errors import AgentError, PublishError, RejectedError
from insta_agent.models import CarouselDraft, PastPost, SlideRecord, SlideReview, VisualReview
from insta_agent.render import open_image, render_slide, save_jpeg
from insta_agent.services.image_host import CatboxHost, ImageHost, PublicDirHost
from insta_agent.services.instagram_service import InstagramService
from insta_agent.services.openrouter_service import OpenRouterService
from insta_agent.storage.performance import PerformanceStore

logger = logging.getLogger(__name__)

_FATAL_IMAGE = (
    "Could not reach OpenRouter",
    "OpenRouter image request failed",
    "OpenRouter returned no image",
    "OpenRouter image was larger than 15MB",
)


def run_cycle(settings: Settings, *, dry_run: bool) -> Path:
    """Create one carousel. Publish it only after the visual editor approves."""

    settings.validate_for_run(dry_run=dry_run)
    if dry_run:
        logger.info("Dry run: the carousel will be checked and saved, then left unpublished.")
    store = PerformanceStore(settings.performance_path)
    openrouter = OpenRouterService(settings)
    instagram = InstagramService(settings) if settings.instagram_ready else None
    try:
        history = _refresh_history(store, instagram)
        analyst = Analyst(openrouter, settings)
        creator = Creator(openrouter, settings)
        visual_qa = VisualQA(openrouter, settings)
        brief = analyst.decide(history)
        draft = creator.draft(brief)
        run_dir = _new_run_dir(settings)
        backgrounds: dict[int, Image.Image] = {}
        prompts: dict[int, str] = {}
        review: VisualReview | None = None
        slide_paths: list[Path] = []
        for attempt in range(settings.max_qa_retries + 1):
            slide_paths, failures = _materialize(
                settings, openrouter, run_dir, draft, backgrounds, prompts
            )
            if failures:
                review = _local_rejection(draft, failures)
                logger.info("Skipping the visual editor until every slide can be rendered.")
            else:
                review = visual_qa.review(brief, draft, slide_paths)
            _write_package(run_dir, settings, brief, draft, review, slide_paths, status="generating")
            if review.approved:
                break
            for slide in review.slides:
                if not slide.approved:
                    logger.info("Slide %s needs a change: %s", slide.index, "; ".join(slide.issues))
            for issue in review.series_issues:
                logger.info("Series needs a change: %s", issue)
            if attempt == settings.max_qa_retries:
                _write_package(run_dir, settings, brief, draft, review, slide_paths, status="rejected")
                raise RejectedError(review.summary, run_dir)
            draft = creator.revise(brief, draft, review)
        if review is None or not review.approved:
            raise RejectedError("The visual editor did not approve this carousel.", run_dir)
        _write_package(run_dir, settings, brief, draft, review, slide_paths, status="ready")
        logger.info("Carousel is ready at %s", run_dir)
        if dry_run:
            return run_dir
        _publish_directory(settings, store, instagram, run_dir)
        return run_dir
    finally:
        openrouter.close()
        if instagram is not None:
            instagram.close()


def publish_existing(settings: Settings, run_dir: Path) -> str:
    """Publish a carousel the visual editor already approved."""

    settings.validate_for_run(dry_run=False)
    store = PerformanceStore(settings.performance_path)
    with InstagramService(settings) as instagram:
        return _publish_directory(settings, store, instagram, run_dir)


def refresh_history(settings: Settings) -> int:
    """Pull recent Instagram posts into the local performance file."""

    if not settings.instagram_ready:
        raise PublishError("Set INSTAGRAM_ACCESS_TOKEN and INSTAGRAM_ACCOUNT_ID before refreshing.")
    store = PerformanceStore(settings.performance_path)
    with InstagramService(settings) as instagram:
        posts = store.upsert(instagram.recent_posts())
    logger.info("Saved %s posts to %s.", len(posts), settings.performance_path)
    return len(posts)


def _refresh_history(store: PerformanceStore, instagram: InstagramService | None) -> list[PastPost]:
    if instagram is None:
        posts = store.load()
        logger.info("Instagram keys are placeholders, so the analyst is using %s saved posts.", len(posts))
        return posts
    try:
        return store.upsert(instagram.recent_posts())
    except PublishError as exc:
        logger.warning("Could not refresh Instagram insights. Using saved history. %s", exc)
        return store.load()


def _materialize(
    settings: Settings,
    openrouter: OpenRouterService,
    run_dir: Path,
    draft: CarouselDraft,
    backgrounds: dict[int, Image.Image],
    prompts: dict[int, str],
) -> tuple[list[Path], dict[int, str]]:
    paths: list[Path] = []
    failures: dict[int, str] = {}
    total = len(draft.slides)
    for index, slide in enumerate(draft.slides, start=1):
        prompt = _scene_prompt(slide.image_prompt, draft.visual_style)
        if prompts.get(index) != prompt:
            logger.info("Generating photograph %s of %s.", index, total)
            try:
                data = openrouter.generate_image(prompt, slide_index=index)
                backgrounds[index] = open_image(data)
            except AgentError as exc:
                message = str(exc)
                if message.startswith(_FATAL_IMAGE):
                    raise
                failures[index] = message
                continue
            except (UnidentifiedImageError, OSError) as exc:
                failures[index] = f"OpenRouter returned an unreadable image: {exc}"
                continue
            prompts[index] = prompt
            save_jpeg(backgrounds[index], run_dir / f"background_{index:02d}.jpg")
        else:
            logger.info("Reusing photograph %s of %s.", index, total)
        if index not in backgrounds:
            failures.setdefault(index, "No photograph is available for this slide.")
            continue
        rendered = render_slide(
            backgrounds[index],
            headline=slide.headline,
            body=slide.body,
            brand=settings.brand_name,
            accent_hex=settings.accent_color,
            index=index,
            total=total,
        )
        path = run_dir / f"slide_{index:02d}.jpg"
        save_jpeg(rendered.image, path)
        paths.append(path)
        if rendered.overflow:
            failures[index] = "Headline or body does not fit the slide. Shorten the copy."
    return paths, failures


def _scene_prompt(image_prompt: str, visual_style: str) -> str:
    return (
        f"{image_prompt.strip()} Series style: {visual_style.strip()} "
        "One vertical photograph, natural detail, no text, no letters, no numbers, "
        "no watermark, no logo, no border, no collage."
    )


def _local_rejection(draft: CarouselDraft, failures: dict[int, str]) -> VisualReview:
    slides = []
    for index in range(1, len(draft.slides) + 1):
        issue = failures.get(index)
        slides.append(
            SlideReview(
                index=index,
                approved=issue is None,
                issues=[] if issue is None else [issue],
            )
        )
    named = ", ".join(str(index) for index in sorted(failures))
    return VisualReview(
        approved=False,
        summary=f"Slides {named} could not be rendered, so the visual editor has not approved them.",
        series_issues=[],
        slides=slides,
    )


def _publish_directory(
    settings: Settings,
    store: PerformanceStore,
    instagram: InstagramService | None,
    run_dir: Path,
) -> str:
    if instagram is None:
        raise PublishError("Instagram credentials are missing, so this carousel was not published.")
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise PublishError(f"{run_dir} has no manifest.json.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") == "published":
        raise PublishError(f"{run_dir} is already published as {manifest.get('instagram_media_id')}.")
    if not manifest.get("qa_approved"):
        raise PublishError("Refusing to publish a carousel the visual editor did not approve.")
    run_id = str(manifest.get("id") or run_dir.name)
    caption = (run_dir / "caption.txt").read_text(encoding="utf-8").strip()
    if not caption:
        raise PublishError("caption.txt is empty.")
    slide_rows = manifest.get("slides") or []
    if len(slide_rows) < 2:
        raise PublishError("An approved carousel manifest needs at least 2 slides.")
    paths: list[Path] = []
    for row in slide_rows:
        name = row.get("file")
        if not name:
            raise PublishError("manifest.json is missing a slide filename.")
        path = run_dir / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row.get("sha256"):
            raise PublishError(
                f"{path.name} no longer matches the file the visual editor approved. Start a new run."
            )
        paths.append(path)
    host = _build_host(settings, run_id)
    try:
        urls = [host.upload(path) for path in paths]
    finally:
        host.close()
    published = instagram.publish_carousel(image_urls=urls, caption=caption)
    published_at = published.published_at or datetime.now(timezone.utc).isoformat()
    record = published.model_copy(
        update={
            "published_at": published_at,
            "theme": manifest.get("theme") or "",
            "slides": [
                SlideRecord(headline=row.get("headline", ""), body=row.get("body", ""))
                for row in slide_rows
            ],
        }
    )
    store.upsert([record])
    manifest["status"] = "published"
    manifest["instagram_media_id"] = record.id
    manifest["permalink"] = record.permalink
    _write_json(manifest_path, manifest)
    logger.info("Published %s", record.permalink or record.id)
    return record.id


def _build_host(settings: Settings, run_id: str) -> ImageHost:
    if settings.image_host == "public_dir":
        if settings.image_public_dir is None or not settings.image_public_base_url:
            raise PublishError("IMAGE_HOST=public_dir requires IMAGE_PUBLIC_DIR and IMAGE_PUBLIC_BASE_URL.")
        return PublicDirHost(settings.image_public_dir, settings.image_public_base_url, run_id)
    return CatboxHost(run_id)


def _new_run_dir(settings: Settings) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = settings.output_dir / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_package(
    run_dir: Path,
    settings: Settings,
    brief,
    draft: CarouselDraft,
    review: VisualReview,
    slide_paths: list[Path],
    *,
    status: str,
) -> None:
    _write_json(run_dir / "brief.json", brief.model_dump())
    _write_json(run_dir / "draft.json", draft.model_dump())
    _write_json(run_dir / "qa.json", review.model_dump())
    (run_dir / "caption.txt").write_text(draft.caption.strip() + "\n", encoding="utf-8")
    by_name = {path.name: path for path in slide_paths}
    slides = []
    for index, slide in enumerate(draft.slides, start=1):
        name = f"slide_{index:02d}.jpg"
        path = by_name.get(name)
        if path is None or not path.is_file():
            continue
        slides.append(
            {
                "file": name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "headline": slide.headline,
                "body": slide.body,
                "alt_text": slide.alt_text,
            }
        )
    manifest = {
        "id": run_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "theme": brief.theme,
        "qa_approved": review.approved,
        "instagram_media_id": None,
        "permalink": "",
        "models": {
            "text": settings.text_model,
            "vision": settings.vision_model,
            "image": settings.image_model,
        },
        "summary": review.summary,
        "slides": slides,
    }
    _write_json(run_dir / "manifest.json", manifest)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
