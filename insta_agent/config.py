"""Load settings from the environment and .env."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from insta_agent.errors import ConfigError

ROOT = Path(__file__).resolve().parents[1]
_HEX_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")
_PLACEHOLDER_MARKERS = ("placeholder", "changeme", "your-key", "your_key", "replace-me")


def is_placeholder(value: str) -> bool:
    """True when a credential or required setting has not been filled in."""

    lowered = value.strip().lower()
    if not lowered:
        return True
    return any(marker in lowered for marker in _PLACEHOLDER_MARKERS)


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer.") from exc


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw.strip())
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number.") from exc


@dataclass(frozen=True)
class Settings:
    """Runtime settings. Secrets stay on the instance and are never logged."""

    openrouter_api_key: str
    openrouter_api_base: str
    text_model: str
    vision_model: str
    image_model: str
    image_seed: int | None
    image_aspect_ratio: str
    image_resolution: str
    instagram_access_token: str
    instagram_account_id: str
    graph_api_base: str
    account_niche: str
    brand_name: str
    brand_voice: str
    accent_color: str
    carousel_slides: int
    post_interval_hours: float
    post_dry_run: bool
    max_qa_retries: int
    image_host: str
    image_public_dir: Path | None
    image_public_base_url: str
    output_dir: Path
    performance_path: Path

    @classmethod
    def load(cls) -> Settings:
        load_dotenv(ROOT / ".env")
        public_dir = _env("IMAGE_PUBLIC_DIR")
        seed_raw = _env("IMAGE_SEED")
        seed: int | None = None
        if seed_raw:
            try:
                seed = int(seed_raw)
            except ValueError as exc:
                raise ConfigError("IMAGE_SEED must be an integer.") from exc
        settings = cls(
            openrouter_api_key=_env("OPENROUTER_API_KEY"),
            openrouter_api_base=_env("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1"),
            text_model=_env("TEXT_MODEL", "openai/gpt-4o-mini"),
            vision_model=_env("VISION_MODEL", "openai/gpt-4o"),
            image_model=_env("IMAGE_MODEL", "black-forest-labs/flux.2-pro"),
            image_seed=seed,
            image_aspect_ratio=_env("IMAGE_ASPECT_RATIO", "4:5"),
            image_resolution=_env("IMAGE_RESOLUTION", "1K"),
            instagram_access_token=_env("INSTAGRAM_ACCESS_TOKEN"),
            instagram_account_id=_env("INSTAGRAM_ACCOUNT_ID"),
            graph_api_base=_env("GRAPH_API_BASE", "https://graph.facebook.com/v27.0"),
            account_niche=_env("ACCOUNT_NICHE"),
            brand_name=_env("BRAND_NAME", "Your Brand"),
            brand_voice=_env("BRAND_VOICE", "clear, specific, and warm"),
            accent_color=_env("ACCENT_COLOR", "#111111"),
            carousel_slides=_int("CAROUSEL_SLIDES", 5),
            post_interval_hours=_float("POST_INTERVAL_HOURS", 24),
            post_dry_run=_bool("POST_DRY_RUN", True),
            max_qa_retries=_int("MAX_QA_RETRIES", 2),
            image_host=_env("IMAGE_HOST", "catbox").lower(),
            image_public_dir=Path(public_dir) if public_dir else None,
            image_public_base_url=_env("IMAGE_PUBLIC_BASE_URL"),
            output_dir=Path(_env("OUTPUT_DIR") or str(ROOT / "output")),
            performance_path=Path(_env("PERFORMANCE_PATH") or str(ROOT / "data" / "performance.json")),
        )
        settings._validate_shape()
        return settings

    @property
    def instagram_ready(self) -> bool:
        return not is_placeholder(self.instagram_access_token) and not is_placeholder(
            self.instagram_account_id
        )

    def missing_for_run(self, *, dry_run: bool) -> list[str]:
        """Names of settings that would stop this run."""

        missing: list[str] = []
        if is_placeholder(self.openrouter_api_key):
            missing.append("OPENROUTER_API_KEY")
        if is_placeholder(self.account_niche):
            missing.append("ACCOUNT_NICHE")
        if not dry_run and is_placeholder(self.instagram_access_token):
            missing.append("INSTAGRAM_ACCESS_TOKEN")
        if not dry_run and is_placeholder(self.instagram_account_id):
            missing.append("INSTAGRAM_ACCOUNT_ID")
        if not dry_run and self.image_host == "public_dir":
            if self.image_public_dir is None:
                missing.append("IMAGE_PUBLIC_DIR")
            if is_placeholder(self.image_public_base_url):
                missing.append("IMAGE_PUBLIC_BASE_URL")
        return missing

    def validate_for_run(self, *, dry_run: bool) -> None:
        missing = self.missing_for_run(dry_run=dry_run)
        if missing:
            raise ConfigError(
                "Fill these in .env before running: " + ", ".join(missing) + ". See .env.example."
            )

    def _validate_shape(self) -> None:
        if not _HEX_COLOR.match(self.accent_color):
            raise ConfigError("ACCENT_COLOR must be a hex color like #112233.")
        if not 2 <= self.carousel_slides <= 10:
            raise ConfigError("CAROUSEL_SLIDES must be from 2 to 10, which is Instagram's carousel limit.")
        if self.post_interval_hours < 1:
            raise ConfigError("POST_INTERVAL_HOURS must be at least 1.")
        if self.max_qa_retries < 0:
            raise ConfigError("MAX_QA_RETRIES cannot be negative.")
        if self.image_host not in {"catbox", "public_dir"}:
            raise ConfigError("IMAGE_HOST must be catbox or public_dir.")
        if self.image_seed is not None and self.image_seed < 0:
            raise ConfigError("IMAGE_SEED must be a positive integer.")
        if self.image_aspect_ratio != "auto" and ":" not in self.image_aspect_ratio:
            raise ConfigError("IMAGE_ASPECT_RATIO must look like 4:5, or be auto.")
        if not self.image_resolution:
            raise ConfigError("IMAGE_RESOLUTION cannot be empty.")
        if not self.text_model or not self.vision_model or not self.image_model:
            raise ConfigError("TEXT_MODEL, VISION_MODEL, and IMAGE_MODEL must be set.")
        if not self.openrouter_api_base.startswith("https://"):
            raise ConfigError("OPENROUTER_API_BASE must start with https://.")
