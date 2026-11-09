"""Local history the analyst reads on every run."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from insta_agent.errors import ConfigError
from insta_agent.models import PastPost


class PerformanceStore:
    """JSON file of past posts. Instagram refreshes merge into the same records."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write({"posts": []})

    def load(self) -> list[PastPost]:
        with self._lock:
            return self._read()

    def upsert(self, incoming: list[PastPost]) -> list[PastPost]:
        """Insert or refresh posts by id. Local theme and slide copy are kept."""

        with self._lock:
            current = {post.id: post for post in self._read()}
            for post in incoming:
                existing = current.get(post.id)
                if existing is None:
                    current[post.id] = post
                    continue
                preserved = {
                    "theme": post.theme or existing.theme,
                    "slides": post.slides or existing.slides,
                    "caption": post.caption or existing.caption,
                    "permalink": post.permalink or existing.permalink,
                }
                for metric in ("reach", "saved", "shares", "views"):
                    if getattr(post, metric) is None:
                        preserved[metric] = getattr(existing, metric)
                current[post.id] = post.model_copy(update=preserved)
            posts = sorted(current.values(), key=lambda item: item.published_at, reverse=True)
            self._write({"posts": [post.model_dump() for post in posts]})
            return posts

    def _read(self) -> list[PastPost]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{self.path} is not valid JSON: {exc}") from exc
        if not isinstance(raw, dict) or not isinstance(raw.get("posts"), list):
            raise ConfigError(f"{self.path} must be an object with a posts array.")
        return [PastPost.model_validate(item) for item in raw["posts"]]

    def _write(self, payload: dict) -> None:
        text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(self.path)
