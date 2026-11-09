"""Instagram Graph API: insights refresh and carousel publishing."""

from __future__ import annotations

import logging
import re
import time

import httpx

from insta_agent.config import Settings
from insta_agent.errors import PublishError
from insta_agent.models import PastPost

logger = logging.getLogger(__name__)

_INSIGHT_METRICS = ("reach", "saved", "shares", "views")
_TOKEN_PATTERN = re.compile(r"access_token=[^&\s]+")


class InstagramService:
    """Read performance and publish a carousel for the configured professional account."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = httpx.Client(timeout=httpx.Timeout(60.0, connect=20.0))

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> InstagramService:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def recent_posts(self, *, limit: int = 15) -> list[PastPost]:
        """Pull recent media and whatever insight metrics this account allows."""

        payload = self._get(
            f"{self._settings.instagram_account_id}/media",
            {
                "fields": "id,caption,media_type,timestamp,permalink,like_count,comments_count",
                "limit": str(limit),
            },
        )
        posts: list[PastPost] = []
        for item in payload.get("data", []):
            insights = self._insights(item["id"])
            posts.append(
                PastPost(
                    id=item["id"],
                    caption=item.get("caption") or "",
                    published_at=item.get("timestamp") or "",
                    permalink=item.get("permalink") or "",
                    media_type=item.get("media_type") or "",
                    like_count=int(item.get("like_count") or 0),
                    comments_count=int(item.get("comments_count") or 0),
                    reach=_as_int(insights.get("reach")),
                    saved=_as_int(insights.get("saved")),
                    shares=_as_int(insights.get("shares")),
                    views=_as_int(insights.get("views")),
                )
            )
        logger.info("Loaded %s recent Instagram posts.", len(posts))
        return posts

    def publish_carousel(self, *, image_urls: list[str], caption: str) -> PastPost:
        """Create child containers, a carousel container, and publish it."""

        if not 2 <= len(image_urls) <= 10:
            raise PublishError("Instagram carousels need between 2 and 10 images.")
        child_ids: list[str] = []
        for url in image_urls:
            created = self._post(
                f"{self._settings.instagram_account_id}/media",
                {"image_url": url, "is_carousel_item": "true"},
            )
            child_id = created["id"]
            self._wait_until_finished(child_id)
            child_ids.append(child_id)
            logger.info("Instagram prepared slide container %s.", child_id)
        parent = self._post(
            f"{self._settings.instagram_account_id}/media",
            {
                "media_type": "CAROUSEL",
                "children": ",".join(child_ids),
                "caption": caption,
            },
        )
        creation_id = parent["id"]
        self._wait_until_finished(creation_id)
        published = self._post(
            f"{self._settings.instagram_account_id}/media_publish",
            {"creation_id": creation_id},
        )
        media_id = published["id"]
        details = self._get(
            media_id,
            {"fields": "id,caption,media_type,timestamp,permalink,like_count,comments_count"},
        )
        logger.info("Published Instagram media %s.", media_id)
        return PastPost(
            id=details.get("id", media_id),
            caption=details.get("caption") or caption,
            published_at=details.get("timestamp") or "",
            permalink=details.get("permalink") or "",
            media_type=details.get("media_type") or "CAROUSEL_ALBUM",
            like_count=int(details.get("like_count") or 0),
            comments_count=int(details.get("comments_count") or 0),
        )

    def _insights(self, media_id: str) -> dict:
        found: dict[str, int] = {}
        for metric in _INSIGHT_METRICS:
            try:
                payload = self._get(f"{media_id}/insights", {"metric": metric})
            except PublishError as exc:
                logger.debug("Insight %s unavailable for %s: %s", metric, media_id, exc)
                continue
            for item in payload.get("data", []):
                value = _metric_value(item)
                if value is not None:
                    found[item.get("name", metric)] = value
        return found

    def _wait_until_finished(self, container_id: str) -> None:
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            payload = self._get(container_id, {"fields": "status_code"})
            status = payload.get("status_code")
            if status == "FINISHED":
                return
            if status in {"ERROR", "EXPIRED"}:
                raise PublishError(f"Instagram container {container_id} is {status}.")
            time.sleep(5)
        raise PublishError(
            f"Instagram did not finish container {container_id} within 5 minutes."
        )

    def _get(self, path: str, params: dict[str, str]) -> dict:
        return self._request("GET", path, params=params)

    def _post(self, path: str, data: dict[str, str]) -> dict:
        return self._request("POST", path, data=data)

    def _request(self, method: str, path: str, **kwargs) -> dict:
        params = dict(kwargs.pop("params", {}) or {})
        params["access_token"] = self._settings.instagram_access_token
        url = f"{self._settings.graph_api_base.rstrip('/')}/{path.lstrip('/')}"
        try:
            response = self._client.request(method, url, params=params, **kwargs)
        except httpx.HTTPError as exc:
            raise PublishError(f"Could not reach Instagram: {_redact(str(exc))}") from exc
        if response.status_code >= 400:
            raise PublishError(_error_message(response))
        try:
            return response.json()
        except ValueError as exc:
            raise PublishError("Instagram returned a response that was not JSON.") from exc


def _metric_value(item: dict) -> int | None:
    values = item.get("values") or []
    if values and isinstance(values[-1], dict) and "value" in values[-1]:
        return _as_int(values[-1]["value"])
    total = item.get("total_value")
    if isinstance(total, dict) and "value" in total:
        return _as_int(total["value"])
    if "value" in item:
        return _as_int(item["value"])
    return None


def _as_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _redact(text: str) -> str:
    return _TOKEN_PATTERN.sub("access_token=REDACTED", text)


def _error_message(response: httpx.Response) -> str:
    try:
        body = response.json()
        message = body.get("error", {}).get("message", response.text)
    except ValueError:
        message = response.text
    redacted = _redact(message)
    return f"Instagram request failed ({response.status_code}): {' '.join(redacted.split())[:400]}"
