"""Put finished JPEGs where Instagram's servers can fetch them."""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

import httpx

from insta_agent.errors import PublishError

logger = logging.getLogger(__name__)


class ImageHost:
    """Uploads one JPEG and returns a public HTTPS URL."""

    def upload(self, path: Path) -> str:
        raise NotImplementedError

    def close(self) -> None:
        return None


class CatboxHost(ImageHost):
    """Anonymous upload to catbox.moe. No extra API key."""

    def __init__(self, run_id: str):
        self._run_id = run_id
        self._client = httpx.Client(
            timeout=httpx.Timeout(90.0, connect=20.0),
            follow_redirects=True,
            headers={"User-Agent": "insta-agent/0.1"},
        )

    def close(self) -> None:
        self._client.close()

    def upload(self, path: Path) -> str:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with path.open("rb") as handle:
                    response = self._client.post(
                        "https://catbox.moe/user/api.php",
                        data={"reqtype": "fileupload"},
                        files={
                            "fileToUpload": (
                                f"{self._run_id}_{path.name}",
                                handle,
                                "image/jpeg",
                            )
                        },
                    )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                last_error = exc
                time.sleep(2 * (attempt + 1))
                continue
            url = response.text.strip()
            if url.startswith("https://"):
                logger.info("Uploaded %s.", path.name)
                return url
            last_error = PublishError(url[:200] or "empty response")
            time.sleep(2 * (attempt + 1))
        raise PublishError(
            "Could not upload images to catbox.moe. Set IMAGE_HOST=public_dir and serve the "
            f"folder over HTTPS instead. Last error: {last_error}"
        )


class PublicDirHost(ImageHost):
    """Copy into a directory that is already published at a base URL."""

    def __init__(self, directory: Path, base_url: str, run_id: str):
        self._directory = directory
        self._base_url = base_url.rstrip("/")
        self._run_id = run_id
        self._directory.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        return None

    def upload(self, path: Path) -> str:
        if not self._base_url.startswith("https://"):
            raise PublishError("IMAGE_PUBLIC_BASE_URL must start with https:// so Instagram can fetch it.")
        dest_name = f"{self._run_id}_{path.name}"
        dest = self._directory / dest_name
        shutil.copyfile(path, dest)
        logger.info("Copied %s into %s.", path.name, dest)
        return f"{self._base_url}/{dest_name}"
