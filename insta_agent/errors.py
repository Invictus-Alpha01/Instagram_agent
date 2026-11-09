"""Expected failures raised by the carousel pipeline."""

from __future__ import annotations

from pathlib import Path


class InstaAgentError(Exception):
    """An expected failure the operator can act on."""


class ConfigError(InstaAgentError):
    """Missing or invalid configuration."""


class AgentError(InstaAgentError):
    """A model call failed or returned something unusable."""


class PublishError(InstaAgentError):
    """Image hosting or Instagram publishing failed."""


class RejectedError(InstaAgentError):
    """The visual check refused the carousel, so nothing was published."""

    def __init__(self, summary: str, run_dir: Path):
        super().__init__(
            f"Visual check rejected the carousel ({run_dir}): {summary}"
        )
        self.run_dir = run_dir
        self.summary = summary
