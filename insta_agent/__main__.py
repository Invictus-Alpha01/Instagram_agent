"""Command line for a periodic Instagram carousel."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler

from insta_agent.config import Settings, is_placeholder
from insta_agent.errors import ConfigError, InstaAgentError, RejectedError
from insta_agent.pipeline import publish_existing, refresh_history, run_cycle

logger = logging.getLogger("insta_agent")


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    try:
        settings = Settings.load()
        if args.command == "check":
            return _check(settings)
        if args.command == "refresh":
            refresh_history(settings)
            return 0
        if args.command == "run":
            run_cycle(settings, dry_run=_dry_run(settings, args))
            return 0
        if args.command == "schedule":
            _schedule(settings, dry_run=_dry_run(settings, args))
            return 0
        if args.command == "publish":
            run_dir = Path(args.run_dir)
            if not run_dir.is_absolute():
                run_dir = Path.cwd() / run_dir
            publish_existing(settings, run_dir)
            return 0
    except RejectedError as exc:
        logger.error("%s", exc)
        return 2
    except InstaAgentError as exc:
        logger.error("%s", exc)
        return 1
    return 0


def _schedule(settings: Settings, *, dry_run: bool) -> None:
    mode = "dry-run" if dry_run else "live publishing"
    logger.info(
        "Scheduling a carousel every %s hour(s) (%s). The first run starts now. Press Ctrl+C to stop.",
        settings.post_interval_hours,
        mode,
    )
    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        _safe_cycle,
        "interval",
        hours=settings.post_interval_hours,
        next_run_time=datetime.now(timezone.utc),
        kwargs={"settings": settings, "dry_run": dry_run},
        max_instances=1,
        coalesce=True,
        id="carousel",
    )
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped.")


def _safe_cycle(settings: Settings, dry_run: bool) -> None:
    try:
        run_cycle(settings, dry_run=dry_run)
    except InstaAgentError as exc:
        logger.error("%s", exc)
    except Exception:
        logger.exception("Carousel run failed.")


def _dry_run(settings: Settings, args: argparse.Namespace) -> bool:
    forced_dry = bool(getattr(args, "dry_run", False))
    forced_live = bool(getattr(args, "publish", False))
    if forced_dry and forced_live:
        raise ConfigError("Use either --dry-run or --publish, not both.")
    if forced_dry:
        return True
    if forced_live:
        return False
    return settings.post_dry_run


def _check(settings: Settings) -> int:
    dry_run = settings.post_dry_run
    rows = [
        ("OPENAI_API_KEY", settings.openai_api_key),
        ("FLUX_API_KEY", settings.flux_api_key),
        ("INSTAGRAM_ACCESS_TOKEN", settings.instagram_access_token),
        ("INSTAGRAM_ACCOUNT_ID", settings.instagram_account_id),
        ("ACCOUNT_NICHE", settings.account_niche),
    ]
    for name, value in rows:
        state = "placeholder" if is_placeholder(value) else "set"
        print(f"{name}: {state}")
    print(f"text model: {settings.openai_text_model}")
    print(f"vision model: {settings.openai_vision_model}")
    print(f"flux model: {settings.flux_model}")
    print(f"slides: {settings.carousel_slides}")
    print(f"interval hours: {settings.post_interval_hours}")
    print(f"dry run: {dry_run}")
    print(f"image host: {settings.image_host}")
    print(f"brand: {settings.brand_name}")
    print(f"output: {settings.output_dir}")
    missing = settings.missing_for_run(dry_run=dry_run)
    if missing:
        print("Not ready. Fill in: " + ", ".join(missing))
        return 1
    print("Ready.")
    return 0


def _configure_logging(verbose: bool) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.INFO)


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--verbose", action="store_true", help="Log prompts and API status.")
    parser = argparse.ArgumentParser(
        prog="python -m insta_agent",
        description="Create and publish an Instagram carousel on a schedule.",
        parents=[common],
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("check", parents=[common], help="Show which settings are still placeholders.")
    commands.add_parser("refresh", parents=[common], help="Download recent post performance from Instagram.")

    run = commands.add_parser("run", parents=[common], help="Create one carousel now.")
    _add_mode_flags(run)
    schedule = commands.add_parser("schedule", parents=[common], help="Create a carousel on POST_INTERVAL_HOURS.")
    _add_mode_flags(schedule)

    publish = commands.add_parser(
        "publish",
        parents=[common],
        help="Publish an approved run that was saved locally.",
    )
    publish.add_argument("run_dir", help="Path to a folder under output/, for example output/20261109T120000Z")
    return parser


def _add_mode_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dry-run", action="store_true", help="Generate and check, but do not publish.")
    parser.add_argument("--publish", action="store_true", help="Publish even when POST_DRY_RUN=true.")


if __name__ == "__main__":
    raise SystemExit(main())
