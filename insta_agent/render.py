"""Turn a generated photograph and the creator's lines into a 4:5 carousel slide."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from insta_agent.errors import ConfigError

CANVAS_WIDTH = 1080
CANVAS_HEIGHT = 1350
SIDE_MARGIN = 64
BOTTOM_MARGIN = 78

_BOLD_CANDIDATES = (
    Path(r"C:\Windows\Fonts\segoeuib.ttf"),
    Path(r"C:\Windows\Fonts\arialbd.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    Path("/Library/Fonts/Arial Bold.ttf"),
)
_REGULAR_CANDIDATES = (
    Path(r"C:\Windows\Fonts\segoeui.ttf"),
    Path(r"C:\Windows\Fonts\arial.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/Library/Fonts/Arial.ttf"),
)


@dataclass(frozen=True)
class RenderedSlide:
    """A finished JPEG-ready image and whether the copy actually fit."""

    image: Image.Image
    overflow: bool


def cover(image: Image.Image, width: int, height: int) -> Image.Image:
    """Scale and center-crop so the photo fills the canvas."""

    source = image.convert("RGB")
    scale = max(width / source.width, height / source.height)
    resized = source.resize(
        (round(source.width * scale), round(source.height * scale)),
        Image.Resampling.LANCZOS,
    )
    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def open_image(data: bytes) -> Image.Image:
    """Decode image bytes and force them to load before the buffer is dropped."""

    with Image.open(BytesIO(data)) as image:
        image.load()
        return image.copy()


def save_jpeg(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, format="JPEG", quality=90, optimize=True)


def render_slide(
    background: Image.Image,
    *,
    headline: str,
    body: str,
    brand: str,
    accent_hex: str,
    index: int,
    total: int,
) -> RenderedSlide:
    """Composite brand, index, headline, and body over a darkened photograph."""

    base = cover(background, CANVAS_WIDTH, CANVAS_HEIGHT).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    shade = ImageDraw.Draw(overlay)
    _paint_scrims(shade)
    accent = _hex_to_rgb(accent_hex)
    shade.rectangle((0, 0, 14, CANVAS_HEIGHT), fill=(*accent, 255))
    composed = Image.alpha_composite(base, overlay)
    draw = ImageDraw.Draw(composed)

    overflow = not " ".join(headline.split())
    brand_font, brand_lines, _brand_overflow = _fit_text(
        draw, brand, CANVAS_WIDTH - 280, max_lines=1, start_size=30, min_size=20, bold=True
    )
    if brand_lines:
        draw.text((SIDE_MARGIN, 58), brand_lines[0], font=brand_font, fill=(255, 255, 255))

    index_font = _font(bold=True, size=26)
    index_label = f"{index:02d}  /  {total:02d}"
    index_width = draw.textlength(index_label, font=index_font)
    draw.text(
        (CANVAS_WIDTH - SIDE_MARGIN - index_width, 60),
        index_label,
        font=index_font,
        fill=(255, 255, 255),
    )

    text_width = CANVAS_WIDTH - SIDE_MARGIN * 2
    headline_font, headline_lines, headline_overflow = _fit_text(
        draw, headline, text_width, max_lines=3, start_size=68, min_size=40, bold=True
    )
    body_font, body_lines, body_overflow = _fit_text(
        draw, body, text_width, max_lines=4, start_size=34, min_size=26, bold=False
    )
    overflow = overflow or headline_overflow or body_overflow

    headline_height = _block_height(headline_lines, headline_font, 6)
    body_height = _block_height(body_lines, body_font, 6)
    gap = 20 if body_lines else 0
    block_height = headline_height + gap + body_height
    top = CANVAS_HEIGHT - BOTTOM_MARGIN - block_height
    if top < 240:
        overflow = True
    cursor = _draw_lines(draw, headline_lines, headline_font, SIDE_MARGIN, top, (255, 255, 255), 6)
    if body_lines:
        _draw_lines(draw, body_lines, body_font, SIDE_MARGIN, cursor + gap, (255, 255, 255), 6)

    return RenderedSlide(image=composed.convert("RGB"), overflow=overflow)


def _paint_scrims(draw: ImageDraw.ImageDraw) -> None:
    gradient_start = int(CANVAS_HEIGHT * 0.42)
    span = CANVAS_HEIGHT - gradient_start
    for y in range(gradient_start, CANVAS_HEIGHT):
        progress = (y - gradient_start) / span
        alpha = int(235 * (progress ** 1.05))
        draw.line([(0, y), (CANVAS_WIDTH, y)], fill=(0, 0, 0, alpha))
    for y in range(160):
        alpha = int(140 * (1 - y / 160))
        draw.line([(0, y), (CANVAS_WIDTH, y)], fill=(0, 0, 0, alpha))


def _fit_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    *,
    max_lines: int,
    start_size: int,
    min_size: int,
    bold: bool,
) -> tuple[ImageFont.FreeTypeFont, list[str], bool]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return _font(bold=bold, size=min_size), [], False
    for size in range(start_size, min_size - 1, -2):
        font = _font(bold=bold, size=size)
        lines = _wrap(draw, cleaned, font, max_width)
        if len(lines) <= max_lines:
            return font, lines, False
    font = _font(bold=bold, size=min_size)
    lines = _wrap(draw, cleaned, font, max_width)
    return font, lines[:max_lines], True


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        if draw.textlength(word, font=font) > max_width:
            if current:
                lines.append(current)
                current = ""
            chunk = ""
            for char in word:
                trial = chunk + char
                if draw.textlength(trial, font=font) <= max_width:
                    chunk = trial
                else:
                    if chunk:
                        lines.append(chunk)
                    chunk = char
            current = chunk
            continue
        trial = word if not current else f"{current} {word}"
        if draw.textlength(trial, font=font) <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _draw_lines(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    font: ImageFont.ImageFont,
    x: int,
    y: int,
    fill: tuple[int, int, int],
    spacing: int,
) -> int:
    line_height = _line_height(font)
    for line in lines:
        draw.text((x + 2, y + 3), line, font=font, fill=(0, 0, 0))
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height + spacing
    return y


def _block_height(lines: list[str], font: ImageFont.ImageFont, spacing: int) -> int:
    if not lines:
        return 0
    return len(lines) * _line_height(font) + (len(lines) - 1) * spacing


def _line_height(font: ImageFont.ImageFont) -> int:
    bbox = font.getbbox("Ag")
    return bbox[3] - bbox[1]


def _font(bold: bool, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(_font_file(bold), size=size)


@lru_cache(maxsize=2)
def _font_file(bold: bool) -> str:
    candidates = _BOLD_CANDIDATES if bold else _REGULAR_CANDIDATES
    for path in candidates:
        if path.is_file():
            return str(path)
    fallback = _REGULAR_CANDIDATES if bold else _BOLD_CANDIDATES
    for path in fallback:
        if path.is_file():
            return str(path)
    raise ConfigError("No TrueType font found. Install Segoe UI, Arial, or DejaVu Sans.")


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    hex_value = value.removeprefix("#")
    return int(hex_value[0:2], 16), int(hex_value[2:4], 16), int(hex_value[4:6], 16)
