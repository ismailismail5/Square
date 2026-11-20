"""Turns the icon source into every icon resource the app ships.

Run it after changing the artwork:

    python3 tools/icon.py

Source: app/icon-src/icona.png. Light and dark launchers get the same icon —
it is a dark plate with a bright mark, which is legible either way, and two
variants of one design would only drift apart.

Three things here are not a plain resize.

The artwork is sized by *the mark*, not by the image. An adaptive icon is 108
units wide, of which the launcher shows 72 and guarantees only the middle 66;
sizing by the image left the wave's ends outside that window and the launcher
cut them off.

The plate's own colour fills the rest of the canvas — sampled from the
artwork, and also written into colors.xml as the adaptive background. Whatever
mask the launcher applies then cuts into flat colour instead of into the mark.

The source's backdrop (a half-transparent grey checkerboard stand-in) is
removed, and the pixels it leaves behind are recoloured to the plate colour
before the edge is softened, so no grey fringe survives on the plate's edge.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = REPO_ROOT / "app/icon-src/icona.png"
DEFAULT_RES = REPO_ROOT / "app/src/main/res"

DENSITIES = {"mdpi": 1, "hdpi": 1.5, "xhdpi": 2, "xxhdpi": 3, "xxxhdpi": 4}

FOREGROUND_DP = 108
SAFE_DP = 66
LEGACY_DP = 48

OPAQUE_ALPHA = 200   # alpha at or above this counts as part of the plate
EDGE_BLUR = 0.8      # pixels of softening on the plate's edge
RING_RADIUS = 0.92   # where to sample the plate colour, as a fraction of its radius
RING_SAMPLES = 24
MARK_TOLERANCE = 40  # summed RGB distance from the plate that counts as "mark"
CIRCLE_SUPERSAMPLE = 4

COLOUR_NAME = "ic_launcher_background"


def plate_mask(img: Image.Image) -> Image.Image:
    """Hard mask of the plate: everything at or above full-ish opacity."""
    return img.getchannel("A").point(lambda v: 255 if v >= OPAQUE_ALPHA else 0)


def plate_colour(img: Image.Image, mask: Image.Image) -> tuple[int, int, int, int]:
    """The plate's colour where it meets the background.

    Read around a ring just inside the plate's edge, and taken as a median.
    Not a corner — the artwork is a circle on transparency, so the corners are
    empty — and not a single point either: the plate is shaded, so one sample
    picks up whatever the gradient happens to be doing there and the seam shows.
    The ring is placed from the plate's own bounds, so padding around the plate
    in the source does not throw it off.
    """
    bbox = mask.getbbox()
    if bbox is None:
        raise ValueError("The source has no opaque pixels; is it the right file?")

    left, top, right, bottom = bbox
    cx, cy = (left + right) / 2, (top + bottom) / 2
    radius = min(right - left, bottom - top) / 2 * RING_RADIUS

    pixels, mask_px = img.load(), mask.load()
    samples = []
    for step in range(RING_SAMPLES):
        angle = step / RING_SAMPLES * 2 * math.pi
        x = round(cx + radius * math.cos(angle))
        y = round(cy + radius * math.sin(angle))
        if 0 <= x < img.width and 0 <= y < img.height and mask_px[x, y]:
            samples.append(pixels[x, y][:3])

    if not samples:
        raise ValueError("Could not sample the plate colour; the plate is not where expected.")

    r, g, b = (sorted(channel)[len(channel) // 2] for channel in zip(*samples))
    return (r, g, b, 255)


def flatten_backdrop(img: Image.Image, mask: Image.Image, plate) -> Image.Image:
    """Drops the source's own backdrop, keeping the plate.

    Everything outside the plate is recoloured to the plate colour *before* the
    edge is softened, so the half-transparent pixels the blur creates blend
    towards the plate instead of towards the backdrop's grey.
    """
    solid = Image.new("RGBA", img.size, plate)
    out = Image.composite(img, solid, mask)
    out.putalpha(mask.filter(ImageFilter.GaussianBlur(EDGE_BLUR)))
    return out


def mark_bounds(img: Image.Image, plate) -> tuple[int, int, int, int]:
    """The wave's bounding box: what differs from the plate and is opaque."""
    r, g, b, a = img.split()

    distance = None
    for channel, target in zip((r, g, b), plate[:3]):
        diff = ImageChops.difference(channel, Image.new("L", img.size, target))
        distance = diff if distance is None else ImageChops.add(distance, diff)

    differs = distance.point(lambda v: 255 if v >= MARK_TOLERANCE else 0)
    opaque = a.point(lambda v: 255 if v >= OPAQUE_ALPHA else 0)

    bbox = ImageChops.multiply(differs, opaque).getbbox()
    if bbox is None:
        raise ValueError("No mark found on the plate; try lowering MARK_TOLERANCE.")
    return bbox


def circle_crop(img: Image.Image) -> Image.Image:
    """Crops to a circle with smooth edges (drawn large, then scaled down)."""
    big = img.width * CIRCLE_SUPERSAMPLE, img.height * CIRCLE_SUPERSAMPLE
    mask = Image.new("L", big, 0)
    ImageDraw.Draw(mask).ellipse((0, 0, big[0] - 1, big[1] - 1), fill=255)
    out = img.copy()
    out.putalpha(mask.resize(img.size, Image.LANCZOS))
    return out


def legacy_icon(src: Image.Image, plate, size: int) -> Image.Image:
    """The whole icon, for launchers older than adaptive icons.

    Composited onto the plate: those launchers do not mask, so transparency
    would show the home screen through the artwork's edges. Padded rather than
    stretched if the source is not square.
    """
    base = Image.new("RGBA", (size, size), plate)
    fitted = ImageOps.pad(src, (size, size), Image.LANCZOS, color=plate)
    base.paste(fitted, (0, 0), fitted)
    return base


def build(source: Path, res: Path) -> tuple[int, int, int, int]:
    raw = Image.open(source).convert("RGBA")
    mask = plate_mask(raw)
    plate = plate_colour(raw, mask)
    src = flatten_backdrop(raw, mask, plate)

    left, top, right, bottom = mark_bounds(src, plate)
    mark = max(right - left, bottom - top)
    mark_centre = ((left + right) / 2, (top + bottom) / 2)

    for density, density_scale in DENSITIES.items():
        canvas_px = round(FOREGROUND_DP * density_scale)
        safe_px = round(SAFE_DP * density_scale)

        scale = safe_px / mark
        art = src.resize(
            (max(1, round(src.width * scale)), max(1, round(src.height * scale))),
            Image.LANCZOS,
        )
        # Centred on the mark rather than on the image: the two need not agree.
        centre = (mark_centre[0] * scale, mark_centre[1] * scale)

        canvas = Image.new("RGBA", (canvas_px, canvas_px), plate)
        canvas.paste(
            art,
            (round(canvas_px / 2 - centre[0]), round(canvas_px / 2 - centre[1])),
            art,
        )

        out = res / f"mipmap-{density}"
        out.mkdir(parents=True, exist_ok=True)
        canvas.save(out / "ic_launcher_foreground.png", optimize=True)

        legacy = legacy_icon(src, plate, round(LEGACY_DP * density_scale))
        legacy.save(out / "ic_launcher.png", optimize=True)
        circle_crop(legacy).save(out / "ic_launcher_round.png", optimize=True)

    return plate


def write_background_colour(path: Path, colour) -> bool:
    """Sets the adaptive background colour, leaving the rest of the file alone.

    Returns True if the file changed.
    """
    value = f"#{colour[0]:02X}{colour[1]:02X}{colour[2]:02X}"
    entry = f'<color name="{COLOUR_NAME}">{value}</color>'
    existing = re.compile(rf'<color\s+name="{COLOUR_NAME}"\s*>[^<]*</color>')

    if path.exists():
        text = path.read_text(encoding="utf-8")
        if existing.search(text):
            new = existing.sub(lambda _: entry, text, count=1)
        elif "</resources>" in text:
            new = text.replace(
                "</resources>",
                "    <!-- Sampled from the icon's plate by tools/icon.py. -->\n"
                f"    {entry}\n</resources>",
            )
        else:
            raise ValueError(f"{path} has no </resources>; not touching it.")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        new = (
            '<?xml version="1.0" encoding="utf-8"?>\n'
            "<resources>\n"
            "    <!-- Sampled from the icon's plate by tools/icon.py, so the\n"
            "         adaptive icon's background and its artwork meet invisibly. -->\n"
            f"    {entry}\n"
            "</resources>\n"
        )
        text = None

    if new == text:
        return False
    path.write_text(new, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the app's icon resources.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--res", type=Path, default=DEFAULT_RES)
    args = parser.parse_args()

    if not args.source.is_file():
        print(f"Source image not found: {args.source}", file=sys.stderr)
        return 1

    plate = build(args.source, args.res)
    changed = write_background_colour(args.res / "values" / "colors.xml", plate)
    print(f"plate #{plate[0]:02X}{plate[1]:02X}{plate[2]:02X}",
          "(colors.xml updated)" if changed else "(colors.xml unchanged)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
