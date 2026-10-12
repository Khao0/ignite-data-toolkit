
import argparse
import json
import math
from pathlib import Path

import openslide
from PIL import Image

base_dir = Path(__file__).resolve().parent
slide_path = base_dir / "slides" / "S19001216_A1_HE_R3.mrxs"
parser = argparse.ArgumentParser(description="Export an RGB slide crop at 0.5 microns/pixel for IGNITE.")
parser.add_argument("--slide", type=Path, default=slide_path)
parser.add_argument("--x", type=int, default=10000, help="Level-0 pixel coordinate")
parser.add_argument("--y", type=int, default=10000, help="Level-0 pixel coordinate")
parser.add_argument("--size", type=int, default=1024, help="Output width and height in pixels")
args = parser.parse_args()
slide_path = args.slide.resolve()
if args.size < 1 or args.x < 0 or args.y < 0:
    parser.error("Size must be positive and coordinates must be nonnegative")
output_dir = base_dir / "output"
output_dir.mkdir(parents=True, exist_ok=True)

if not slide_path.is_file():
    raise FileNotFoundError(f"Slide file not found: {slide_path}")

with openslide.OpenSlide(slide_path) as slide:
    print("Slide dimensions:", slide.dimensions)
    print("Resolution levels:", slide.level_dimensions)
    print("Microns/pixel X:", slide.properties.get("openslide.mpp-x"))
    print("Microns/pixel Y:", slide.properties.get("openslide.mpp-y"))

    # Overview for locating tissue; not an inference-resolution image.
    slide.get_thumbnail((2048, 2048)).save(output_dir / "overview.png")

    # Example region: change x and y to a tissue location.
    # Coordinates are measured in full-resolution slide pixels.
    x, y = args.x, args.y
    try:
        mpp_x = float(slide.properties["openslide.mpp-x"])
        mpp_y = float(slide.properties["openslide.mpp-y"])
    except (KeyError, ValueError):
        raise ValueError("Slide has no valid microns/pixel metadata; obtain the scanner resolution before exporting")
    if not all(math.isfinite(v) and v > 0 for v in (mpp_x, mpp_y)):
        raise ValueError("Microns/pixel values must be finite and positive")
    target_mpp = 0.5
    if max(mpp_x, mpp_y) > target_mpp:
        print("Warning: upsampling a lower-resolution slide cannot recover missing detail")
    extent_x = args.size * target_mpp / mpp_x
    extent_y = args.size * target_mpp / mpp_y
    if x + extent_x > slide.dimensions[0] or y + extent_y > slide.dimensions[1]:
        raise ValueError("Requested crop extends outside the slide; choose smaller coordinates or size")
    level = slide.get_best_level_for_downsample(min(target_mpp / mpp_x, target_mpp / mpp_y))
    scale = slide.level_downsamples[level]
    region = slide.read_region((x, y), level, (math.ceil(extent_x / scale), math.ceil(extent_y / scale)))

    # Composite transparent areas onto white, then save RGB PNG.
    image = Image.new("RGB", region.size, "white")
    image.paste(region, mask=region.getchannel("A"))
    image = image.resize((args.size, args.size), Image.Resampling.LANCZOS,
                         box=(0, 0, extent_x / scale, extent_y / scale))
    image.save(output_dir / "region.png")
    (output_dir / "region.json").write_text(json.dumps({
        "slide": str(slide_path), "location_level0": [x, y],
        "source_mpp": [mpp_x, mpp_y], "output_mpp": target_mpp,
        "output_size": [args.size, args.size], "read_level": level,
    }, indent=2))
    print(f"Saved {output_dir / 'region.png'} at {target_mpp} microns/pixel")
