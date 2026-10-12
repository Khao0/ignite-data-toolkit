"""Export non-overlapping tissue crops for IGNITE H&E inference."""
import argparse
import json
import math
from pathlib import Path

import openslide
from PIL import Image, ImageDraw

base = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--slide', type=Path, default=base / 'slides/S19001216_A1_HE_R3.mrxs')
parser.add_argument('--output', type=Path, default=base / 'output/batch')
parser.add_argument('--size', type=int, default=1024)
parser.add_argument('--min-tissue', type=float, default=0.15)
args = parser.parse_args()
if args.size < 1 or not 0 <= args.min_tissue <= 1:
    parser.error('Size must be positive; min-tissue must be between 0 and 1')
patch_dir = args.output / 'patches'
patch_dir.mkdir(parents=True, exist_ok=True)


def rgb_on_white(region):
    rgb = Image.new('RGB', region.size, 'white')
    rgb.paste(region, mask=region.getchannel('A'))
    return rgb


def tissue_fraction(image):
    # Simple screening heuristic; inspect the contact sheet for artifacts.
    pixels = list(image.resize((64, 64)).getdata())
    return sum(max(p) - min(p) > 15 and 40 < sum(p) / 3 < 235
               for p in pixels) / len(pixels)


with openslide.OpenSlide(str(args.slide.resolve())) as slide:
    mpp = [float(slide.properties[k]) for k in ('openslide.mpp-x', 'openslide.mpp-y')]
    if not all(math.isfinite(v) and v > 0 for v in mpp):
        raise ValueError('Valid scanner microns/pixel metadata is required')
    target = 0.5
    extent = [args.size * target / v for v in mpp]
    level = slide.get_best_level_for_downsample(min(target / v for v in mpp))
    scale = slide.level_downsamples[level]
    read_size = tuple(math.ceil(v / scale) for v in extent)
    box = (0, 0, extent[0] / scale, extent[1] / scale)
    records, previews = [], []
    # Only complete crops: avoid introducing padded edges into inference.
    for row in range(math.floor(slide.dimensions[1] / extent[1])):
        for col in range(math.floor(slide.dimensions[0] / extent[0])):
            x, y = round(col * extent[0]), round(row * extent[1])
            image = rgb_on_white(slide.read_region((x, y), level, read_size))
            fraction = tissue_fraction(image)
            if fraction < args.min_tissue:
                continue
            image = image.resize((args.size, args.size), Image.Resampling.LANCZOS, box=box)
            name = f'patch_x{x}_y{y}.png'
            image.save(patch_dir / name)
            records.append({'file': name, 'location_level0': [x, y], 'tissue_fraction': fraction})
            preview = image.resize((160, 160))
            previews.append((name, preview))
    if not records:
        raise ValueError('No tissue patches found; inspect the slide or lower --min-tissue')
    # Paginate so each sheet remains legible.
    for start in range(0, len(previews), 48):
        page = previews[start:start + 48]
        sheet = Image.new('RGB', (6 * 180, math.ceil(len(page) / 6) * 190), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, (name, preview) in enumerate(page):
            x, y = (i % 6) * 180, (i // 6) * 190
            sheet.paste(preview, (x, y))
            draw.text((x, y + 162), name.removesuffix('.png'), fill='black')
        sheet.save(args.output / f'contact_sheet_{start // 48 + 1:02d}.png')
    (args.output / 'manifest.json').write_text(json.dumps({
        'slide': str(args.slide.resolve()), 'source_mpp': mpp, 'output_mpp': target,
        'size': args.size, 'patches': records,
    }, indent=2))
    print(f'Exported {len(records)} RGB tissue patches at {target} microns/pixel to {patch_dir}')
