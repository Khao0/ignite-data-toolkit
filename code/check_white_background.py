"""Count images containing only white background and report their percentage."""

import argparse
import csv
import sys
from pathlib import Path

from PIL import Image, ImageChops


IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'}


def white_percentage(path, threshold):
    """Check every pixel of the first frame, displaying transparency on white."""
    with Image.open(path) as image:
        rgba = image.convert('RGBA')
        background = Image.new('RGBA', rgba.size, 'white')
        rgb = Image.alpha_composite(background, rgba).convert('RGB')

    red, green, blue = rgb.split()
    # A pixel is white only when ALL three channels meet the threshold.
    minimum_channel = ImageChops.darker(ImageChops.darker(red, green), blue)
    histogram = minimum_channel.histogram()
    return 100.0 * sum(histogram[threshold:]) / (rgb.width * rgb.height)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path, help='Folder containing images')
    parser.add_argument('--recursive', action='store_true', help='Also check subfolders')
    parser.add_argument(
        '--threshold', type=int, default=255,
        help='Minimum value of each RGB channel for a white pixel (0-255; default: 255)',
    )
    parser.add_argument(
        '--min-white-percent', type=float, default=100.0,
        help='Minimum white pixel percentage to classify a file as white (default: 100)',
    )
    parser.add_argument('--csv', type=Path, help='Save per-image percentages to this CSV file')
    args = parser.parse_args()

    if not args.folder.is_dir():
        parser.error(f'Folder does not exist: {args.folder}')
    if not 0 <= args.threshold <= 255:
        parser.error('--threshold must be between 0 and 255')
    if not 0 <= args.min_white_percent <= 100:
        parser.error('--min-white-percent must be between 0 and 100')

    paths = args.folder.rglob('*') if args.recursive else args.folder.iterdir()
    image_paths = sorted(
        path for path in paths if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        print(f'No supported images found in {args.folder}', file=sys.stderr)
        return 1

    if args.csv and args.csv.suffix.lower() != '.csv':
        parser.error('--csv must have a .csv extension')

    checked = white_count = failed = 0
    report = None
    writer = None
    try:
        if args.csv:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            report = args.csv.open('w', newline='', encoding='utf-8')
            writer = csv.writer(report)
            writer.writerow(['file', 'white_percent', 'is_white_background', 'error'])

        for path in image_paths:
            relative_path = path.relative_to(args.folder).as_posix()
            try:
                percentage = white_percentage(path, args.threshold)
            except (OSError, ValueError, Image.DecompressionBombError) as error:
                failed += 1
                print(f'Could not read {path}: {error}', file=sys.stderr)
                if writer is not None:
                    writer.writerow([relative_path, '', '', str(error)])
                continue

            is_white = percentage >= args.min_white_percent
            checked += 1
            white_count += int(is_white)
            if writer is not None:
                writer.writerow([relative_path, f'{percentage:.6f}', is_white, ''])
    finally:
        if report is not None:
            report.close()

    print(f'Folder: {args.folder.resolve()}')
    print(f'White rule: RGB channels >= {args.threshold}; white pixels >= {args.min_white_percent:g}%')
    print(f'Image files found: {len(image_paths)}')
    print(f'Images checked: {checked}')
    print(f'Unreadable images: {failed}')
    print(f'White-background images: {white_count}')
    print(f'Other images: {checked - white_count}')
    if checked:
        print(f'White-background percentage: {100.0 * white_count / checked:.2f}%')
    else:
        print('White-background percentage: N/A (no readable images)')
    if args.csv:
        print(f'CSV report: {args.csv.resolve()}')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
