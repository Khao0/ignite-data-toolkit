"""Run the IGNITE H&E segmentation ensemble on one RGB image."""

import argparse
from pathlib import Path


def predict_mask(trainer, params, image):
    """Average fold probabilities and return an H x W uint8 class-ID mask."""
    import numpy as np

    if not params:
        raise ValueError('At least one model fold is required')
    # nnU-Net expects channels, depth (one plane), height, width.
    data = image.astype(np.float32).transpose(2, 0, 1)[:, None] / 255.0
    probability_sum = None
    for checkpoint in params:
        trainer.load_checkpoint_ram(checkpoint, False)
        probabilities = trainer.predict_preprocessed_data_return_seg_and_softmax(
            data, verbose=False, do_mirroring=False, mirror_axes=[],
        )[-1]
        if probability_sum is None:
            probability_sum = probabilities.astype(np.float32, copy=True)
        else:
            probability_sum += probabilities
    probability_sum /= len(params)
    return probability_sum.argmax(axis=0).squeeze(axis=0).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(
        description='Run IGNITE H&E segmentation on one image at 0.5 microns/pixel.',
    )
    parser.add_argument('--input', type=Path, required=True, help='Path to one H&E image')
    parser.add_argument('--output', type=Path, required=True, help='Output directory')
    parser.add_argument('--preview', action='store_true', help='Also save a colored segmentation PNG')
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f'Input image does not exist: {args.input}')

    import numpy as np
    from PIL import Image

    try:
        with Image.open(args.input) as image:
            pixels = np.array(image.convert('RGB'))
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
        parser.error(f'Cannot read input image {args.input}: {error}')

    args.output.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[2]
    model_path = repo_root / 'data/models/he/nnUNetTrainerV2_BN_pathology_DA_ignore0_hed005__nnUNet_RGB_scaleTo_0_1_bs8_ps512'

    from nnunet.training.model_restore import load_model_and_checkpoint_files

    print(f'Loading model: {model_path}')
    trainer, params = load_model_and_checkpoint_files(
        str(model_path), (0, 1, 2, 3, 4),
        mixed_precision=None, checkpoint_name='model_best',
    )
    mask = predict_mask(trainer, params, pixels)
    mask_path = args.output / f'{args.input.stem}.npz'
    np.savez_compressed(mask_path, mask=mask)
    print(f'Saved mask: {mask_path}')

    if args.preview:
        from label_colors import PALETTE

        preview_path = args.output / 'previews' / f'{args.input.stem}.png'
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.asarray(PALETTE, dtype=np.uint8)[mask]).save(preview_path)
        print(f'Saved preview: {preview_path}')


if __name__ == '__main__':
    main()
