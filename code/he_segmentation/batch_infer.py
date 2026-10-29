import argparse
import os
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
from nnunet.training.model_restore import load_model_and_checkpoint_files
from label_colors import PALETTE

print('RUNNING INFERENCE SCRIPT')
parser = argparse.ArgumentParser(description="Run IGNITE H&E segmentation on your RGB PNGs at 0.5 microns/pixel.")
parser.add_argument('--input', type=Path, required=True, help='Your H&E PNG file or folder of PNG crops, searched recursively (exclude thumbnails)')
parser.add_argument('--output', type=Path, help='Output compressed NumPy (.npz) mask folder; preserves input subfolders')
parser.add_argument('--preview-output', type=Path, help='Colored PNG folder (default: <output>/previews)')
parser.add_argument('--start-folder', help='Resume at this immediate input subfolder, inclusive, in alphabetical order')
parser.add_argument('--folder-suffix', help='Only process immediate input subfolders ending with this suffix')
parser.add_argument('--batch-size', type=int, default=32, help='Images per group sharing each fold load (default: 4; uses more CPU RAM for larger groups)')
args = parser.parse_args()
if args.batch_size < 1:
    parser.error('--batch-size must be positive')
if (args.start_folder or args.folder_suffix) and not args.input.is_dir():
    parser.error('--start-folder and --folder-suffix require --input to be a directory')
if args.start_folder and not (args.input / args.start_folder).is_dir():
    parser.error(f'Start folder does not exist: {args.input / args.start_folder}')
repo_root = Path(__file__).resolve().parents[2]
output_dir = args.output or repo_root / 'data/inference/he'
output_dir.mkdir(parents=True, exist_ok=True)
preview_dir = args.preview_output or output_dir / 'previews'
preview_dir.mkdir(parents=True, exist_ok=True)


##################################################
### Import model
##################################################

def norm_01(x_batch): # Use this for models trained on 0-1 scaled data
    x_batch = x_batch / 255
    x_batch = x_batch.transpose(3, 0, 1, 2)
    return x_batch

def ensemble_softmax_batch(trainer, params, images):
    """Load each fold once per group; predict individual images with nnU-Net."""
    if not params:
        raise ValueError('At least one fold is required')
    softmax_sums = [None] * len(images)
    for p in params: # params are the weight from fold 0,1, ...
        trainer.load_checkpoint_ram(p, False)
        for index, image in enumerate(images):
            softmax = trainer.predict_preprocessed_data_return_seg_and_softmax(
                image, verbose=False, do_mirroring=False, mirror_axes=[],
            )[-1].transpose(1, 2, 3, 0).squeeze()
            if softmax_sums[index] is None:
                softmax_sums[index] = softmax.copy()
            else:
                softmax_sums[index] += softmax
    for softmax in softmax_sums:
        softmax /= len(params)
    return softmax_sums


def iter_image_folders(input_path, start_folder=None, folder_suffix=None):
    """Discover PNG paths one directory at a time, pruning excluded subjects."""
    if input_path.is_file():
        yield input_path.parent, [input_path]
        return

    def raise_walk_error(error):
        raise error

    for directory, subfolders, filenames in os.walk(input_path, onerror=raise_walk_error):
        folder = Path(directory)
        subfolders.sort()
        if folder == input_path:
            subfolders[:] = [
                name for name in subfolders
                if (not start_folder or name >= start_folder)
                and (not folder_suffix or name.endswith(folder_suffix))
            ]
            if start_folder or folder_suffix:
                continue
        paths = [folder / name for name in sorted(filenames) if name.endswith('.png')]
        if paths:
            yield folder, paths


model_base_path = str(repo_root / 'data/models/he/nnUNetTrainerV2_BN_pathology_DA_ignore0_hed005__nnUNet_RGB_scaleTo_0_1_bs8_ps512')
norm = norm_01 
output_minus_1 = False 

print('\n> Model path:')
print(model_base_path, '\n')

folds = (0, 1, 2, 3, 4)
mixed_precision = None
checkpoint_name = "model_best"

print('> Loading model')
trainer, params = load_model_and_checkpoint_files(model_base_path, folds, mixed_precision=mixed_precision,
                                                  checkpoint_name=checkpoint_name)


##################################################
### Run inference
##################################################

print('> Running inference')
print('Saving inference in:', output_dir)

input_root = args.input if args.input.is_dir() else args.input.parent
if not args.input.is_dir() and (not args.input.is_file() or args.input.suffix.lower() != '.png'):
    parser.error('--input must be an existing PNG or a folder containing PNGs')

report_path = output_dir / 'report.txt'
failed_images = 0
batch_images = []
batch_paths = []


def save_batch(images, paths):
    palette = np.array(PALETTE, dtype=np.uint8)
    softmax_means = ensemble_softmax_batch(trainer, params, images)
    for relative_path, softmax_mean in zip(paths, softmax_means):
        mask_path = (output_dir / relative_path).with_suffix('.npz')
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        pred_output = softmax_mean.argmax(axis=-1)-(1 if output_minus_1 else 0)
        np.savez_compressed(mask_path, mask=pred_output.astype(np.uint8))
        preview_path = (preview_dir / relative_path).with_suffix('.png')
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        colored = Image.fromarray(palette[pred_output])
        colored.save(preview_path)


found_images = False
for folder, inference_images in iter_image_folders(args.input, args.start_folder, args.folder_suffix):
    found_images = True
    for path in tqdm(
        inference_images,
        desc=str(folder.relative_to(input_root)),
        ascii=True,
        position=0,
        leave=True,
    ):
        relative_path = Path(path).relative_to(input_root)
        try:
            with Image.open(path) as image:
                i = np.array(image.convert('RGB'))
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
            with report_path.open('a', encoding='utf-8') as report:
                report.write(f'{Path(relative_path).as_posix()}\n')
            failed_images += 1
            tqdm.write(f'Skipping unreadable image {relative_path}: {error}')
            continue

        i = np.expand_dims(i, axis=0)
        batch_images.append(norm(i).astype(np.float32))
        batch_paths.append(relative_path)
        if len(batch_images) >= args.batch_size:
            save_batch(batch_images, batch_paths)
            batch_images.clear()
            batch_paths.clear()

    # Finish this directory before discovering the next one.
    if batch_images:
        save_batch(batch_images, batch_paths)
        batch_images.clear()
        batch_paths.clear()

if not found_images:
    parser.error('No PNGs match the input and requested folder filters')

if failed_images:
    print(f'Skipped {failed_images} unreadable images; paths appended to {report_path}')
print('DONE')
