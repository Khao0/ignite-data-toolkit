import argparse
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
from nnunet.training.model_restore import load_model_and_checkpoint_files
from label_colors import PALETTE

print('RUNNING INFERENCE SCRIPT')
parser = argparse.ArgumentParser(description="Run IGNITE H&E segmentation on the published test set or your RGB PNGs at 0.5 microns/pixel.")
parser.add_argument('--input', type=Path, help='Your H&E PNG file or folder of PNG crops, searched recursively (exclude thumbnails)')
parser.add_argument('--output', type=Path, help='Output compressed NumPy (.npz) mask folder; preserves input subfolders')
parser.add_argument('--preview-output', type=Path, help='Colored PNG folder (default: <output>/previews)')
parser.add_argument('--start-folder', help='Resume at this immediate input subfolder, inclusive, in alphabetical order')
parser.add_argument('--folder-suffix', help='Only process immediate input subfolders ending with this suffix')
args = parser.parse_args()
if (args.start_folder or args.folder_suffix) and (args.input is None or not args.input.is_dir()):
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

def ensemble_softmax_list(trainer, params, x_batch):
    softmax_list = []
    for p in params:
        trainer.load_checkpoint_ram(p, False)
        softmax_list.append(
            trainer.predict_preprocessed_data_return_seg_and_softmax(x_batch.astype(np.float32), verbose=False,
                                                                     do_mirroring=False, mirror_axes=[])[
                -1].transpose(1, 2, 3, 0).squeeze())
    return softmax_list

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

if args.input:
    input_root = args.input if args.input.is_dir() else args.input.parent
    inference_images = sorted(args.input.rglob('*.png')) if args.input.is_dir() else [args.input]
    if not inference_images or any(not p.is_file() or p.suffix.lower() != '.png' for p in inference_images):
        parser.error('--input must be an existing PNG or a folder containing PNGs')
    if args.start_folder or args.folder_suffix:
        inference_images = [
            path for path in inference_images
            if len(path.relative_to(input_root).parts) > 1
            and (not args.start_folder or path.relative_to(input_root).parts[0] >= args.start_folder)
            and (not args.folder_suffix or path.relative_to(input_root).parts[0].endswith(args.folder_suffix))
        ]
        if not inference_images:
            parser.error('No PNGs match the requested start folder and folder suffix')
else:
    input_root = None
    df = pd.read_csv(repo_root / 'data/data_overview.csv')
    inference_images = sorted(df[(df['task'] == 'he_tissue_segmentation') & (df['split'] == 'test')]['image_path'].str.replace('.png', '_with_context.png').tolist())

for i in tqdm(inference_images):
    stem = Path(i).stem
    relative_path = Path(i).relative_to(input_root) if input_root is not None else Path(i).name
    mask_path = (output_dir / relative_path).with_suffix('.npz')
    preview_path = (preview_dir / relative_path).with_suffix('.png')
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.open(i).convert('RGB')
    i = np.array(image)
    i = np.expand_dims(i, axis=0)
    i = norm(i)

    softmax_list = ensemble_softmax_list(trainer, params, i)
    softmax_mean = np.array(softmax_list).mean(0)
    pred_output = softmax_mean.argmax(axis=-1)-(1 if output_minus_1 else 0)
    np.savez_compressed(mask_path, mask=pred_output.astype(np.uint8))
    print(f'{stem}: class IDs and pixel counts:', dict(zip(*np.unique(pred_output, return_counts=True))))
    # Raw IDs 0-16 look almost black in a normal grayscale image viewer.
    # palette = np.array(PALETTE, dtype=np.uint8)
    # colored = Image.fromarray(palette[pred_output])
    # colored.save(preview_path)

print('DONE')
