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
parser.add_argument('--input', type=Path, help='Your H&E PNG file or folder of PNG crops (exclude thumbnails)')
parser.add_argument('--output', type=Path, help='Output mask folder')
args = parser.parse_args()
repo_root = Path(__file__).resolve().parents[2]
output_dir = args.output or repo_root / 'data/inference/he'
output_dir.mkdir(parents=True, exist_ok=True)


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
    inference_images = sorted(args.input.glob('*.png')) if args.input.is_dir() else [args.input]
    if not inference_images or any(not p.is_file() or p.suffix.lower() != '.png' for p in inference_images):
        parser.error('--input must be an existing PNG or a folder containing PNGs')
else:
    df = pd.read_csv(repo_root / 'data/data_overview.csv')
    inference_images = sorted(df[(df['task'] == 'he_tissue_segmentation') & (df['split'] == 'test')]['image_path'].str.replace('.png', '_with_context.png').tolist())

for i in tqdm(inference_images):
    stem = Path(i).stem

    image = Image.open(i).convert('RGB')
    i = np.array(image)
    i = np.expand_dims(i, axis=0)
    i = norm(i)

    softmax_list = ensemble_softmax_list(trainer, params, i)
    softmax_mean = np.array(softmax_list).mean(0)
    pred_output = softmax_mean.argmax(axis=-1)-(1 if output_minus_1 else 0)
    Image.fromarray(pred_output.astype(np.uint8)).save(output_dir / f'{stem}.png') # each pixel have val btw 0-16 so the image look dark
    print(f'{stem}: class IDs and pixel counts:', dict(zip(*np.unique(pred_output, return_counts=True))))
    # Raw IDs 0-16 look almost black in a normal grayscale image viewer.
    palette = np.array(PALETTE, dtype=np.uint8)
    preview_dir = output_dir / 'previews'
    preview_dir.mkdir(exist_ok=True)
    colored = Image.fromarray(palette[pred_output])
    colored.save(preview_dir / f'{stem}_color.png')
    Image.blend(image, colored, 0.45).save(preview_dir / f'{stem}_overlay.png')

print('DONE')
