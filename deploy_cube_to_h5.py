# %cd /workspaces/dlg_segmentation
import pandas as pd
import pytorch_lightning as pl
import torch
import h5py
import numpy as np
from tqdm import tqdm
from src.dataset import *
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import transforms
from src.helper import *
from src.models import *

path_img = "path/to/img/files"

#obtain list of files
df = pd.read_json('progressa_master.json')
od_scans_baseline = list(df['od_scans_macular_baseline_name'])
od_scans_baseline = [value for value in od_scans_baseline if value!=None]

#args
ckpt_path = 'lightning_logs/UnetL4_skip/version_0/checkpoints/unet-epoch=99-validation_dice_argmax=0.91.ckpt'
mean = 0.1493
std = 0.1001
num_workers = 15
batch_size = 16
continue_generation = True

#image specific transforms    
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean), (std))
    ])
#mask specific transforms
mask_trans = transforms.Compose([
    transforms.ToTensor()
    ])

#generate list
def chunks(xs, n):
    n = max(1, n)
    return (xs[i:i+n] for i in range(0, len(xs), n))

#generate list of images
df = pd.read_json('progressa_master.json')
od_scans_baseline = list(df['od_scans_macular_baseline_name'])
od_scans_baseline = [value for value in od_scans_baseline if value!=None]
os_scans_baseline = list(df['os_scans_macular_baseline_name'])
os_scans_baseline = [value for value in os_scans_baseline if value!=None]
list_scans = od_scans_baseline + os_scans_baseline

if continue_generation:
    f = h5py.File('cube_predictions.h5', 'r')
    a = list_scans
    b = list(f.keys())
    list_scans = [item for item in a if item not in b]
    f.close()

new_list = chunks(list_scans, 3)
#load data
for list_files in tqdm(list(new_list)):
    cube_set = CubeDataset(path_img, list_files, transform=trans)
    cube_loader = DataLoader(cube_set, batch_size=batch_size, num_workers=num_workers)

    #model - determining lightning logs path based on whether user wants to continue or start from epoch 0
    args = None
    model_name = ckpt_path.split('/')[1]
    model = LitSegmentation(args, models[model_name])

    #trainer
    trainer = pl.Trainer(accelerator="gpu", devices=1, logger=False)
    inferences = trainer.predict(model, cube_loader, ckpt_path = ckpt_path)

    #save to hdf5
    cube_stack = torch.concat(tuple([inference for inference in inferences]))
    cube_stack = cube_stack.numpy()
    cube_stack = np.array([onehot_to_rgb(cube_stack[image,:,:,:], color_dict) for image in range(cube_stack.shape[0])], dtype='uint8')

    f = h5py.File('cube_predictions.h5', 'a')
    print('Saving cubes')
    for index_img in tqdm(range(len(list_files))):
        cube = cube_stack[index_img*128:(index_img+1)*128]
        try:
            f.create_dataset(list_files[index_img], data=cube)
        except:
            continue
    f.close()