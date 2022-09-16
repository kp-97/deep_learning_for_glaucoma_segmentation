import numpy as np
from skimage import io
from torch.utils.data import Dataset 
from src.helper import *
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import pdb

class SegmentationDataset(Dataset):
    """Segmentaiton Dataset. Load and train OCT cross sections."""

    def __init__(self, path_csv, path_save, split, transform=None, mask_transform=None):
        """
        Args:
            path_csv (string): Path to the csv file with annotations.
            path_save (string): Path to save the csv file that was randomly generated for the train, val and test splits.
            split (string): Options include: 'train', 'val', 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
            mask_transform (callable, optional): Optional transform to be applied on a sample.
        """
                
        df = random_stratified_split(path_csv, path_save)
        df = df[df['Split'] == split]
        
        df = df[df['Eye'] == "OS"] #OS or OD
        # df = df[df['Foveal'] == 'nonfoveal'] #foveal or nonfoveal
        self.segmentation_frame = df
        
        self.transform = transform
        self.mask_transform = mask_transform
        self.color_dict =  {0: (  0,   0,   0),
                            1: (183, 188,  22),
                            2: (195, 105, 149),
                            3: (58,  232, 120),
                            4: (59,  167, 195),
                            5: (83,  114, 168),
                            6: (27,  62,  186),
                            7: (87,  180, 175)}

    def __len__(self):
        return len(self.segmentation_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.segmentation_frame.iloc[idx, 0]
        mask_name = self.segmentation_frame.iloc[idx, 1]
        image = io.imread(img_name, as_gray = True)
        masks = io.imread(mask_name)
        masks = rgb_to_onehot(masks, self.color_dict)
        flip = self.segmentation_frame.iloc[idx]['Flip']
        
        if self.transform:
            image = self.transform(image)
            if flip == 'yes':
                image = torch.fliplr(image)
                
        if self.mask_transform:
            masks = self.mask_transform(masks)
            masks = masks.type(torch.FloatTensor)
            if flip == 'yes':
                masks = torch.fliplr(masks)
        
        return [image, masks]

#testing prediction of single image
class SegmentationDataset_single(Dataset):
    """Segmentaiton Dataset. Load and train OCT cross sections."""

    def __init__(self, path_image, transform=None, mask_transform=None):
        """
        Args:
            path_csv (string): Path to the csv file with annotations.
            path_save (string): Path to save the csv file that was randomly generated for the train, val and test splits.
            split (string): Options include: 'train', 'val', 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
            mask_transform (callable, optional): Optional transform to be applied on a sample.
        """
                
        # df = random_stratified_split(path_csv, path_save)
        # df = df[df['Split'] == split]
        # self.segmentation_frame = df
        self.image_path = path_image
        self.transform = transform
        self.mask_transform = mask_transform
        # self.color_dict =  {0: (  0,   0,   0),
        #                     1: (183, 188,  22),
        #                     2: (195, 105, 149),
        #                     3: (58,  232, 120),
        #                     4: (59,  167, 195),
        #                     5: (83,  114, 168),
        #                     6: (27,  62,  186),
        #                     7: (87,  180, 175)}

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.image_path
        image = io.imread(img_name, as_gray = True)
        masks = torch.empty(1, dtype=torch.float)
        
        if self.transform:
            image = self.transform(image)
              
        return [image, masks]

    
class SegmentationDataset_singlecube(Dataset):
    """Segmentaiton Dataset. Load and train OCT cross sections."""

    def __init__(self, path_cube, transform=None, mask_transform=None):
        """
        Args:
            path_csv (string): Path to the csv file with annotations.
            path_save (string): Path to save the csv file that was randomly generated for the train, val and test splits.
            split (string): Options include: 'train', 'val', 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
            mask_transform (callable, optional): Optional transform to be applied on a sample.
        """

        d, h, w = 128, 1024, 512
        arr_1d = np.fromfile(path_cube, dtype='uint8')
        arr_3d = arr_1d.reshape(d, h, w).transpose()
        arr_3d = np.flip(np.rot90(arr_3d, 1, (0,1)), 1)

        # df = random_stratified_split(path_csv, path_save)
        # df = df[df['Split'] == split]
        # self.segmentation_frame = df
        self.image = arr_3d[:,:,64]
        self.transform = transform
        self.mask_transform = mask_transform
        # self.color_dict =  {0: (  0,   0,   0),
        #                     1: (183, 188,  22),
        #                     2: (195, 105, 149),
        #                     3: (58,  232, 120),
        #                     4: (59,  167, 195),
        #                     5: (83,  114, 168),
        #                     6: (27,  62,  186),
        #                     7: (87,  180, 175)}

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.image.copy()
        masks = torch.empty(1, dtype=torch.float)
        
        if self.transform:
            image = self.transform(image)
              
        return [image, masks]

class CubeDataset(Dataset):
    """CubeDataset. Load OCT cubes."""

    def __init__(self, path_cubes, path_list, transform=None, mask_transform=None):
        """
        Args:
            path_cubes(string): path to the folder container .img exports
            path_list (list): list of file names
            transform (callable, optional): Optional transform to be applied on a sample. Must be set to correctly make predictions.
            mask_transform (callable, optional): Optional transform to be applied on a sample.
        """
        arr_3d_appended = np.empty((1024, 512, 1), dtype='uint8')
        print('Loading cubes')
        for path in tqdm(path_list):
            path_cube = next(Path(path_cubes).rglob('*'+path))
            d, h, w = 128, 1024, 512
            arr_1d = np.fromfile(path_cube, dtype='uint8')
            arr_3d = arr_1d.reshape(d, h, w).transpose()
            arr_3d = np.flip(np.rot90(arr_3d, 1, (0,1)), 1)
            arr_3d_appended = np.concatenate((arr_3d_appended, arr_3d), axis=2)
        self.arr_3d = arr_3d_appended[:,:,1:]
        self.depth = self.arr_3d.shape[2]
        self.transform = transform
    def __len__(self):
        return self.depth

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.arr_3d[:,:,idx]
        image = image.copy()
        
        if self.transform:
            image = self.transform(image)
                
        image = image.to(torch.float32)
        masks = torch.empty(1, dtype=torch.float)
        
        return [image, masks]

class deprecated_CubeDataset(Dataset):
    """CubeDataset. Load OCT cubes."""

    def __init__(self, path_list, transform=None, mask_transform=None):
        """
        Args:
            path_csv (string): Path to the csv file with annotations.
            path_save (string): Path to save the csv file that was randomly generated for the train, val and test splits.
            split (string): Options include: 'train', 'val', 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
            mask_transform (callable, optional): Optional transform to be applied on a sample.
        """
        d, h, w = 128, 1024, 512
        arr_1d = np.fromfile(path_cube, dtype='uint8')
        arr_3d = arr_1d.reshape(d, h, w).transpose()
        arr_3d = np.flip(np.rot90(arr_3d, 1, (0,1)), 1)
        self.depth = d
        self.arr_3d = arr_3d
        self.transform = transform
    def __len__(self):
        return self.depth

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.arr_3d[:,:,idx]
        image = Image.fromarray(image)
        image = np.array(image)
        
        if self.transform:
            image = self.transform(image)
        
        image = image.to(device=device, dtype=torch.float)
        masks = torch.empty(1, dtype=torch.float)
        return [image, masks]

import pytorch_lightning as pl
from torch.utils.data import DataLoader
# Note - you must have torchvision installed for this example
from torchvision.datasets import MNIST
from torchvision import transforms
from typing import Optional

class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, mean, std, path_csv, path_save, batch_size, num_workers, eye=None):
        super().__init__()
        self.path_csv = path_csv
        self.path_save = path_save
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.eye = eye
        #image specific transforms    
        self.trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean), (std))
        ])

        #mask specific transforms
        self.mask_trans = transforms.Compose([
        transforms.ToTensor()
        ])

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_set = SegmentationDataset(self.path_csv, self.path_save, 'train', transform = self.trans, mask_transform = self.mask_trans)
            self.val_set = SegmentationDataset(self.path_csv, self.path_save, 'val', transform = self.trans, mask_transform = self.mask_trans)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_set = SegmentationDataset(self.path_csv, self.path_save, 'test2', transform = self.trans, mask_transform = self.mask_trans)

        if stage == "validate" or stage is None:
            self.val_set = SegmentationDataset(self.path_csv, self.path_save, 'val', transform = self.trans, mask_transform = self.mask_trans)
        # if stage == "predict" or stage is None:
        #     self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)

    # def predict_dataloader(self):
    #     return DataLoader(self.mnist_predict, batch_size=32)