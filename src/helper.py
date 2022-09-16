import numpy as np
import pandas as pd
import re
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ast
import os
import time
from torch.nn import functional as F

# from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
# from skimage import io
# import pytorch_lightning as pl
# import matplotlib.pyplot as plt

#---dataset loading---#
def cube_to_numpy(path_cube):
    d, h, w = 128, 1024, 512
    arr_1d = np.fromfile(path_cube, dtype='uint8')
    arr_3d = arr_1d.reshape(d, h, w).transpose()
    arr_3d = np.flip(np.rot90(arr_3d, 1, (0,1)), 1)

    return arr_3d

#calculate optimal num_workers
class Calculations:
    def calc_num_workers(train_set, batch_size):
        pin_memory = True
        print('pin_memory is', pin_memory) 
        for num_workers in reversed(range(0, os.cpu_count()+1, 1)):
            train_loader = DataLoader(train_set, batch_size= batch_size, num_workers=num_workers, pin_memory=pin_memory)
            start = time.time()
            for epoch in range(1, 5):
                for i, data in enumerate(train_loader):
                    pass
            end = time.time()
            print("Finish with:{} second, num_workers={}".format(end - start, num_workers))

    def calc_mean_std(train_loader):
        mean = 0.
        std = 0.
        for images, _ in train_loader:
            batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)

        print('mean', mean / len(train_loader.dataset))
        print('std', std / len(train_loader.dataset))

#---Metrics---#

def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 0.0001
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def dice_coef_multilabel(y_true, y_pred, numLabels):
    dice=0
    for index in range(numLabels):
        dice += dice_coef(y_true[:,index,:,:], y_pred[:,index,:,:])
    return dice/numLabels # taking average

def dice_coef_separate_multilabel(y_true, y_pred, numLabels):
    dices = []
    for index in range(numLabels):
        dice = dice_coef(y_true[:,index,:,:], y_pred[:,index,:,:])
        dices.append(dice)
    return dices # taking average

class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

class DiceCoefficient(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceCoefficient, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]

# color_dict =  {0: (  0,   0,   0),
#                             1: (183, 188,  22),
#                             2: (195, 105, 149),
#                             3: (58,  232, 120),
#                             4: (59,  167, 195),
#                             5: (83,  114, 168),
#                             6: (154, 3,   214),
#                             7: (27,  62,  186),
#                             8: (87,  180, 175)}

color_dict =  {0: (  0,   0,   0),
                            1: (183, 188,  22),
                            2: (195, 105, 149),
                            3: (58,  232, 120),
                            4: (59,  167, 195),
                            5: (83,  114, 168),
                            6: (27,  62,  186),
                            7: (87,  180, 175)}

def rgb_to_onehot(rgb_arr, color_dict):
    """
    Input shape = (h, w, c)
    """
    num_classes = len(color_dict)
    shape = rgb_arr.shape[:2]+(num_classes,)
    arr = np.zeros( shape, dtype=np.int8 )
    for i, cls in enumerate(color_dict):
        arr[:,:,i] = np.all(rgb_arr.reshape( (-1,3) ) == color_dict[i], axis=1).reshape(shape[:2])
    return arr

def onehot_to_rgb(onehot, color_dict):
    """
    Input shape = (h, w, c)
    """
    single_layer = np.argmax(onehot, axis=0)
    output = np.zeros( onehot.shape[1:] + (3,) )
    # single_layer = np.argmax(onehot, axis=0)
    # output = np.zeros( (3,) + onehot.shape[1:] )
    for k in color_dict.keys():
        output[single_layer==k] = color_dict[k]
    return np.uint8(output)

def select_mask(df, index):
    """
    For each flip and non-flip image pair, randomly select one image to keep and one image to flip.

    Parameters
    ----------
    df: pandas dataframe
        Dataframe of image and set of mask paths.
    index: even integer
        Integer to allow the selection of flip and non-flip pair to apply the function to.

    Returns
    -------
    None: Instead, the dataframe will be updated.
    """
    
    mask_set = ast.literal_eval(df.iloc[index]['Mask'])
    random.shuffle(mask_set)
    if any('task_adjudication' in mask for mask in mask_set):
        mask = [mask for mask in mask_set if 'task_adjudication' in mask][0]
        df.at[index, 'Mask'] = mask
        df.at[index+1, 'Mask'] = mask
    elif len(mask_set) == 1:
        df.at[index, 'Mask'] = mask_set[0]
        df.at[index+1, 'Mask'] = mask_set[0]
    elif len(mask_set) >= 2:
        df.at[index, 'Mask'] = mask_set[0]
        df.at[index+1, 'Mask'] = mask_set[1]

def random_stratified_split(path_csv, path_save, val_set_size = int(20), test_set_size = int(20)):
    """
    This function opens a csv, splits the dataset according to eye (OD or OS) and slice position (Foveal or Non-foveal), 
    allowing the dataset to be stratified to those factors when making the train-validation-test set split.

    Parameters
    ----------
    path_csv: Relative path to the dataset csv file.
        The CSV file should contain the list of images and masks and their corresponding paths.
    path_save: Relative path to save new dataframe.
        Dataframe generated from stratifying and randomly splitting the dataset.

    Returns
    -------
    Dataframe of the train val test split.
    """
    #Adding columns: eye, slice and foveal
    df = pd.read_csv(path_csv, index_col=0)
    # df = df[df['Image_quality']=='normal']
    df.insert(2, 'Eye', '')
    df.insert(3, 'Slice', '')
    df.insert(4, 'Foveal', '')
    df = df.reset_index(drop=True)
    for index, row in df.iterrows():
        image = row['Image']
        result = re.search('slice_(.*).bmp', image)
        df.at[index, 'Slice'] = int(result.group(1))
        if 54 <= int(result.group(1)) <= 74:
            df.at[index, 'Foveal'] = 'foveal'
        else:
            df.at[index, 'Foveal'] = 'nonfoveal'
        result = image.split('_')
        df.at[index, 'Eye'] = result[-6]

    #stratifying dataset based on (1) eye and (2) foveal/nonfoveal cross section
    df_od_foveal = df[(df['Eye'] == 'OD') & (df['Foveal'] == 'foveal') & (df['Image_quality']=='normal')]
    df_od_nonfoveal = df[(df['Eye'] == 'OD') & (df['Foveal'] == 'nonfoveal') & (df['Image_quality']=='normal')]
    df_os_foveal = df[(df['Eye'] == 'OS') & (df['Foveal'] == 'foveal') & (df['Image_quality']=='normal')]
    df_os_nonfoveal = df[(df['Eye'] == 'OS') & (df['Foveal'] == 'nonfoveal') & (df['Image_quality']=='normal')]
    df_test2 = df[df['Image_quality'] !='normal']
    #shuffle indices of dataframe
    # pl.seed_everything(1234)
    indices_a = df_od_foveal.index.values
    indices_b = df_od_nonfoveal.index.values
    indices_c = df_os_foveal.index.values
    indices_d = df_os_nonfoveal.index.values
    indices_e = df_test2.index.values
    random.shuffle(indices_a)
    random.shuffle(indices_b)
    random.shuffle(indices_c)
    random.shuffle(indices_d)

    #Generate stratified indices for train val test splits
    val_split_index = int(val_set_size/4)
    test_split_index = val_split_index + int(test_set_size/4)
    val_set_indices = np.concatenate(((indices_a[0:val_split_index]), 
        (indices_b[:val_split_index]), 
        (indices_c[:val_split_index]), 
        (indices_d[:val_split_index])))
    test_set_indices = np.concatenate(((indices_a[val_split_index:test_split_index]), 
        (indices_b[val_split_index:test_split_index]), 
        (indices_c[val_split_index:test_split_index]), 
        (indices_d[val_split_index:test_split_index])))
    train_set_indices = np.concatenate(((indices_a[test_split_index:]), 
        (indices_b[test_split_index:]), 
        (indices_c[test_split_index:]), 
        (indices_d[test_split_index:])))

    #Add column for split status and flip, collate indices
    df.insert(2, 'Split', '')
    df.insert(3, 'Flip', '')
    for index in train_set_indices:
        index = int(index)
        df.at[index, 'Split'] = 'train'
    for index in val_set_indices:
        index = int(index)
        df.at[index, 'Split'] = 'val'
    for index in test_set_indices:
        index = int(index)
        df.at[index, 'Split'] = 'test'
    for index in indices_e:
        index = int(index)
        df.at[index, 'Split'] = 'test2'

    #Drop unnecessary columns, duplicate dataframe 
    df = df.drop(labels='Slice', axis=1)
    newdf = pd.DataFrame(np.repeat(df.values, 2, axis=0))
    newdf.columns = df.columns
    df = newdf
    df.reset_index(drop=True)

    #Build indices list to set flip value
    indices = list(range(len(df)))
    even_indices = [index for index in indices if index % 2 == 0]
    odd_indices = [index for index in indices if index % 2 == 1]
    for index in even_indices: df.at[index, 'Flip'] = 'no'
    for index in odd_indices: df.at[index, 'Flip'] = 'yes'

    #use even indices to select from mask set
    for index in even_indices:
        select_mask(df, index)
    
    #Save random seed dataframe, then select split
    df.to_csv(path_save)

    return df