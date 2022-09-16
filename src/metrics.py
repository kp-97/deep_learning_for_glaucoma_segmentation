import numpy as np
import pdb

def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 0.0001
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

#Metric 1
def dice_coef_multilabel(y_true, y_pred, numLabels):
    numBatches = y_true.shape[0]
    dice_scores = []
    for batch in range(numBatches):
        dice_scores_list = np.array([dice_coef(y_true[batch,index,:,:], y_pred[batch,index,:,:]) for index in range(0, numLabels)]) #change this to 1, numLabels
        dice = np.array([np.sum(dice_scores_list)/numLabels])
        dice_scores_list = np.concatenate((dice, dice_scores_list[1:numLabels]))
        dice_scores.append(dice_scores_list)
    return np.array(dice_scores)

def thickness_diff(y_true, y_pred):
    y_pred = y_pred.flatten()
    try:
        y_pred = np.bincount(y_pred.astype(int))[1]/512
    except:
        y_pred = 0
    
    y_true = y_true.flatten()
    try:
        y_true = np.bincount(y_true.astype(int))[1]/512
    except:
        y_true = 0
    
    return (y_pred - y_true) * (2000/1024)

#Metric 2
def signed_thickness_diff_multilabel(y_true, y_pred, numLabels):
    numBatches = y_true.shape[0]
    # thickness_overall = []
    for batch in range(numBatches):
        diff_list = np.array([thickness_diff(y_true[batch,index,:,:], y_pred[batch,index,:,:]) for index in range(0, numLabels)])
        # diff_overall = np.array([np.sum(diff_list)/numLabels])
        # diff_list = np.concatenate((diff_overall, diff_list))
        # thickness_overall.append(diff_list)
    return diff_list #np.array(thickness_overall)

#Metric 3
def unsigned_thickness_diff_multilabel(y_true, y_pred, numLabels):
    numBatches = y_true.shape[0]
    # thickness_overall = []
    for batch in range(numBatches):
        diff_list = np.array([thickness_diff(y_true[batch,index,:,:], y_pred[batch,index,:,:]) for index in range(0, numLabels)])
        diff_list = [abs(val) for val in diff_list]
        # diff_overall = np.array([np.sum(abs(diff_list))/numLabels]) #remember to get absolute values for individual layers
        # diff_list = np.concatenate((diff_overall, diff_list))
        # thickness_overall.append(diff_list)
    return diff_list #np.array(thickness_overall)

# Pairwise Metric 1
def pair_dice_coef_multilabel(y_true, y_pred, numLabels):
    # dice_overall = []
    dice_list = [dice_coef(y_true[:,:,index], y_pred[:,:,index]) for index in range(1, numLabels)]
    dice = sum(dice_list)/numLabels
    dice_overall = [dice] + dice_list
    # print(dice_overall)
    return dice_overall

# Pairwise Metric 2
def pair_signed_thickness_diff_multilabel(y_true, y_pred, numLabels):
    diff_list = [(thickness_diff(y_true[:,:,index], y_pred[:,:,index])) for index in range(1, numLabels)]
    diff_overall = sum(diff_list)/numLabels
    thickness_overall = [diff_overall] + diff_list
    return thickness_overall

# Pairwise Metric 3
def pair_unsigned_thickness_diff_multilabel(y_true, y_pred, numLabels):
    diff_list = [abs(thickness_diff(y_true[:,:,index], y_pred[:,:,index])) for index in range(1, numLabels)]
    diff_overall = sum(diff_list)/numLabels
    thickness_overall = [diff_overall] + diff_list
    return thickness_overall
    