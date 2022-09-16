import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from src.helper import *

### Load calculation regions ###
def calculation_regions(df_inner, df_outer):
    #overall donut
    overall_donut = np.zeros((128, 1024, 512, 8))
    overall_total_a_scans = 0
    for slice in range(22,53+1):
        x1 = round(df_outer.loc[slice, 'first_a_scan'])
        x2 = round(df_outer.loc[slice, 'last_a_scan'])
        overall_donut[slice,:,x1:x2+1,:] = 1
        overall_total_a_scans += x2+1-x1
    for slice in range(54,74+1):
        x1 = round(df_outer.loc[slice, 'first_a_scan'])
        x2 = round(df_inner.loc[slice, 'first_a_scan'])
        x3 = round(df_inner.loc[slice, 'last_a_scan'])
        x4 = round(df_outer.loc[slice, 'last_a_scan'])
        overall_donut[slice,:,x1:x2+1,:] = 1
        overall_donut[slice,:,x3:x4+1,:] = 1
        overall_total_a_scans += x2+1-x1
        overall_total_a_scans += x4+1-x3
    for slice in range(75,106+1):
        x1 = round(df_outer.loc[slice, 'first_a_scan'])
        x2 = round(df_outer.loc[slice, 'last_a_scan'])
        overall_donut[slice,:,x1:x2+1,:] = 1
        overall_total_a_scans += x2+1-x1
    #inferior donut
    inferior_donut = np.zeros((128, 1024, 512, 8))
    inferior_total_a_scans = 0
    for slice in range(22,53+1):
        x1 = round(df_outer.loc[slice, 'first_a_scan'])
        x2 = round(df_outer.loc[slice, 'last_a_scan'])
        inferior_donut[slice,:,x1:x2+1,:] = 1
        inferior_total_a_scans += x2+1-x1
    for slice in range(54,64+1):
        x1 = round(df_outer.loc[slice, 'first_a_scan'])
        x2 = round(df_inner.loc[slice, 'first_a_scan'])
        inferior_donut[slice,:,x1:x2+1,:] = 1
        inferior_total_a_scans += x2+1-x1
    #superior donut
    superior_donut = np.zeros((128, 1024, 512, 8))
    superior_total_a_scans = 0
    for slice in range(64,74+1):
        x1 = round(df_inner.loc[slice, 'last_a_scan'])
        x2 = round(df_outer.loc[slice, 'last_a_scan'])
        superior_donut[slice,:,x1:x2+1,:] = 1
        superior_total_a_scans += x2+1-x1
    for slice in range(75,106+1):
        x1 = round(df_outer.loc[slice, 'first_a_scan'])
        x2 = round(df_outer.loc[slice, 'last_a_scan'])
        superior_donut[slice,:,x1:x2+1,:] = 1
        superior_total_a_scans += x2+1-x1

    return overall_donut, overall_total_a_scans, superior_donut, superior_total_a_scans, inferior_donut, inferior_total_a_scans

df_inner = pd.read_excel('dataset/glaucoma_macular_inner_ellipse_coordinates.xlsx')
df_outer = pd.read_excel('dataset/glaucoma_macular_outer_ellipse_coordinates.xlsx')
overall_donut, overall_total_a_scans, superior_donut, superior_total_a_scans, inferior_donut, inferior_total_a_scans = calculation_regions(df_inner, df_outer)

### Load cube dataset ###
def chunks(xs, n):
    n = max(1, n)
    return (xs[i:i+n] for i in range(0, len(xs), n))

def cube_calculations(cube, donut, total_a_scans):
    zone = cube*donut
    layer_thicknesses = []
    for number in range(1,8):
        cube_arr = zone[:,:,:,number].flatten()
        count = np.bincount(cube_arr.astype(int))[1]
        layer_thicknesses.append(count/total_a_scans*(2000/1024))
    layer_thicknesses = [sum(layer_thicknesses)] + layer_thicknesses

    return(layer_thicknesses)

def fill_results(df_results, key, selection, layer_thicknesses):
    prog_id_int = key.split('_')[1]
    eye = key.split('_')[-4]
    df_results.loc[df_results['id_zeiss_int'] == prog_id_int, eye.lower()+'_'+selection+'_macula'] = layer_thicknesses[0]
    df_results.loc[df_results['id_zeiss_int'] == prog_id_int, eye.lower()+'_'+selection+'_macula_1_rnfl'] = layer_thicknesses[1]
    df_results.loc[df_results['id_zeiss_int'] == prog_id_int, eye.lower()+'_'+selection+'_macula_2_gcipl'] = layer_thicknesses[2]
    df_results.loc[df_results['id_zeiss_int'] == prog_id_int, eye.lower()+'_'+selection+'_macula_3_inl'] = layer_thicknesses[3]
    df_results.loc[df_results['id_zeiss_int'] == prog_id_int, eye.lower()+'_'+selection+'_macula_4_opl'] = layer_thicknesses[4]
    df_results.loc[df_results['id_zeiss_int'] == prog_id_int, eye.lower()+'_'+selection+'_macula_5_onl'] = layer_thicknesses[5]
    df_results.loc[df_results['id_zeiss_int'] == prog_id_int, eye.lower()+'_'+selection+'_macula_6_pr'] = layer_thicknesses[6]
    df_results.loc[df_results['id_zeiss_int'] == prog_id_int, eye.lower()+'_'+selection+'_macula_7_rpe'] = layer_thicknesses[7]

f = h5py.File('cube_predictions.h5', 'r')
list_keys = list(f.keys())
list_keys = chunks(list_keys, 3)
df_results = pd.read_json('progressa_master.json')

for chunk in tqdm(list_keys):
    ### Make calculations ###
    for key in (list(chunk)):
        try:
            f = h5py.File('cube_predictions.h5', 'r')
            cube = f.get(key)[:]
            f.close()
            cube = np.array([rgb_to_onehot(slice, color_dict) for slice in cube])
            
            overall_layer_thicknesses = cube_calculations(cube, overall_donut, overall_total_a_scans)
            superior_layer_thicknesses = cube_calculations(cube, superior_donut, superior_total_a_scans)
            inferior_layer_thicknesses = cube_calculations(cube, inferior_donut, inferior_total_a_scans)
                
            fill_results(df_results, key, 'overall', overall_layer_thicknesses)
            fill_results(df_results, key, 'superior', superior_layer_thicknesses)
            fill_results(df_results, key, 'inferior', superior_layer_thicknesses)
        except:
            pass
    df_results.to_json('progressa_thicknesses.json')