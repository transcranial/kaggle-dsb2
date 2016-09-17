import os
import csv
import re
import pickle
import random
import math
import dicom
import numpy as np
from tqdm import tqdm
from natsort import natsorted
from skimage import transform
from sklearn.externals import joblib
from scipy import ndimage

def create_filepaths_dict(dataset='test'):
    if dataset == 'train':
        nrange = range(1, 501)
    elif dataset == 'validate':
        nrange = range(501, 701)
    elif dataset == 'test':
        nrange = range(701, 1141)
    else:
        raise

    data_path = '../../data/test'

    slice_filepaths = {}
    for i in tqdm(nrange):
        filepaths = {}
        folders = natsorted([(x.name, os.path.abspath(x.path)) for x in os.scandir('{}/{}/study'.format(data_path, i)) \
                             if x.is_dir()])
        for folder_name, folder_path in folders:
            files = natsorted([(x.name, os.path.abspath(x.path)) for x in os.scandir(folder_path) \
                               if x.is_file() and x.name.endswith('.dcm')])
            filepaths[folder_name] = files
        slice_filepaths[i] = filepaths

    return slice_filepaths

filepaths_test = create_filepaths_dict(dataset='test')

def create_label(pt, mode='ED'):
    if mode == 'ES':
        return systole_labels[pt] < np.arange(600)
    elif mode == 'ED':
        return diastole_labels[pt] < np.arange(600)
    else:
        raise

        
def apply_window(arr, window_center, window_width):
    return np.clip(arr, window_center - window_width/2, window_center + window_width/2)


def apply_per_slice_norm(arr):
    mean = np.mean(arr.ravel())
    std = np.std(arr.ravel())
    if std == 0:
        return np.zeros(arr.shape)
    return (arr - mean) / std


def crop_to_square(arr, size):
    x_len, y_len = arr.shape
    shorter_len = min(x_len, y_len)
    x_start = (arr.shape[0] - shorter_len) // 2
    x_end = x_start + shorter_len
    y_start = (arr.shape[1] - shorter_len) // 2
    y_end = y_start + shorter_len
    return transform.resize(arr[x_start:x_end, y_start:y_end], 
                            (size, size), order=1, clip=True, preserve_range=True)


def crop_to_square_normalized(img_orig, pixel_spacing, size):
    img_norm = ndimage.interpolation.zoom(img_orig, [float(x) for x in pixel_spacing], order=0, mode='constant')
    
    length_x, length_y = img_norm.shape
    if length_x >= size:
        x_start = length_x // 2 - size // 2
        x_end = length_x // 2 + size // 2
    else:
        x_start = 0
        x_end = length_x
    if length_y >= size:
        y_start = length_y // 2 - size // 2
        y_end = length_y // 2 + size // 2
    else:
        y_start = 0
        y_end = length_y
    
    img_new = np.zeros((size, size))
    new_x_shift = (size - (x_end - x_start)) // 2
    new_y_shift = (size - (y_end - y_start)) // 2
    img_new[new_x_shift:(new_x_shift + x_end - x_start), 
            new_y_shift:(new_y_shift + y_end - y_start)] = img_norm[x_start:x_end, y_start:y_end]
    
    return img_new


def localize_to_centroid(img, centroid, width_about_centroid):
    # assumes already cropped to square
    x, y = centroid
    x = int(round(x))
    y = int(round(y))
    x_start = x - width_about_centroid // 2
    x_end = x + width_about_centroid // 2
    y_start = y - width_about_centroid // 2
    y_end = y + width_about_centroid // 2
    
    if x_start < 0:
        x_end += (0 - x_start)
        x_start = 0
    if x_end > img.shape[0]:
        x_start -= (img.shape[0] - x_end)
        x_end = img.shape[0]
    if y_start < 0:
        y_end += (0 - y_start)
        y_start = 0
    if y_end > img.shape[1]:
        y_start -= (img.shape[1] - y_end)
        y_end = img.shape[1]
        
    return img[x_start:x_end, y_start:y_end], (x_start, x_end), (y_start, y_end)


def normalized_z_loc(df):
    # assumes patient position HFS
    
    position = [float(s) for s in df.ImagePositionPatient]
    orientation = [float(s) for s in df.ImageOrientationPatient]
    
    # first voxel coordinates from DICOM ImagePositionPatient field
    x_loc, y_loc, z_loc = position
    
    # row/column direction cosines from DICOM ImageOrientationPatient field
    row_dircos_x, row_dircos_y, row_dircos_z, col_dircos_x, col_dircos_y, col_dircos_z = orientation
    
    # normalized direction cosines
    dircos_x = row_dircos_y * col_dircos_z - row_dircos_z * col_dircos_y
    dircos_y = row_dircos_z * col_dircos_x - row_dircos_x * col_dircos_z
    dircos_z = row_dircos_x * col_dircos_y - row_dircos_y * col_dircos_x
    
    # z-coordinate location recalculated based on reference
    z_loc_norm = dircos_x * x_loc + dircos_y * y_loc + dircos_z * z_loc
    return z_loc_norm


def get_all_series_filepaths(filepaths):
    t_slices = 30
    
    # create sax series filepaths
    # handles irregularies such as those including z-slices and t-slices in the same folder
    series_filepaths_all = []
    for view in filepaths.keys(): 
        if not re.match(r'^sax', view):
            continue
        
        if len(filepaths[view]) == t_slices:
            series_filepaths_all.append(filepaths[view])
        elif len(filepaths[view]) < t_slices:
            series_filepaths_all.append(filepaths[view][:] + filepaths[view][:(t_slices - len(filepaths[view]))])
        else:
            if re.match(r'^\w+-\d+-\d+-\d+.dcm$', filepaths[view][0][0]) is not None:
                series_filepaths_split = []
                slices_list = []
                series_filepaths_sort_by_slice = sorted(filepaths[view][:], 
                                                        key=lambda x: '{}-{}'.format(x[0].split('-')[-1].split('.')[0], 
                                                                                     x[0].split('-')[-2]))
                for fname, fpath in series_filepaths_sort_by_slice:
                    nslice = fname.split('-')[-1].split('.')[0]
                    tframe = fname.split('-')[-2]
                    if nslice not in slices_list:
                        if len(series_filepaths_split) == t_slices:
                            series_filepaths_all.append(series_filepaths_split)
                        elif len(series_filepaths_split) < t_slices and len(series_filepaths_split) > 0:
                            series_filepaths_all.append((series_filepaths_split[:] + 
                                                         series_filepaths_split[:(t_slices - len(series_filepaths_split))]))
                        series_filepaths_split = []
                        series_filepaths_split.append((fname, fpath))
                        slices_list.append(nslice)
                    else:
                        series_filepaths_split.append((fname, fpath))
                        
    return series_filepaths_all

def create_mean_percent_diff_seq(series_filepaths, img_size=256):
    nb_frames = 30
    if len(series_filepaths) != nb_frames:
        return None

    mean_diff_seq = []

    fname, fpath = series_filepaths[0]
    df = dicom.read_file(fpath)
    img_ED = apply_per_slice_norm(crop_to_square_normalized(df.pixel_array, df.PixelSpacing, img_size))
    for fname, fpath in series_filepaths[1:]:
        differences = []

        df = dicom.read_file(fpath)
        img_frame = apply_per_slice_norm(crop_to_square_normalized(df.pixel_array, df.PixelSpacing, img_size))
        img_percent_diff = (img_frame - img_ED) / (img_ED + 1e-6)
        differences.append(np.mean(img_percent_diff))
        differences.append(np.mean(np.abs(img_percent_diff)))

        for scaling in [2, 4, 8]:
            for i in range(scaling):
                for j in range(scaling):
                    differences.append(np.mean(img_percent_diff[(i*img_size//scaling):((i+1)*img_size//scaling),
                                                                (i*img_size//scaling):((i+1)*img_size//scaling)],
                                               axis=None))
                    differences.append(np.mean(np.abs(img_percent_diff[(i*img_size//scaling):((i+1)*img_size//scaling),
                                                                       (i*img_size//scaling):((i+1)*img_size//scaling)]),
                                               axis=None))

        mean_diff_seq.append(differences)

    return np.array(mean_diff_seq).astype(np.float32)

def predict_es_frame(filepaths):
    clf = joblib.load('../../model_weights/ES_detection_GBR.pkl')
    series_filepaths_all = get_all_series_filepaths(filepaths)
    data = []
    for series_filepaths in series_filepaths_all:
        data.append(create_mean_percent_diff_seq(series_filepaths, img_size=256))
    preds = clf.predict([x.ravel() for x in data])
    return np.round(np.mean(preds))

pt_es_frame = {}
for pt in tqdm(range(701, 1141)):
    pt_es_frame[pt] = int(predict_es_frame(filepaths_test[pt]))

with open('../../data_supp/pt_es_frame_test.pkl', 'wb') as f:
    pickle.dump(pt_es_frame, f)
