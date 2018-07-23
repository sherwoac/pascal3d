#!/usr/bin/env python

import pascal3d
from pascal3d import utils
import os.path as osp
import os
import scipy.misc
import numpy as np
import pandas as pd


from joblib import Parallel, delayed
import multiprocessing

num_cores = 1  # multiprocessing.cpu_count()
data_columns = ['id', 'file_name', 'test', '2d_bb8', '3d_bb8', 'D', 'gt_camera_pose']

def process_training_data(i, data_set1, data_set2, output_directory, image_file_type):
    len_datasets = len(data_set1) + len(data_set2)
    if i >= len(data_set1):
        data_set_index = i - len(data_set1)
        data_set = data_set2
    else:
        data_set_index = i
        data_set = data_set1

    data = data_set.get_data(data_set_index)

    if i % int(0.1 * len_datasets) == 0:
        print('percent: %s' % int(round((100 * i / len_datasets))))

    # only want to train against singular examples
    for object_number in len(data['objects']):
        img1 = data['img']
        class_dir = data['objects'][object_number][0]
        output_image_filename = osp.join(output_directory, class_dir,
                                         class_dir + '_' + '{0:05d}'.format(i) + image_file_type)
        bb8s = data_set.camera_transform_cad_bb8(data)
        assert len(bb8s) == 1, 'more than one bb8?'
        (bb8, Dx, Dy, Dz, bb83d) = bb8s[0]
        x_values, y_values = np.split(np.transpose(bb8), 2)

        if np.min(x_values) > 0 \
                and np.min(y_values) > 0 \
                and np.max(x_values) < img1.shape[1] \
                and np.max(y_values) < img1.shape[0]:

            dir_name = osp.dirname(output_image_filename)
            if not osp.isdir(dir_name):
                os.makedirs(dir_name)

            scipy.misc.imsave(output_image_filename, img1)

            # add ground truth camera
            obj = data['objects'][0][1]
            R_gt = utils.get_transformation_matrix(
                obj['viewpoint']['azimuth'],
                obj['viewpoint']['elevation'],
                obj['viewpoint']['distance'],
            )
            d = {'id':i,
                 'file_name':output_image_filename,
                 'test':False,
                 '2d_bb8':bb8,
                 '3d_bb8':bb83d,
                 'D':(Dx, Dy, Dz),
                 'gt_camera_pose':R_gt}

            return pd.DataFrame(d)

    return i, None


def main():
    output_directory = os.path.expanduser('~/Documents/UCL/PROJECT/DATA/BB8_PASCAL_DATA/')
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    df_test = create_testing_data(output_directory)
    df_train = create_training_data(output_directory)


def create_testing_data(output_directory):
    dataset = pascal3d.dataset.Pascal3DDataset('val',
                                                pascal3d.dataset.Pascal3DDataset.dataset_source_enum.pascal)

    if num_cores > 1:
        results = Parallel(n_jobs=num_cores)(delayed(processData)(i) for i in range(len_datasets))
    else:
        results = [processData(i) for i in range(len_datasets)]

    return pd.concat(results)

def create_training_data():
    data_set2 = pascal3d.dataset.Pascal3DDataset('all',
                                                pascal3d.dataset.Pascal3DDataset.dataset_source_enum.imagenet)
    data_set1 = pascal3d.dataset.Pascal3DDataset('train',
                                                pascal3d.dataset.Pascal3DDataset.dataset_source_enum.pascal)
    len_datasets = len(data_set1) + len(data_set2)

    output_directory = os.path.expanduser('~/Documents/UCL/PROJECT/DATA/BB8_PASCAL_DATA/TRAIN')
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    bb8_dict_file = osp.join(output_directory, 'bb8_points')
    image_file_type = '.jpg'
    bb8_points = {}


    if num_cores > 1:
        results = Parallel(n_jobs=num_cores)(delayed(processData)(i,
                                                                  data_set1,
                                                                  data_set2,
                                                                  output_directory,
                                                                  image_file_type) for i in range(len_datasets))
    else:
        results = [processData(i,
                               data_set1,
                               data_set2,
                               output_directory,
                               image_file_type) for i in range(len_datasets)]

    for i, bb8_result in results:
        if bb8_result is not None:
            assert len(bb8_result) == 4, "incorrect label pack size"
            bb8_points[i] = bb8_result

    np.save(bb8_dict_file, bb8_points)


if __name__ == '__main__':
    main()
