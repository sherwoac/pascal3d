#!/usr/bin/env python

import pascal3d
from pascal3d import utils
import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import itertools
import multiprocessing

num_cores = multiprocessing.cpu_count()
cropped_image_size = (224, 224)


def get_pascal_datasets(data_type, class_name, dataset_source):
    pascal_data_set_val = pascal3d.dataset.Pascal3DDataset(data_type,
                                                           dataset_source,
                                                           use_split_files=True,
                                                           class_only=class_name)

    return pascal_data_set_val


def create_data(p3d_data_set):

    if num_cores > 1:
        results = Parallel(n_jobs=num_cores)(delayed(process_data)(index, row['file_name'], p3d_data_set) for index, row in p3d_data_set.data_ids.iterrows())
    else:
        results = [process_data(index, row['file_name'], p3d_data_set) for index, row in p3d_data_set.data_ids.iterrows()]

    return pd.concat(results)


def process_data(index, data_set_index, p3d_data_set):
    data = p3d_data_set.get_data(data_set_index)
    if index % int(0.1 * len(p3d_data_set)) == 0:
        print('percent: %s' % int(round((100 * index / len(p3d_data_set)))))

    class_cads = data['class_cads']
    # only want to train against singular examples
    df_return = pd.DataFrame()

    for nth_object_in_file, object in enumerate(data['objects']):
        df_dict = {}
        df_dict['file_name'] = data['file_name']
        df_dict['data_source'] = data['data_source']
        df_dict['data_set'] = data['data_set']

        obj = object[1]
        df_dict['image_data'] = data['img']
        df_dict['class_name'] = object[0]
        (df_dict['bb82d'], Dx, Dy, Dz, df_dict['bb83d']) = p3d_data_set.camera_transform_cad_bb8_object(df_dict['class_name'], obj, class_cads)
        df_dict['D'] = (Dx, Dy, Dz)

        x_values, y_values = np.split(np.transpose(df_dict['bb82d']), 2)
        bb82d_x_min = np.min(x_values)
        bb82d_x_max = np.max(x_values)
        bb82d_y_min = np.min(y_values)
        bb82d_y_max = np.max(y_values)

        # does the bb8 fit within the frame
        if bb82d_x_min > 0 \
                and bb82d_y_min > 0 \
                and bb82d_x_max < df_dict['image_data'].shape[1] \
                and bb82d_y_max < df_dict['image_data'].shape[0]:

            # add ground truth camera
            df_dict['R_gt'] = utils.get_transformation_matrix(obj['viewpoint']['azimuth'],
                                                   obj['viewpoint']['elevation'],
                                                   obj['viewpoint']['distance'])
        else:
            df_dict['R_gt'] = None

        df_dict['bb8_outside'] = True if df_dict['R_gt'] is None else False

        # ['image_file_name',
        #  'class_name',
        #  'bb82d',
        #  'bb83d',
        #  'D',
        #  'gt_camera_pose',
        #  'truncated',
        df_dict['truncated'] = obj['truncated']
        #  'occluded',
        df_dict['occluded'] = obj['occluded']
        #  'nth_object_in_file'
        df_dict['nth_object_in_file'] = nth_object_in_file
        #  'image_data']
        df_return = df_return.append(df_dict, ignore_index=True)

    return df_return


def make_dataset(class_name):
    output_directory = os.path.expanduser('~/Documents/UCL/PROJECT/DATA/BB8_PASCAL_DATA/')
    set_types = list(itertools.product(['val',
                                        'train'],
                                       [pascal3d.dataset.Pascal3DDataset.dataset_source_enum.pascal,
                                        pascal3d.dataset.Pascal3DDataset.dataset_source_enum.imagenet]))

    data_sets = [get_pascal_datasets(set_demand[0], class_name, set_demand[1]) for set_demand in set_types]

    data_sets
    # create the bb8s
    all_data = pd.concat([create_data(dataset) for dataset in data_sets])
    h5_filename_proto = 'pascal3d_data_frame_{}.h5'
    h5_filename = h5_filename_proto.format(class_name)
    store = pd.HDFStore(os.path.join(output_directory, h5_filename))
    store['df'] = all_data


def main():
    make_dataset('car')


if __name__ == '__main__':
    main()
