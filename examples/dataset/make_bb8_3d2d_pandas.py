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
string_columns = ['file_name', 'data_source', 'data_set']
output_directory = os.path.expanduser('~/Documents/UCL/PROJECT/DATA/BB8_PASCAL_DATA/')


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
        if object[0] == p3d_data_set.class_only:
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
                df_dict['R_gt'] = np.zeros(shape=(4, 4), dtype=np.float64)

            assert df_dict['R_gt'].shape == (4, 4) and df_dict['R_gt'].dtype == np.float64, "wrong trousers"

            df_dict['bb8_outside'] = not np.any(df_dict['R_gt'])
            df_dict['truncated'] = obj['truncated']
            df_dict['occluded'] = obj['occluded']
            df_dict['nth_object_in_file'] = nth_object_in_file
            df_temp = pd.DataFrame(columns=df_dict.keys())
            df_temp = df_temp.append(df_dict, ignore_index=True)
            for column in string_columns:
                df_temp[column] = df_temp[column].astype('str')
            df_return = df_return.append(df_temp, ignore_index=True)

    return df_return


def df_to_pickle(df, class_name):
    pkl_filename_proto = 'pascal3d_data_frame_{}.pkl'
    pkl_filename = pkl_filename_proto.format(class_name)
    pkl_filename = os.path.join(output_directory, pkl_filename)

    if os.path.exists(pkl_filename):
        os.remove(pkl_filename)
        print('removed:{}'.format(pkl_filename))

    df.to_pickle(pkl_filename)


def df_to_hdf(df, class_name):
    h5_filename_proto = 'pascal3d_data_frame_{}.h5'
    h5_filename = h5_filename_proto.format(class_name)
    h5_filename = os.path.join(output_directory, h5_filename)

    if os.path.exists(h5_filename):
        os.remove(h5_filename)
        print('removed:{}'.format(h5_filename))

    object_data_columns = []
    for key in df.keys():
        print(key, df[key].dtype)
        if df[key].dtype == type(object):
            object_data_columns.append(key)

    hdf_store = pd.HDFStore(h5_filename)
    hdf_store.put('df', df, format='table', data_columns=object_data_columns)
    hdf_store.close()


def make_dataset(class_name):

    set_types = list(itertools.product(['val',
                                        'train'],
                                       [pascal3d.dataset.Pascal3DDataset.dataset_source_enum.imagenet,
                                        pascal3d.dataset.Pascal3DDataset.dataset_source_enum.pascal]))

    #set_types = [['val', pascal3d.dataset.Pascal3DDataset.dataset_source_enum.imagenet]]
    data_sets = [get_pascal_datasets(set_demand[0], class_name, set_demand[1]) for set_demand in set_types]

    # create the bb8s
    all_data = pd.concat([create_data(dataset) for dataset in data_sets])
    all_data['class_name'].astype(str)
    df_to_hdf(all_data, class_name)
    df_to_pickle(all_data, class_name)


def main():
    make_dataset('car')


if __name__ == '__main__':
    main()
