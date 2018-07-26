#!/usr/bin/env python

import pascal3d
from pascal3d import utils
import os.path as osp
import os
import scipy.misc
import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing

num_cores = multiprocessing.cpu_count()
cropped_image_size = (224, 224)

def main():
    output_directory = os.path.expanduser('~/Documents/UCL/PROJECT/DATA/BB8_PASCAL_DATA/')
    pascal_data_set = pascal3d.dataset.Pascal3DDataset('all', pascal3d.dataset.Pascal3DDataset.dataset_source_enum.pascal)

    # get image set membership
    pascal_image_set_val_dataframe = pascal_data_set.load_image_set_files('val')
    pascal_image_set_val_dataframe['val'] = True
    #pascal_image_set_test_dataframe = pascal_data_set.load_image_set_files('test')
    pascal_image_set_train_dataframe = pascal_data_set.load_image_set_files('train')
    pascal_image_set_train_dataframe['val'] = False
    df_pascal_train_val = pd.concat([pascal_image_set_val_dataframe, pascal_image_set_train_dataframe])

    df_pascal = create_data(pascal_data_set, offset=0)
    print(df_pascal.shape)
    print(df_pascal_train_val.shape)
    print(df_pascal.columns.values)
    print(df_pascal_train_val.columns.values)

    df_pascal_merged = pd.merge(df_pascal, df_pascal_train_val, on=['file_name'], how='inner')
    print(df_pascal_merged.shape)
    df_pascal_merged['pascal'] = True
    print(df_pascal_merged.shape)
    print(df_pascal_merged.columns.values)

    imagenet_data_set = pascal3d.dataset.Pascal3DDataset('all',
                                                               pascal3d.dataset.Pascal3DDataset.dataset_source_enum.imagenet)

    imagenet_image_set_val_dataframe = pascal_data_set.load_image_set_files('val')
    imagenet_image_set_val_dataframe['val'] = True
    #imagenet_image_set_test_dataframe = pascal_data_set.load_image_set_files('test')
    imagenet_image_set_train_dataframe = pascal_data_set.load_image_set_files('train')
    imagenet_image_set_train_dataframe['train'] = False
    df_imagenet_train_val = pd.concat([imagenet_image_set_val_dataframe, imagenet_image_set_train_dataframe])

    df_imagenet = create_data(imagenet_data_set, len(pascal_data_set))

    df_imagenet_merged = pd.merge(df_imagenet, df_imagenet_train_val, on=['file_name'], how='inner')
    df_imagenet_merged['pascal'] = False

    all_data = pd.concat([df_pascal_merged, df_imagenet_merged])

    store = pd.HDFStore(os.path.join(output_directory, 'pascal3d_data_frame.h5'))
    store['df'] = all_data


def create_data(data_set, offset):

    if num_cores > 1:
        results = Parallel(n_jobs=num_cores)(delayed(process_data)(i, offset, data_set) for i in range(len(data_set)))
    else:
        results = [process_data(i, offset, data_set) for i in range(len(data_set))]

    return pd.concat(results)


def process_data(data_set_index, offset, data_set):
    columns = ['file_name', 'class_name', '2d_bb8', '3d_bb8', 'D', 'gt_camera_pose', 'image']
    data = data_set.get_data(data_set_index)
    image_file_name = data['data_id']
    overall_id = offset + data_set_index
    if data_set_index % int(0.1 * len(data_set)) == 0:
        print('percent: %s' % int(round((100 * data_set_index / len(data_set)))))

    class_cads = data['class_cads']
    # only want to train against singular examples
    df_return = pd.DataFrame(columns=columns)
    for object in data['objects']:
        if object[1]['truncated'] or object[1]['occluded']:
            continue
        original_image = data['img']
        image_height = original_image.shape[0]
        image_width = original_image.shape[1]
        class_name = object[0]

        (bb8, Dx, Dy, Dz, bb83d) = data_set.camera_transform_cad_bb8_object(class_name, object[1], class_cads)
        x_values, y_values = np.split(np.transpose(bb8), 2)
        bb82d_x_min = np.min(x_values)
        bb82d_x_max = np.max(x_values)
        bb82d_y_min = np.min(y_values)
        bb82d_y_max = np.max(y_values)
        # does the bb8 fit within the frame
        if bb82d_x_min > 0 \
                and bb82d_y_min > 0 \
                and bb82d_x_max < image_width \
                and bb82d_y_max < image_height:



            # add ground truth camera
            obj = object[1]
            R_gt = utils.get_transformation_matrix(
                obj['viewpoint']['azimuth'],
                obj['viewpoint']['elevation'],
                obj['viewpoint']['distance'],
            )
            df_this_one = pd.DataFrame([[image_file_name,
                                         class_name,
                                         bb8,
                                         bb83d,
                                         (Dx, Dy, Dz),
                                         R_gt,
                                         original_image]], columns = columns)

            df_return = pd.concat([df_return, df_this_one])

    return df_return

if __name__ == '__main__':
    main()
