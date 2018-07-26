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
data_columns = ['id', 'file_name', '2d_bb8', '3d_bb8', 'D', 'gt_camera_pose']

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

    df_pascal = create_data(pascal_data_set,
                            output_directory,
                            offset=0)

    df_pascal = df_pascal.merge(df_pascal_train_val, left_on='file_name', right_on='file_name')
    df_pascal = df_pascal['pascal'] = True

    imagenet_data_set = pascal3d.dataset.Pascal3DDataset('all',
                                                               pascal3d.dataset.Pascal3DDataset.dataset_source_enum.imagenet)

    imagenet_image_set_val_dataframe = pascal_data_set.load_image_set_files('val')
    imagenet_image_set_val_dataframe['val'] = True
    #imagenet_image_set_test_dataframe = pascal_data_set.load_image_set_files('test')
    imagenet_image_set_train_dataframe = pascal_data_set.load_image_set_files('train')
    imagenet_image_set_train_dataframe['train'] = False
    df_imagenet_train_val = pd.concat([imagenet_image_set_val_dataframe, imagenet_image_set_train_dataframe])

    df_imagenet = create_data(imagenet_data_set,
                              output_directory,
                              len(pascal_data_set))

    df_imagenet = df_imagenet.merge(df_imagenet_train_val, left_on='file_name', right_on='file_name')
    df_imagenet['pascal'] = False
    all_data = pd.concat([df_pascal, df_imagenet])

    store = pd.HDFStore(os.path.join(output_directory, 'pascal3d_data_frame.h5'))
    store['df'] = all_data


def create_data(data_set, output_directory, offset):

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    if num_cores > 1:
        results = Parallel(n_jobs=num_cores)(delayed(process_data)(i, offset, data_set, output_directory) for i in range(len(data_set)))
    else:
        results = [process_data(i, offset, data_set, output_directory) for i in range(len(data_set))]

    return pd.concat(results)


def process_data(data_set_index, offset, data_set, output_directory, image_file_type=".jpg"):
    data = data_set.get_data(data_set_index)
    overall_id = offset + data_set_index
    if data_set_index % int(0.1 * len(data_set)) == 0:
        print('percent: %s' % int(round((100 * data_set_index / len(data_set)))))

    class_cads = data['class_cads']
    # only want to train against singular examples
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

            output_image_filename = osp.join(output_directory,
                                             class_name,
                                             class_name + '_' + '{0:05d}'.format(overall_id) + image_file_type)

            dir_name = osp.dirname(output_image_filename)
            if not osp.isdir(dir_name):
                os.makedirs(dir_name)

            scipy.misc.imsave(output_image_filename, original_image)

            # add ground truth camera
            obj = object[1]
            R_gt = utils.get_transformation_matrix(
                obj['viewpoint']['azimuth'],
                obj['viewpoint']['elevation'],
                obj['viewpoint']['distance'],
            )

            return pd.DataFrame([[overall_id,
                                  output_image_filename,
                                  bb8,
                                  bb83d,
                                  (Dx, Dy, Dz),
                                  R_gt]], columns=['id', 'file_name', '2d_bb8', '3d_bb8', 'D', 'gt_camera_pose'])

    return None


if __name__ == '__main__':
    main()
