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
data_columns = ['id', 'file_name', 'test', '2d_bb8', '3d_bb8', 'D', 'gt_camera_pose']

def main():
    output_directory = os.path.expanduser('~/Documents/UCL/PROJECT/DATA/BB8_PASCAL_DATA/')
    test_output_directory = os.path.join(output_directory, 'TEST')
    test_data_set = pascal3d.dataset.Pascal3DDataset('val', pascal3d.dataset.Pascal3DDataset.dataset_source_enum.pascal)
    df_test = create_data(test_data_set, test_output_directory, 0, is_test=True)

    train_output_directory = os.path.join(output_directory, 'TRAIN')
    train_pascal_data_set = pascal3d.dataset.Pascal3DDataset('train', pascal3d.dataset.Pascal3DDataset.dataset_source_enum.pascal)
    df_train1 = create_data(train_pascal_data_set,
                            train_output_directory,
                            len(test_data_set),
                            is_test=False)
    train_imagenet_data_set = pascal3d.dataset.Pascal3DDataset('all',
                                                               pascal3d.dataset.Pascal3DDataset.dataset_source_enum.imagenet)
    df_train2 = create_data(train_imagenet_data_set,
                            train_output_directory,
                            len(test_data_set) + len(train_imagenet_data_set),
                            is_test=False)

    all_data = pd.concat([df_test, df_train1, df_train2])
    store = pd.HDFStore(os.path.join(output_directory, 'pascal3d_data_frame.h5'))
    store['df'] = all_data


def create_data(data_set, output_directory, offset, is_test):

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    if num_cores > 1:
        results = Parallel(n_jobs=num_cores)(delayed(process_data)(i, offset, data_set, output_directory, is_test) for i in range(len(data_set)))
    else:
        results = [process_data(i, offset, data_set, output_directory, is_test) for i in range(len(data_set))]

    return pd.concat(results)


def process_data(data_set_index, offset, data_set, output_directory, is_test, image_file_type=".jpg"):
    data = data_set.get_data(data_set_index)
    overall_id = offset + data_set_index
    if data_set_index % int(0.1 * len(data_set)) == 0:
        print('percent: %s' % int(round((100 * data_set_index / len(data_set)))))

    class_cads = data['class_cads']
    # only want to train against singular examples
    for object in data['objects']:
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

            if is_test:
                bb_x_min, bb_y_min, bb_x_max, bb_y_max = object[1]['bbox']
                if bb_x_min > bb82d_x_min or \
                        bb_y_min > bb82d_y_min or \
                        bb_x_max < bb82d_x_max or \
                        bb_y_max < bb82d_y_max:
                    print('bb8 outside 2d bb class: {} id: {}'.format(class_name, overall_id))

                cropped_image = original_image[bb_y_min:bb_y_max+1, bb_x_min:bb_x_max+1, :]
                resized_image = cv2.resize(cropped_image, cropped_image_size)
                scipy.misc.imsave(output_image_filename, resized_image)
            else:
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
                                  is_test,
                                  bb8,
                                  bb83d,
                                  (Dx, Dy, Dz),
                                  R_gt]], columns=data_columns)

    return None


if __name__ == '__main__':
    main()
    print(illegal_bb_count)
