#!/usr/bin/env python

import pascal3d
from pascal3d import utils
import os
import numpy as np
import pandas as pd
import itertools
from joblib import Parallel, delayed
import multiprocessing
import cv2
import matplotlib.pyplot as plt

num_cores = 1 #multiprocessing.cpu_count()
cropped_image_size = (224, 224)
string_columns = ['file_name', 'data_source', 'data_set']
output_directory = os.path.expanduser('~/Documents/UCL/PROJECT/DATA/BB8_PASCAL_DATA/')


class Bb82d(object):
    def __init__(self, bb8_label):
        self.bb8_label = np.copy(bb8_label)

    def scale_labels(self, image_scale_factor):
        self.bb8_label *= image_scale_factor

    def make_label_int(self):
        self.bb8_label = self.bb8_label.astype(dtype=int)

    @property
    def x_values(self):
        return self.bb8_label[:, 0]

    @x_values.setter
    def x_values(self, value):
        self.bb8_label[:, 0] = value

    @property
    def y_values(self):
        return self.bb8_label[:, 1]

    @y_values.setter
    def y_values(self, value):
        self.bb8_label[:, 1] = value

    @property
    def x_min(self):
        return np.min(self.x_values)

    @property
    def y_min(self):
        return np.min(self.y_values)

    @property
    def x_max(self):
        return np.max(self.x_values)

    @property
    def y_max(self):
        return np.max(self.y_values)

    @property
    def x_mid(self):
        return 0.5 * (self.x_max + self.x_min)

    @property
    def y_mid(self):
        return 0.5 * (self.y_max + self.y_min)

    @property
    def height(self):
        return self.y_max - self.y_min

    @property
    def width(self):
        return self.x_max - self.x_min

    @property
    def front_face(self):
        return self.bb8_label[0:4]

    @property
    def back_face(self):
        return self.bb8_label[4:8]

    def limits(self):
        return self.x_min, self.y_min, self.x_max, self.y_max

    @property
    def box_limits(self):
        box = np.array(
            [[self.x_max, self.y_min],
             [self.x_max, self.y_max],
             [self.x_min, self.y_max],
             [self.x_min, self.y_min]])
        return box

    def offset_labels(self, x_offset, y_offset):
        self.x_values += x_offset
        self.y_values += y_offset


line_colour = (0, 255, 0)  # green
front_square_colour = (0, 0, 255)  # red
inferred_front_square_colour = (255, 0, 0)  # bloo
inferred_front_line_colour = (255, 255, 0)


def show_image(image):
    image_copy = np.copy(image)
    plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
    plt.show()


def draw_bb82d_on_image(bb82d_label,
                        image,
                        front_face_colour=front_square_colour,
                        line_colour=line_colour):
    inted_bb8 = Bb82d(bb82d_label)
    inted_bb8.make_label_int()
    copy_image = np.copy(image)
    # front and back faces
    cv2.polylines(copy_image, [inted_bb8.front_face], True, front_face_colour, thickness=1)
    cv2.polylines(copy_image, [inted_bb8.back_face], True, line_colour, thickness=1)

    # side lines
    for line_number in range(4):
        pt1 = inted_bb8.front_face[line_number]
        pt2 = inted_bb8.back_face[line_number]
        cv2.line(copy_image, (pt1[0], pt1[1]), (pt2[0], pt2[1]), line_colour, thickness=1)

    return copy_image

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
            (df_dict['bb82d'], Dx, Dy, Dz, df_dict['bb83d'], df_dict['viewpoint']) = p3d_data_set.camera_transform_cad_bb8_object(df_dict['class_name'], obj, class_cads)
            df_dict['D'] = (Dx, Dy, Dz)

            # image = df_dict['image_data']
            # show_image(draw_bb82d_on_image(df_dict['bb82d'], image))

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
                df_dict['truncated'] = obj['truncated']
            else:
                df_dict['truncated'] = True
                df_dict['R_gt'] = np.zeros(shape=(4, 4), dtype=np.float64)

            assert df_dict['R_gt'].shape == (4, 4) and df_dict['R_gt'].dtype == np.float64, "wrong trousers"

            df_dict['camera_intrinsic'] = utils.make_camera_matrix(df_dict['viewpoint'])

            df_dict['bb8_outside'] = not np.any(df_dict['R_gt'])
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
    #df_to_hdf(all_data, class_name)
    df_to_pickle(all_data, class_name)


def main():
    classes = ['aeroplane',
    'bicycle',
    'boat',
    'bottle',
    'bus',
    'chair',
    'diningtable',
    'motorbike',
    'sofa',
    'train',
    'tvmonitor']
    for class_name in classes:
        make_dataset(class_name)


if __name__ == '__main__':
    main()
