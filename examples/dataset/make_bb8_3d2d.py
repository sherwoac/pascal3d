#!/usr/bin/env python

import pascal3d
import os.path as osp
import os
import scipy.misc
import numpy as np


def main():
    data_type = 'all'
    dataset = pascal3d.dataset.Pascal3DDataset(data_type, pascal3d.dataset.Pascal3DDataset.dataset_source_enum.imagenet)
    output_directory = os.path.expanduser('~/Documents/UCL/PROJECT/DATA/BB8_PASCAL_DATA')
    bb8_dict_file = osp.join(output_directory, 'bb8_points')
    image_file_type = '.jpg'
    bb8_points = {}
    for i in range(len(dataset)):
        dataset.show_bb8(i)

        data = dataset.get_data(i)
        # only want to train against singular examples
        if len(data['objects']) == 1:
            img1 = data['img']
            class_dir = data['objects'][0][0]
            output_image_filename = osp.join(output_directory, class_dir, class_dir +'_' + '{0:03d}'.format(i) + image_file_type)
            bb8s = dataset.camera_transform_cad_bb8(data)
            assert len(bb8s) == 1, 'more than one bb8?'
            bb8 = bb8s[0]
            x_values, y_values = np.split(np.transpose(bb8), 2)

            if np.min(x_values) > 0 \
                and np.min(y_values) > 0 \
                and np.max(x_values) < img1.shape[1] \
                and np.max(y_values) < img1.shape[0]:
                bb8_points[i] = bb8

                dir_name = osp.dirname(output_image_filename)
                if not osp.isdir(dir_name): #
                    os.makedirs(dir_name)

                scipy.misc.imsave(output_image_filename, img1)

    np.save(bb8_dict_file, bb8_points)

if __name__ == '__main__':
    main()
