#!/usr/bin/env python
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

import pascal3d

model_names = ['InceptionV3', 'MobileNet', 'Resnet50Trainable', 'Vgg19Trainable', 'Xception']
def main():
    cmap = plt.get_cmap('jet_r')
    data_type = 'val'
    for index, model_name in enumerate(model_names):
        #boat 2011_002854_1 2009_000516_2 197 85
        colour = cmap(float(index) / len(model_names))
        dataset = pascal3d.dataset.Pascal3DDataset(data_type, use_split_file=True, class_name='boat')
        dataset.show_cad_overlay_inferred('2011_002854', 197, 'boat', model_name, colour)
        dataset.show_cad_overlay_inferred('2009_000516', 85, 'boat', model_name, colour)

        #car 2011_002536_0 2010_005860_1 230 205
        dataset_car = pascal3d.dataset.Pascal3DDataset(data_type, use_split_file=True, class_name='car')
        dataset_car.show_cad_overlay_inferred('2011_002536', 230, 'car', model_name, colour)
        dataset_car.show_cad_overlay_inferred('2010_005860', 205, 'car', model_name, colour)

if __name__ == '__main__':
    main()
