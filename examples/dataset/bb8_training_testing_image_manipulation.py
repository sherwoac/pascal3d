import numpy as np
import math

def get_ideal_image_dimensions(image, bb82d, iou_overlap):
    image_height, image_width = image[0], image[1]
    first_dimension = min(math.sqrt(image_height * image_width / iou_overlap), max(image_height, image_width))
    return

def get_bounding_box_dimensions(bb8_label):
    """returns bounding box dimensions for a given set of bb8 labels"""

    bb8_label_xs = bb8_label[:16:2]
    bb8_label_ys = bb8_label[1:16:2]

    x_min = np.min(bb8_label_xs)
    y_min = np.min(bb8_label_ys)

    # max
    x_max = np.max(bb8_label_xs)
    y_max = np.max(bb8_label_ys)
    return x_min, y_min, x_max, y_max