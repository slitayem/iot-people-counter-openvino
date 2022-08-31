"""
Images related helper functions module
@author: Saloua Litayem
"""
import os
import sys
import logging as log

import cv2
from webcolors import name_to_rgb
import numpy as np

import utils


def get_labels_mapping():
    """load coco labels mapping"""
    # https://github.com/openvinotoolkit/open_model_zoo/blob/master/data/dataset_classes/coco_91cl_bkgr.txt
    labels = dict()
    with open(
        os.path.join(utils.get_script_folder(), "coco_91cl_bkgr.txt")) as file_:
        for index, line in enumerate(file_):
            labels[index] = line.rstrip()
    return labels

LABELS_MAPPING = get_labels_mapping()

def preprocess(image, input_shape):
    """
    Preprocess frame image
    :param frame: numpy.ndarray image
    :param: image shape
    """
    batch_size, channels, height, width = input_shape
    image_ = cv2.resize(image, (width, height))
    # Change data layout to have channels first: H, W, C to C, H, W
    image_ = image_.transpose((2, 0, 1))
    image_ = image_[np.newaxis, ...]
    return image_


def write_text(image, text, xpos, ypos, color='green'):
    """
    Write text message on an image
    """
    cv2.putText(
        image, text, (xpos, ypos),
        cv2.FONT_HERSHEY_COMPLEX, 0.5, name_to_rgb(color), 1)


def bufferize(image, width, height):
    """write a frame image to the stdout
    to be piped out to the ffmpeg server
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sys.stdout.buffer.write(image.tobytes())
    sys.stdout.flush()


def draw_bbox(image, xmin, ymin, xmax, ymax, str_label="", color="blue"):
    """
    Draw a bounding box given its coordinates
    """
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax),
        name_to_rgb('green'), 2
    )
    if str_label:
        _loc = 10
        _y = ymax - _loc if ymax > 2 * _loc else ymax + _loc
        cv2.putText(
            image,
            f"{str_label}",
            (xmin, _y),
            cv2.FONT_HERSHEY_COMPLEX,
            fontScale=0.5,
            color=name_to_rgb(color),
            thickness=1,
        )
    return image


def process_ssd_output(image: np.array, result: list, prob_threshold: float,
    width: int, height: int, label_id: int=1):
    """
    Parse SSD(single-shot detector) output.
    :param image:
    :param result: list contains the data to parse ssd
    :param prob_threshold: min  confidence threshold
    :param widht
    :param height
    :param label_id label ID
    :return: label count and image
    """
    label_count = 0
    boxes = []
    confs = []
    # TODO return detected objects boxes sorted by (threshold and size)
    for obj in result[0][0]:
        _, label, confidence, xmin, ymin, xmax, ymax = obj
        if confidence > prob_threshold and label == label_id:
            xmin *= width
            ymin *= height
            xmax *= width
            ymax *= height
            str_label = f"{LABELS_MAPPING[label_id]}: {confidence * 100 :.2f}%"

            draw_bbox(image,
                int(xmin), int(ymin), int(xmax), int(ymax),
                str_label, color="blue")
            label_count += 1
            boxes.append((xmin, ymin, xmax, ymax))
            confs.append(confidence)
    return image, label_count, boxes, confs


def intersection_over_union(box1, box2):
    """
    Compute the IoU of two bounding boxes
    :param box1: first bounding box coordinates
    :param box2: second bounding box coordinates

    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-groundtruth
    # rectangles
    box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-groundtruth
    # areas - the interesection area
    iou = interArea / float(box1Area + box2Area - interArea)

    # return the intersection over union value
    return iou


def boxes_overlap(predicted, groundtruth, threshold=0.8):
    """
    Get the number of bounding boxes overlapping
    with an IoU higher than given threshold value
    :param boxes1: First list of bounding boxes coordinates
    :param groundgroundtruth: Second list of bounding boxes coordinates
    :return number of bounding boxes with expected overlap metric
    value
    """
    n_i = len(predicted)
    n_j = len(groundtruth)
    iou_mat = np.empty((n_i, n_j))
    nb_overlaps = 0
    for i in range(n_i):
        for j in range(n_j):
            iou = intersection_over_union(predicted[i], groundtruth[j])
            if iou > threshold:
                nb_overlaps += 1

    return nb_overlaps


def get_centroid(x1, y1, x2, y2):
    """
    Get bounding box centroid coordinates
    """
    return (x1 + x2) / 2 , (y1 + y2) / 2


def centroids_distance(c1, c2):
    """
    Distance between two centroids points
    :param c1: centroid (x, y) coordinates
    :param c2: centroid (x, y) coordinates
    :return distance: int
    """
    x1, y1 = c1
    x2, y2 = c2
    return math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))


def objects_dists(centroids1, centroids2):
    """
    Check if new centroids correspond to a new object
    id(s) in the frame
    """
    dist = []
    for index1 in enumerate(centroids1):
        for index2 in enumerate(centroids2):
            dist.append(
                centroids_distance(
                    centroids1[index1], centroids2[index2])
            )
    return dist


def get_boxes_centroids(boxes):
    """Extract centroids of bouding boxes
    :param boxes: list of bounding boxes coordinates
    :return centroids: list of centroids coordinates
    """
    centroids = []
    for index in range(len(boxes)):
        centroids.append(get_centroid(*boxes[index]))
    return centroids
