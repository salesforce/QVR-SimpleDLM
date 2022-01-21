'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

import numpy as np
from sklearn.cluster import DBSCAN

class AxisAlignedRectangle():
    def __init__(self, x1, y1, x2, y2):
        try:
            y2 >= y1 and x2 >= x1
        except:
            raise ValueError('Rectangle dimensions are incorrect')

        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def get_width(self):
        return self.x2 - self.x1

    def get_height(self):
        return self.y2 - self.y1

    def get_l1_dist(self, rect):
        outer_rect = AxisAlignedRectangle(min(self.x1, rect.x1),
                                          min(self.y1, rect.y1),
                                          max(self.x2, rect.x2),
                                          max(self.y2, rect.y2))

        inner_width = max(0, outer_rect.get_width() -
                          self.get_width() - rect.get_width())
        inner_height = max(0, outer_rect.get_height() -
                           self.get_height() - rect.get_height())

        return inner_width + inner_height


def dbscan_clustering(image_results, eps, axis_keys):
    """Takes a pandas DF of OCR results as input and does DBSCAN on bboxes"""

    X = _compute_bbox_min_l1_distance(image_results, axis_keys)
    clustering = DBSCAN(eps=eps, min_samples=1, metric='precomputed').fit(X)
    return clustering.labels_


def _compute_bbox_min_l1_distance(bboxes, axis_keys=['x1', 'y1', 'x2', 'y2']):
    '''Computes L1 distance on bboxes'''
    bbox_rects = list(bboxes.apply(lambda bbox: AxisAlignedRectangle(
        bbox[axis_keys[0]], bbox[axis_keys[1]], bbox[axis_keys[2]], bbox[axis_keys[3]]), axis=1))
    dist = np.zeros((len(bbox_rects), len(bbox_rects)), dtype=float)
    for i in range(len(bbox_rects)):
        for j in range(i + 1, len(bbox_rects)):
            dist[i, j] = bbox_rects[i].get_l1_dist(bbox_rects[j])
    return _symmetrize(dist)


def _symmetrize(a):
    return a + a.T - np.diag(a.diagonal())


def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)


def _compute_modified_iou_along_axis(bboxes, axis_start, axis_end):
    dist = np.zeros((len(bboxes), len(bboxes)), dtype=float)
    end_series = bboxes[axis_end]
    start_series = bboxes[axis_start]
    for i in range(len(bboxes)):
        mid_point_i = (start_series.iloc[i] + end_series.iloc[i]) / 2
        for j in range(i + 1, len(bboxes)):
            intersection = max(0, min(
                end_series.iloc[i], end_series.iloc[j]) -
                               max(start_series.iloc[i], start_series.iloc[j]))
            union = (end_series.iloc[i] - start_series.iloc[i]) + \
                    (end_series.iloc[j] - start_series.iloc[j]) - \
                    intersection
            mid_point_j = (start_series.iloc[j] + end_series.iloc[j]) / 2
            mid_point_distance = abs(mid_point_i - mid_point_j)
            # if no overlap, dist = union(a, b) + l1 distance between boxes
            # if perfect overlap, dist = 0
            # if somewhere in the middle, higher values indicate lower overlap
            dist[i][j] = union - intersection + mid_point_distance
    return _symmetrize(dist)


def group_bbox_by_axis_dbscan(bboxes, axis):
    axis_start_dim, axis_end_dim, axis_out = (
        'x1', 'x2', 'col_id') if axis == 0 else ('y1', 'y2', 'row_id')
    eps = mean(bboxes[axis_end_dim] - bboxes[axis_start_dim])
    X = _compute_modified_iou_along_axis(bboxes, axis_start_dim, axis_end_dim)
    clustering = DBSCAN(eps=eps, min_samples=1, metric='precomputed').fit(X)
    bboxes[axis_out] = clustering.labels_
    return bboxes

def intersection_over_batch_word_area(bboxes1, bboxes2):
    """
        bboxes1: numpy array of word boxes, N X 4
        bboxes2: numpy array of block boxes, N X 4
    """
    if len(bboxes1.shape) < 2:
        bboxes1 = np.expand_dims(bboxes1, axis=0)
    if len(bboxes2.shape) < 2:
        bboxes2 = np.expand_dims(bboxes2, axis=0)
    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))

    # compute the area of intersection rectangle
    interArea = np.maximum((xB - xA + 1), 0) * \
                    np.maximum((yB - yA + 1), 0)

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)

    iou = interArea / boxAArea
    return iou
