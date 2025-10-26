# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Evaluate performance of object detection
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# general package imports
import numpy as np
import matplotlib
#matplotlib.use('wxagg') # change backend so that figure maximizing works on Mac as well     
import matplotlib.pyplot as plt

import zlib
import torch
from shapely.geometry import Polygon
from operator import itemgetter

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# object detection tools and helper functions
import misc.objdet_tools as tools


# compute various performance measures to assess object detection
def measure_detection_performance(detections, labels, labels_valid, min_iou=0.5):

    # find best detection for each valid label
    true_positives = 0  # no. of correctly detected objects
    center_devs = []
    ious = []
    
    # Create a set of detection indices that have been matched to a label
    matched_detection_indices = set()

    for label, valid in zip(labels, labels_valid):
        matches_lab_det = []
        if valid:  # exclude all labels from statistics which are not considered valid

            # Step 1: Extract the four corners of the current label bounding-box
            lab_corners = tools.compute_box_corners(label.box.center_x, label.box.center_y, label.box.width, label.box.length, label.box.heading)
            lab_poly = Polygon(lab_corners)

            # Step 2: Loop over all detected objects
            for det_idx, det in enumerate(detections):
                # det format: [class_id, x, y, z, h, w, l, yaw]
                _ , x, y, z, h, w, l, yaw = det

                # Step 3: Extract the four corners of the current detection
                det_corners = tools.compute_box_corners(x, y, w, l, yaw)
                det_poly = Polygon(det_corners)

                # Step 4: Compute the center distance
                dist_x = label.box.center_x - x
                dist_y = label.box.center_y - y
                dist_z = label.box.center_z - z

                # Step 5: Compute the intersection over union (IOU)
                intersection = lab_poly.intersection(det_poly).area
                union = lab_poly.area + det_poly.area - intersection
                iou = intersection / union if union > 0 else 0

                # Step 6: If IOU exceeds min_iou threshold, store match
                if iou > min_iou:
                    matches_lab_det.append([iou, dist_x, dist_y, dist_z, det_idx])

        # Find best match and compute metrics for the current label
        if matches_lab_det:
            # Sort matches by IoU in descending order
            best_match = max(matches_lab_det, key=itemgetter(0))
            
            # Add the index of the matched detection to our set
            matched_detection_indices.add(best_match[4])
            
            ious.append(best_match[0])
            center_devs.append(best_match[1:4]) # Correctly slice the 3 distance values

    # A True Positive is a unique detection that matches a valid label
    true_positives = len(matched_detection_indices)
    
    # --- Performance Metrics (ID_S4_EX2) ---
    all_positives = np.sum(labels_valid)
    false_negatives = all_positives - true_positives
    false_positives = len(detections) - true_positives
    
    pos_negs = [all_positives, true_positives, false_negatives, false_positives]
    det_performance = [ious, center_devs, pos_negs]
    
    return det_performance


    ####### ID_S4_EX2 START #######     
    #######
    print("student task ID_S4_EX2")
    
    ## step 1 : compute the total number of positives present in the scene (i.e., all valid labels)
    all_positives = np.sum(labels_valid)

    ## step 2 : compute the number of false negatives (valid labels that were not detected)
    false_negatives = all_positives - true_positives

    ## step 3 : compute the number of false positives (detections that couldn't be matched to any valid label)
    false_positives = len(detections) - true_positives
    
    #######
    ####### ID_S4_EX2 END #######     
    
    pos_negs = [all_positives, true_positives, false_negatives, false_positives]
    det_performance = [ious, center_devs, pos_negs]
    
    return det_performance


# evaluate object detection performance based on all frames
def compute_performance_stats(det_performance_all, configs_det):

    # extract elements
    ious = []
    center_devs = []
    pos_negs = []
    for item in det_performance_all:
        ious.append(item[0])
        center_devs.append(item[1])
        pos_negs.append(item[2])
    
    ####### ID_S4_EX3 START #######     
    #######    
    print('student task ID_S4_EX3')

    ## step 1 : extract the total number of positives, true positives, false negatives and false positives
    pos_negs_arr = np.asarray(pos_negs)
    total_positives = np.sum(pos_negs_arr[:, 0])
    total_true_positives = np.sum(pos_negs_arr[:, 1])
    total_false_negatives = np.sum(pos_negs_arr[:, 2])
    total_false_positives = np.sum(pos_negs_arr[:, 3])

    ## step 2 : compute precision
    precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0

    ## step 3 : compute recall 
    recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0

    #######    
    ####### ID_S4_EX3 END #######     
    print('precision = ' + str(precision) + ", recall = " + str(recall))   

    # serialize intersection-over-union and deviations in x,y,z
    ious_all = [element for tupl in ious for element in tupl]
    devs_x_all = []
    devs_y_all = []
    devs_z_all = []
    for tuple in center_devs:
        for elem in tuple:
            dev_x, dev_y, dev_z = elem
            devs_x_all.append(dev_x)
            devs_y_all.append(dev_y)
            devs_z_all.append(dev_z)
    

    # compute statistics
    stdev__ious = np.std(ious_all)
    mean__ious = np.mean(ious_all)

    stdev__devx = np.std(devs_x_all)
    mean__devx = np.mean(devs_x_all)

    stdev__devy = np.std(devs_y_all)
    mean__devy = np.mean(devs_y_all)

    stdev__devz = np.std(devs_z_all)
    mean__devz = np.mean(devs_z_all)
    #std_dev_x = np.std(devs_x)

    # plot results
    data = [precision, recall, ious_all, devs_x_all, devs_y_all, devs_z_all]
    #titles = ['detection precision', 'detection recall', 'intersection over union', 'position errors in X', 'position errors in Y', 'position error in Z']
    textboxes = ['', '', '',
                 '\n'.join((r'$\mathrm{mean}=%.4f$' % (np.mean(devs_x_all), ), r'$\mathrm{sigma}=%.4f$' % (np.std(devs_x_all), ), r'$\mathrm{n}=%.0f$' % (len(devs_x_all), ))),
                 '\n'.join((r'$\mathrm{mean}=%.4f$' % (np.mean(devs_y_all), ), r'$\mathrm{sigma}=%.4f$' % (np.std(devs_y_all), ), r'$\mathrm{n}=%.0f$' % (len(devs_x_all), ))),
                 '\n'.join((r'$\mathrm{mean}=%.4f$' % (np.mean(devs_z_all), ), r'$\mathrm{sigma}=%.4f$' % (np.std(devs_z_all), ), r'$\mathrm{n}=%.0f$' % (len(devs_x_all), )))]
    model_name = configs_det.arch
    conf_thresh = configs_det.conf_thresh
    titles = [
        f'Precision\n(Model: {model_name}, Conf: {conf_thresh})', 
        f'Recall\n(Model: {model_name}, Conf: {conf_thresh})',
        'Intersection over Union', 
        'Position Errors in X', 
        'Position Errors in Y', 
        'Position Error in Z'
    ]

    f, a = plt.subplots(2, 3)
    a = a.ravel()
    num_bins = 20
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    for idx, ax in enumerate(a):
        ax.hist(data[idx], num_bins)
        ax.set_title(titles[idx])
        if textboxes[idx]:
            ax.text(0.05, 0.95, textboxes[idx], transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
    plt.tight_layout()
    plt.show()

