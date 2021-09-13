"""
Utility functions for HW4 of CSE512 Spring 2021

By: Minh Hoai Nguyen (minhhoai@cs.stonybrook.edu)
Adapted from original code by: Aggelina Chatziagapi
Created: 4-Apr-2021
Last modified: 4-Apr-2021
"""


import os
import argparse
import numpy as np
import cv2
import random
import xml.etree.ElementTree as ET
import tqdm
import time

from mean_average_precision import MeanAveragePrecision
from detect import prepare_second_stream, inference_second_stream


image_size = (256, 256)
DATA_DIR = "ContactHands/"

images_pool = DATA_DIR + "JPEGImages/"
annotations_pool = DATA_DIR + "Annotations/"


def resize_boxes(boxes, old_h, old_w, new_h, new_w):
    boxes[:, 0] = boxes[:, 0] * new_h / old_h
    boxes[:, 1] = boxes[:, 1] * new_w / old_w
    boxes[:, 2] = boxes[:, 2] * new_h / old_h
    boxes[:, 3] = boxes[:, 3] * new_w / old_w
    boxes[:,0:4] = np.round(boxes[:,0:4])

    return boxes


def resize_im_boxes(img, boxes, new_h, new_w):
    old_h, old_w = img.shape[0], img.shape[1]
    img = cv2.resize(img, (new_h, new_w))
    if boxes is not None:
        boxes = np.array(boxes)
        boxes = resize_boxes(boxes, old_h, old_w, new_h, new_w)

    return img, boxes


def read_content(xml_file: str):
    """ Read annotations from xml """

    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []
    filename = root.find("filename").text

    for boxes in root.iter("object"):
        for box in boxes.findall("bndbox"):
            xmin = int(box.find("xmin").text)
            xmax = int(box.find("xmax").text)
            ymin = int(box.find("ymin").text)
            ymax = int(box.find("ymax").text)

        left = int(np.max([xmin, 0]))
        right = int(xmax)
        top = int(np.max([ymin, 0]))
        bottom = int(ymax)
        list_with_single_boxes = [left, top, right, bottom]
        list_with_all_boxes.append(list_with_single_boxes)

    list_with_all_boxes = np.array(list_with_all_boxes)

    return filename, list_with_all_boxes


def sliding_window(img_size, window_size, step_size):
    y, x = np.mgrid[0:img_size[0] - window_size[0] + 1:step_size[0],
                    0:img_size[1] - window_size[1] + 1:step_size[1]]
    y, x = y.reshape(-1, 1), x.reshape(-1, 1)
    boxes = np.concatenate((x, y, x + window_size[0] - 1, y + window_size[1] - 1), axis=1)

    return boxes


def get_proposals(im_h, im_w, min_h_ratio=1/20, max_h_ratio=1/4, aspect_ratios=[np.exp(-0.5), 1, np.exp(0.5)], scale_step=1.2, stride_ratio=0.3):
    """
        Inputs:
            im_h, im_w: height and width of the image
            min_h_ratio, max_h_ratio: minimum and maximum ratio for the height of a proposal box
            aspect_ratios: aspect ratios width:height of the proposal box to consider
            scale_step: the increment in scale
            stride_ratio: stride ratio between two overlapping box, stride_ratio=0.3 means the two consecutive boxes o
                overlap 30% in the width dimension.
        Outputs:
            boxes: n*4, where n is the number of proposals, each row is: left, top, right, bottom
    """

    cur_ratio = min_h_ratio
    all_boxes = None
    while True:
        if cur_ratio > max_h_ratio:
            break

        box_h = int(round(cur_ratio*im_h))
        stride_h = int(round(box_h*stride_ratio))
        for ar in aspect_ratios:
            box_w = int(round(box_h*ar))
            stride_w = int(round(box_w * stride_ratio))
            boxes = sliding_window((im_h, im_w), (box_h, box_w), (stride_h, stride_w))

            if all_boxes is None:
                all_boxes = boxes
            else:
                all_boxes = np.concatenate((all_boxes, boxes), axis= 0)
                # print("box_h {}, box_w {}".format(box_h, box_w))
                # print(boxes.shape)

        cur_ratio *= scale_step

    return all_boxes


def get_pos_and_random_neg(feat_extractor, dataset, num_img=-1, num_neg_per_img=5):
    """
    Helper function to get training data
    Positive data is feature vectors for annotated hands
    Negative data is feature vectors for random image patches
    Inputs:
        dataset: either 'train' or 'test'
    Outputs:
        D: (n, d) data matrix, each row is a feature vector
        lb: (n,) label vector, entries are 1 or -1
    """
    np.random.seed(0)

    dataset_file = DATA_DIR + "ImageSets/Main/{}.txt".format(dataset)
    with open(dataset_file, "r") as f:
        dataset = f.read().splitlines()
    dataset = (dataset if num_img == -1 else dataset[:num_img])

    D, lb = None, None

    for ann_name in tqdm.tqdm(dataset):
        # Get annotation and corresponding image
        xml_path = os.path.join(annotations_pool, ann_name + ".xml")
        image_file, boxes = read_content(xml_path)
        image_path = os.path.join(images_pool, image_file)
        # If image in dataset (train or test)
        if image_file.split(".jpg")[0] in dataset:
            # Read image
            img = cv2.imread(image_path)
            img, boxes = resize_im_boxes(img, boxes, image_size[0], image_size[1])

            propoal_bboxes = get_proposals(img.shape[0], img.shape[1])
            ## Remove rects that overlap more than 30% with an annotated hand
            for box in boxes:
                iou = get_iou(box, propoal_bboxes)
                propoal_bboxes = propoal_bboxes[iou < 0.3]

                if propoal_bboxes.shape[0] == 0:
                    break

            lb_i = np.ones(boxes.shape[0])
            if propoal_bboxes.shape[0] > num_neg_per_img:
                rand_proposals = propoal_bboxes[np.random.choice(propoal_bboxes.shape[0], num_neg_per_img, replace=False)]
            else:
                rand_proposals = propoal_bboxes

            boxes = np.concatenate((boxes, rand_proposals), axis=0)
            lb_i = np.concatenate((lb_i, -np.ones(rand_proposals.shape[0])))

            # Extract features
            inference_second_stream(feat_extractor, img)
            feats = feat_extractor.extract_features(boxes)
            feats = feats.detach().to('cpu').numpy()

            if D is None:
                D = feats
                lb = lb_i
            else:
                D = np.concatenate((D, feats), axis=0)
                lb = np.concatenate((lb, lb_i))

    # Shuffle data
    ind = np.random.permutation(D.shape[0])
    D = D[ind]
    lb = lb[ind]

    return D, lb


def detect(img, feat_extractor, svm_model, batch_size=-1):
    """
    Perform sliding window detection with SVM.
    Return a list of rectangular regions with scores.
    Return rects: (n, 5) where rects[i, :] corresponds to
                  left, top, right, bottom, detection score
    """
    proposal_boxes = get_proposals(img.shape[0], img.shape[1])
    n = proposal_boxes.shape[0]
    batch_size = (n if batch_size == -1 else batch_size)
    scores = None

    inference_second_stream(feat_extractor, img)

    for i in range(0, n, batch_size):
        # start_time = time.time()
        D = feat_extractor.extract_features(proposal_boxes[i:i+batch_size, :])
        D = D.detach().to('cpu').numpy()
        # print("Feature extraction for {} proposals takes {} seconds".format(proposal_boxes.shape[0], time.time() - start_time))

        # start_time = time.time()
        if scores is None:
            scores = svm_model.decision_function(D)
        else:
            scores = np.concatenate((scores, svm_model.decision_function(D)),
                                    axis=0)
        # print("SVM scoring takes {} seconds".format(time.time() - start_time))

    rects = np.concatenate((proposal_boxes, np.expand_dims(scores, axis=1)), axis=1)

    # Non maximum suppression
    # start_time = time.time()
    rects = nms(rects, overlap_thresh=0.5)
    # print("NMS takes {} seconds".format(time.time() - start_time))

    return rects


def get_iou(ref_rect, rects):
    """Get the iou between a set of rects and a referrence rect
    Inputs:
        rects: n*4 rectangles, each row is [left, top, right, bottom]
        ref_rect: 1*4 rectangle,
    Output:
        a vector of size n for the IoU
    """

    x1 = np.maximum(ref_rect[0], rects[:, 0])
    y1 = np.maximum(ref_rect[1], rects[:, 1])
    x2 = np.minimum(ref_rect[2], rects[:, 2])
    y2 = np.minimum(ref_rect[3], rects[:, 3])

    w = np.maximum(0, x2 - x1 + 1)
    h = np.maximum(0, y2 - y1 + 1)
    inter_area = w*h
    rects_area = (rects[:,2] - rects[:, 0] + 1)*(rects[:,3] - rects[:, 1] + 1)
    ref_rect_area = (ref_rect[2] - ref_rect[0] + 1)*(ref_rect[3] - ref_rect[1] + 1)
    iou = inter_area/(rects_area + ref_rect_area - inter_area)

    return iou


def nms(rects, overlap_thresh):
    """
    Non-maximum suppression.
    Greedily select high-scoring detections and skip detections
    that are significantly covered by a previously selected detection.
    """
    if len(rects) == 0:
        return []

    x1 = rects[:, 0]
    y1 = rects[:, 1]
    x2 = rects[:, 2]
    y2 = rects[:, 3]
    scores = rects[:, 4]

    # Compute the area of bounding boxes
    area = np.abs(np.maximum(x2 - x1 + 1, 0) * np.maximum(y2 - y1 + 1, 0))

    # Sort based on the scores
    ind = np.argsort(scores)

    pick = []

    while len(ind) > 0:
        # Get the last index in the indices
        last = len(ind) - 1
        i = ind[last]
        pick.append(i)
        # Find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[ind[:last]])
        yy1 = np.maximum(y1[i], y1[ind[:last]])
        xx2 = np.minimum(x2[i], x2[ind[:last]])
        yy2 = np.minimum(y2[i], y2[ind[:last]])
        # Compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # Compute the ratio of overlap
        overlap = (w * h) / area[ind[:last]]
        # delete all indexes from the index list that have
        ind = np.delete(ind, np.concatenate(
            ([last], np.where(overlap > overlap_thresh)[0])))

    return rects[pick]


def generate_result_file(feat_extractor, svm_model, dataset, num_img=-1, out_file="result.npy", shld_resize=True):
    """
    Generate detection result file for a particular dataset.
    dataset: either 'train' or 'test'
    out_file: path to a .npy file to save the result
    """
    dataset_file = DATA_DIR + "/ImageSets/Main/{}.txt".format(dataset)
    with open(dataset_file, "r") as f:
        dataset = f.read().splitlines()
    dataset = (dataset if num_img == -1 else dataset[:num_img])
    rects = {}

    for img_name in tqdm.tqdm(dataset):
        img_path = os.path.join(images_pool, img_name + ".jpg")
        if os.path.exists(img_path):
            # Read image
            img = cv2.imread(img_path)
            old_h, old_w = img.shape[0], img.shape[1]
            if shld_resize:
                img = cv2.resize(img, (image_size[0], image_size[1]))

            # Detect hands
            det_boxes = detect(img, feat_extractor, svm_model)

            if shld_resize: # resize to original size
                det_boxes = resize_boxes(det_boxes, image_size[0], image_size[1], old_h, old_w)

            rects[img_name] = det_boxes

    np.save(out_file, rects)
    print("Results have been saved to {}".format(out_file))


def compute_mAP(result_file="result.npy", dataset="test", num_img=-1):
    """
    Compute mean average precision.
    result_file: path to a .npy file to save the result
    dataset: either 'train' or 'test'
    """
    dataset_file = DATA_DIR + "/ImageSets/Main/{}.txt".format(dataset)
    with open(dataset_file, "r") as f:
        dataset = f.read().splitlines()
    dataset = (dataset if num_img == -1 else dataset[:num_img])

    # Initialize mAP
    metric_fn = MeanAveragePrecision(num_classes=1)

    # Load result file
    rects = np.load(result_file, allow_pickle=True).item()

    print("Loading annotations and predictions")
    for ann_name in tqdm.tqdm(dataset):
        # Get annotation
        xml_path = os.path.join(annotations_pool, ann_name + ".xml")
        _, boxes = read_content(xml_path)
        # Groundtruth should be in the format:
        # [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
        gt = np.concatenate((boxes, np.zeros((len(boxes), 3))), axis=1)
        if ann_name in rects:
            # Predictions should be in the format:
            # [xmin, ymin, xmax, ymax, class_id, confidence]
            pred = np.zeros((rects[ann_name].shape[0], 6))
            pred[:, :4] = rects[ann_name][:, :4]
            pred[:, -1] = rects[ann_name][:, -1]
            metric_fn.add(pred, gt)

    # Compute mAP
    ap = metric_fn.value(iou_thresholds=0.5)["mAP"]
    print("mAP: {}".format(ap))

    return ap


def visualize_annotated_data():
    """ Run demo 1 - visualize annotations """
    print("Press ESC to close the window. Any other key to continue")
    while True:
        anno_file = random.choice(os.listdir(annotations_pool))
        xml_path = os.path.join(annotations_pool, anno_file)
        image_file, gt_boxes = read_content(xml_path)
        image_path = os.path.join(images_pool, image_file)

        image = cv2.imread(image_path)

        for box in gt_boxes:
            image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color=(0,255,0), thickness=2)

        cv2.imshow("Visualization", image)

        k = cv2.waitKey(0) & 0xFF
        if k == 27:  # Esc key to stop
            break
        elif k == -1:  # normally -1 returned,so don't print it
            continue

    cv2.destroyAllWindows()
    return


def test_get_proposals():
    print("Press ESC to close the window. Any other key to continue")
    while True:
        anno_file = random.choice(os.listdir(annotations_pool))
        xml_path = os.path.join(annotations_pool, anno_file)
        image_file, gt_boxes = read_content(xml_path)
        image_path = os.path.join(images_pool, image_file)

        image = cv2.imread(image_path)

        for box in gt_boxes:
            image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color=(0,255,0), thickness=2)

        proposal_boxes = get_proposals(image.shape[0], image.shape[1], min_h_ratio=1/10, max_h_ratio=1/4, aspect_ratios=[2/3, 1, 3/2], scale_step=1.2,
                              stride_ratio=0.3)

        print("Number of proposals: {}".format(proposal_boxes.shape[0]))

        for gt_box in gt_boxes:
            iou = get_iou(gt_box, proposal_boxes)
            max_idx = np.argmax(iou)
            box = proposal_boxes[max_idx]
            image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color=(255, 0, 0), thickness=2)
            image = cv2.putText(image, "{:.2f}".format(iou[max_idx]), (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

        cv2.imshow("Visualization", image)

        k = cv2.waitKey(0) & 0xFF
        if k == 27:  # Esc key to stop
            break
        elif k == -1:  # normally -1 returned,so don't print it
            continue

    cv2.destroyAllWindows()
    return


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('-va', '--visualize-anno', action='store_true', help="Visualize annotations")
    args_parser.add_argument('-tp', '--test-proposals', action='store_true', help="Visualize annotations")
    args = args_parser.parse_args()
    if args.visualize_anno:
        visualize_annotated_data()

    if args.test_proposals:
        test_get_proposals()
