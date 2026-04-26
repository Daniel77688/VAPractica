#
# Code for the evaluation of the results of the homeworks of Computer Vision (Visión Artificial).
#
# @date 2022/03
# @author josemiguel.buenaposaada@urjc.es
#

import cv2
import csv
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


class BoundingBox:
    """
    Class to be used as object bounding boxes
    """

    def __init__(self, left: object, top: object, right: object, bottom: object, class_id: object = -1,
                 score: object = 1.0, img_idx: object = -1):
        self.left = int(left)
        self.top = int(top)
        self.right = int(right)
        self.bottom = int(bottom)
        self.class_id = int(class_id)
        self.score = score
        self.img_idx = str(img_idx)

    def area(self):
        return (self.right - self.left + 1) * (self.bottom - self.top + 1)

    def pyplot_plot(self, color, lw):
        plt.plot([self.left, self.right, self.right, self.left, self.left],
                 [self.top, self.top, self.bottom, self.bottom, self.top],
                 color, lw=lw)

    def opencv_plot(self, img, color=(0, 0, 255)):
        bb_width = self.right - self.left + 1
        bb_height = self.bottom - self.top + 1
        cv2.rectangle(img, (self.left, self.top), (self.right, self.bottom), color, thickness=2)
        cv2.putText(img, str(self.class_id), (self.left, int(self.top - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 2)
        cv2.putText(img, str(self.score), (self.right, int(self.bottom + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 2)

    def __repr__(self):
        return str((self.img_idx, self.left, self.top, self.right, self.bottom, self.class_id, self.score))


def bboxes_overlap(gt_bbox, dt_bbox, ig):
    """
        Code adapted from Piotr Dollar Matlab toolbox.

        Uses modified Pascal criteria with "ignore" regions. The overlap area
        (oa) of a ground truth (gt) and detected (dt) bb is defined as:

            oa(gt,dt) = area(intersect(dt,dt)) / area(union(gt,dt))

        In the modified criteria, a gt bb may be marked as "ignore", in which
        case the dt bb can match any subregion of the gt bb. Choosing gt' in
        gt that most closely matches dt can be done using gt'=intersect(dt,gt).

        Computing oa(gt',dt) is equivalent to:

            oa'(gt,dt) = area(intersect(gt,dt)) / area(dt)

    :param self:
    :param gt_bbox: Ground truth BoundingBox
    :param dt_bbox: Detected BoundingBox
    :param ig: False/True ignore flag
    :return: overlap area
    """
    w = min(dt_bbox.right, gt_bbox.right) - max(dt_bbox.left, gt_bbox.left)
    if w <= 0:
        return 0.0

    h = min(dt_bbox.bottom, gt_bbox.bottom) - max(dt_bbox.top, gt_bbox.top)
    if h <= 0:
        return 0.0
    i = w * h
    if ig:
        u = dt_bbox.area()
    else:
        u = dt_bbox.area() + gt_bbox.area() - i

    return i / u


# --------------------------------------------------------------------------------------------------------
# Auxiliary functions for drawing detected bounding boxes
# --------------------------------------------------------------------------------------------------------
def show_bboxes_one_image(img, bboxes_list):
    img2 = img.copy()
    for bb in bboxes_list:
        bb_width = bb.right - bb.left + 1
        bb_height = bb.bottom - bb.top + 1
        cv2.rectangle(img2, (bb.left, bb.top), (bb.right, bb.bottom), (0, 0, 255), thickness=2)
        cv2.rectangle(img2, (bb.left, bb.top - 10), (bb.left + 10, bb.top), (0, 0, 0), thickness=-1)
        cv2.putText(img2, str(bb.class_id), (bb.left, int(bb.top - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 2)
        cv2.putText(img2, str(bb.score), (bb.right, int(bb.bottom + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 2)

    cv2.imshow('Image', img2)


def show_bboxes_and_images_dict(images, bboxes):
    for key in images:
        show_bboxes_one_image(images[key], bboxes[key])
        k = cv2.waitKey()
        if k == 27:
            break


def show_images_dict(images):
    for key in images:
        cv2.imshow('Image', images[key])
        k = cv2.waitKey(3000)
        if k == 27:  # ESC
            break


def compute_class_index(number):
    class_index = -1  # Default class index is -1 and means "ignore this signal"
    prohibitory_list = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16]
    mandatory_list = [38]
    danger_list = [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    if number in prohibitory_list:
        class_index = 1  # prohibitory
    elif number in danger_list:
        class_index = 2  # danger
    elif number == 14:
        class_index = 3  # stop
    elif number == 17:
        class_index = 4  # no-entry
    elif number == 13:
        class_index = 5  # Yield (give way)
    elif number in mandatory_list:
        class_index = 6  # mandatory

    return class_index


def load_results_file(file_name, test_path, load_images=False):
    """
    Given a directory where the data from the GTSD benchmarkt is, load the data
    :return (images, bboxes) where images is a dictionary of images and bboxes is a dictionary of lists (of boxes).
              bboxes['img_name'] is a list of BoundingBox objects. The possible classes are:
                   prohibitory:         1
                   danger:              2
                   Stop:                3
                   Forbidden direction: 4
                   Yield:               5
                   Obligation:          6
  """

    results = csv.reader(open(file_name, 'r'), delimiter=';')
    images = dict()
    bboxes = dict()
    with open(file_name, 'r') as gtfile:
        bbreader = csv.reader(gtfile, delimiter=';', quotechar='#')
        for row in bbreader:
            # First read the image.
            if load_images:
                image_path = os.path.join(test_path, row[0])
                if not row[0] in images:
                    I = cv2.imread(image_path)
                    if not type(I) is np.ndarray:
                        print("*** ERROR: Couldn't read image " + image_path)
                    else:
                        images[row[0]] = I

            # Now read the bounding box.
            if len(row) == 7:
                # File with detections as it has a score.
                bb = BoundingBox(left=row[1], top=row[2], right=row[3], bottom=row[4],
                                 class_id=str(row[5]),
                                 score=float(row[6]),
                                 img_idx=str(row[0]))
            else:
                # Ground truth file. Thus, we use the score 1.0.
                bb = BoundingBox(left=row[1], top=row[2], right=row[3], bottom=row[4],
                                 class_id=compute_class_index(int(row[5])),
                                 score=float(1.0),
                                 img_idx=str(row[0]))

            if not row[0] in bboxes:
                bboxes[row[0]] = []

            bboxes[row[0]].append(bb)
            
    return (images, bboxes)

# --------------------------------------------------------------------------------------------------------
# Plot the Precision-Recall curve of the detector
# --------------------------------------------------------------------------------------------------------
def precision_recall_curve(gt_dbboxes, det_dbboxes, show=False, ovr=0.5, images_dict=None):
    """
        Compute the precision recall curve from detections and ground truth bounding boxes.

        NOTE: Implementation adapted from HeadHunter software:
            Mathias, M., Benenson, R., Pedersoli, M., Van Gool, L. (2014).
            Face Detection without Bells and Whistles.
            ECCV 2014.
            https://doi.org/10.1007/978-3-319-10593-2_47
    """
    dimg = {}
    tot = 0
    for idx, bbox in sorted(gt_dbboxes.items(), key=lambda x: x[0]):
        if bbox:
            dimg[idx] = {"bbox": bbox, "det": [False] * len(gt_dbboxes)}
            for i, bbox_i in enumerate(bbox):
                if bbox_i.class_id != -1:  # "ignore region" identificer is -1
                    tot = tot + 1

    det_list = []
    for idx, bbox in sorted(det_dbboxes.items(), key=lambda x: x[0]):
        det_list = det_list + bbox

    det_list = sorted(det_list,  reverse=True, key=lambda x: x.score)

    im_name = []
    cnt = 0
    tp = np.zeros(len(det_list))
    fp = np.zeros(len(det_list))
    thr = np.zeros(len(det_list))
    for idx, det_bb in enumerate(det_list):
        found = False
        maxovr = 0
        gt = 0
        if det_bb.img_idx in dimg:
            bboxes = dimg[det_bb.img_idx]["bbox"]
            for ir, gt_bb in enumerate(bboxes):
                # print("====================")
                covr = bboxes_overlap(gt_bb, det_bb, ig=(gt_bb.class_id == -1))
                # print("gt_bb=", gt_bb)
                # print("det_bb=", det_bb)
                # print("covr=", covr)
                if covr >= maxovr:
                    maxovr = covr
                    gt = ir

        if maxovr > ovr:
            if dimg[det_bb.img_idx]["bbox"][gt].class_id != -1:  # "ignore region" identificer i)s -1
                if not (dimg[det_bb.img_idx]["det"][gt]):
                    tp[idx] = 1
                    dimg[det_bb.img_idx]["det"][gt] = True
                    found = True
                else:
                    fp[idx] = 1
        else:
            fp[idx] = 1

        thr[idx] = det_bb.score
        if show and det_bb.img_idx in dimg:
            prec = np.sum(tp) / float(np.sum(tp) + np.sum(fp))
            rec = np.sum(tp) / tot
            print("Scr:", det_bb.score, "Prec:%.3f" % prec, "Rec:%.3f" % rec)
            if images_dict is None:
                continue

            img = images_dict[det_bb.img_idx]
            img2 = img.copy()
            for gt_bb in bboxes:
                gt_bb.cv_plot(img2, color=(255, 0, 0)) # Blue
            if found:
                det_bb.opencv_plot(img2, color=(0, 255, 0)) # Green is found
            else:
                det_bb.opencv_plot(img2, color=(0, 0, 255)) # Red is not found
            cv2.imshow("GT vs Det", img2)
            cv2.waitKey()
            bboxes = []

    return tp, fp, thr, tot


def VOCap(rec, prec):
    mrec = np.concatenate(([0], rec, [1]))
    mpre = np.concatenate(([0], prec, [0]))
    for i in range(len(mpre) - 2, 0, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    i = np.where(mrec[1:] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap


def VOColdap(rec, prec):
    rec = np.array(rec)
    prec = np.array(prec)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        pr = prec[rec >= t]
        if pr.size == 0:
            pr = 0
        p = np.max(pr)
        ap = ap + p / 11.0
    return ap


def draw_PR_fast(tp, fp, tot, show=True, col="g"):
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    rec = tp / tot
    prec = tp / (fp + tp)
    ap = VOColdap(rec, prec)
    ap1 = VOCap(rec, prec)
    if show:
        plt.plot(rec, prec, '-%s' % col)
        plt.title("AP=%.1f 11pt(%.1f)" % (ap1 * 100, ap * 100))
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.grid()
        plt.gca().set_xlim((0, 1))
        plt.gca().set_ylim((0, 1))
        plt.show()
        plt.draw()

    return rec, prec, ap1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Plots the results of a homework')
    parser.add_argument(
        '--test_path', default="", help='Select the testing data dir')
    parser.add_argument(
        '--detections_file', default="resultado.txt", help='Select the detections results file')
    parser.add_argument(
        '--instructor_detections_file', default="resultado_jmbuena_road_panels.txt", help='File from homework 1')
    parser.add_argument(
        '--show_detections', default=False, help='Show de detections from detection files on the test images')
    args = parser.parse_args()

    print("Load detection results ...")
    dets_file_path = os.path.join(".", args.detections_file)
    print(" Loading file: ", dets_file_path)
    (det_dimages, det_dbboxes) = load_results_file(dets_file_path, args.test_path, load_images=args.show_detections)

    print("Load detection results (instructor's homework 1 results)...")
    dets_file_path = os.path.join(".", args.instructor_detections_file)
    print(" Loading file: ", dets_file_path)
    (det_dimages_instructor, det_dbboxes_instructor) = load_results_file(dets_file_path, args.test_path)

    print("Load ground truth ...")
    gt_file_path = os.path.join(args.test_path, "gt.txt")
    print(" Loading file: ", gt_file_path)
    (gt_dimages, gt_dbboxes) = load_results_file(gt_file_path, args.test_path, load_images=False)

    # Show detections
    if args.show_detections:
        show_bboxes_and_images_dict(det_dimages, det_dbboxes)

    # --------------------------------------------------------------------------------------------
    #  Figure with overlap (IoU) theshold of 0.5 (>=50% overlap is considered a correct detection)
    # --------------------------------------------------------------------------------------------
    tp, fp, thr, tot = precision_recall_curve(gt_dbboxes, det_dbboxes, show=False, ovr=0.5, images_dict=det_dimages)
    rec_det, prec_det, ap1_det = draw_PR_fast(tp, fp, tot, show=False)

    tp, fp, thr, tot = precision_recall_curve(gt_dbboxes, det_dbboxes_instructor, show=False, ovr=0.5, images_dict=det_dimages)
    rec_1, prec_1, ap1_1 = draw_PR_fast(tp, fp, tot, show=False)


    # Actually plot the precision-recall curves.
    plt.figure()
    plt.plot(rec_det, prec_det, '-r')
    plt.plot(rec_1, prec_1, '-g')
    plt.legend(["Prác-1 AP=%.1f" % (ap1_det * 100),
                "Prác-jmbuena - AP=%.1f" % (ap1_1 * 100)])
    plt.grid()
    plt.gca().set_xlim((0, 1))
    plt.gca().set_ylim((0, 1.1))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Comparación con práctica profesores (IoU>0.5)")
    plt.legend
    
    # --------------------------------------------------------------------------------------------
    #  Figure with overlap (IoU) theshold of 0.7 (>=70% overlap is considered a correct detection)
    # --------------------------------------------------------------------------------------------
    tp, fp, thr, tot = precision_recall_curve(gt_dbboxes, det_dbboxes, show=False, ovr=0.7, images_dict=det_dimages)
    rec_det, prec_det, ap1_det = draw_PR_fast(tp, fp, tot, show=False)

    tp, fp, thr, tot = precision_recall_curve(gt_dbboxes, det_dbboxes_instructor, show=False, ovr=0.7, images_dict=det_dimages)
    rec_1, prec_1, ap1_1 = draw_PR_fast(tp, fp, tot, show=False)


    # Actually plot the precision-recall curves.
    plt.figure()
    plt.plot(rec_det, prec_det, '-r')
    plt.plot(rec_1, prec_1, '-g')
    plt.legend(["Prác-1 AP=%.1f" % (ap1_det * 100),
                "Prác-jmbuena - AP=%.1f" % (ap1_1 * 100)])
    plt.grid()
    plt.gca().set_xlim((0, 1))
    plt.gca().set_ylim((0, 1.1))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Comparación con práctica profesores  (IoU>0.7)")
    plt.legend
    plt.show()
    plt.draw()
