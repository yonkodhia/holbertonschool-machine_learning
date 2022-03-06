#!/usr/bin/env python3
"""Yolo class"""
import tensorflow.keras as K
import numpy as np
import cv2
import glob


class Yolo:
    """Yolo class"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        constructor
        """
        self.model = K.models.load_model(model_path)

        with open(classes_path, "r") as fd:
            classes = fd.read()
            classes = classes.split('\n')
            if len(classes[-1]) == 0:
                classes = classes[:-1]

        self.class_names = classes
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        """calculates a sigmoid function"""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        process outputs
        """
        boxes = []
        image_h, image_w = image_size
        confidence_boxes = []
        prop_boxes = []

        for i, output in enumerate(outputs):
            # there is three outputs one for each grids 13x13, 26x26, 52x52
            grid_h, grid_w, n_anchor, _ = outputs[i].shape
            box = np.zeros((grid_h, grid_w, n_anchor, 4))
            # get coordinates, width height of the outputs
            tx = (output[:, :, :, 0])
            ty = (output[:, :, :, 1])
            tw = (output[:, :, :, 2])
            th = (output[:, :, :, 3])

            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]

            # normalize
            tx_n = self.sigmoid(tx)
            ty_n = self.sigmoid(ty)

            # get corners of grid
            cx = np.tile(np.arange(0, grid_w), grid_h)
            cx = cx.reshape(grid_w, grid_w, 1)

            cy = np.tile(np.arange(0, grid_w), grid_h)
            # y doesn't change until x finish so .T
            cy = cy.reshape(grid_h, grid_h).T
            cy = cy.reshape(grid_h, grid_h, 1)

            # boxes prediction
            bx = tx_n + cx
            by = ty_n + cy
            bw = np.exp(tw) * pw
            bh = np.exp(th) * ph

            # normalize
            bx /= grid_w
            by /= grid_h
            bw /= self.model.input.shape[1].value
            bh /= self.model.input.shape[2].value

            # bounding box coordinates respect image. top left corner (x1, y1)
            # and bottom right corner (x2, y2)
            x1 = (bx - (bw / 2)) * image_w
            y1 = (by - (bh / 2)) * image_h
            x2 = (bx + (bw / 2)) * image_w
            y2 = (by + (bh / 2)) * image_h
            box[:, :, :, 0] = x1
            box[:, :, :, 1] = y1
            box[:, :, :, 2] = x2
            box[:, :, :, 3] = y2
            boxes.append(box)

            aux = output[:, :, :, 4]
            confidence = self.sigmoid(aux)
            confidence = confidence.reshape(grid_h, grid_w, n_anchor, 1)
            confidence_boxes.append(confidence)

            aux = output[:, :, :, 5:]
            class_props = self.sigmoid(aux)
            prop_boxes.append(class_props)
        return boxes, confidence_boxes, prop_boxes

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Select higher probability
        """

        multiply = []
        # take one of each and multiply
        for confidence, classes in zip(box_confidences, box_class_probs):
            multiply.extend(confidence * classes)

        # from list of numpy arrays to one numpy array
        multiply = np.concatenate(multiply)

        # argmax and max to take the highest probability
        index = np.argmax(multiply, -1)
        score_class = np.max(multiply, -1)

        # reshape(-1) turn them into a vector
        index = index.reshape(-1)
        score_class = score_class.reshape(-1)

        # mask is going to return a list with the positions
        # that fulfill the condition
        mask = np.where(score_class >= self.class_t)

        box_class = index[mask]
        box_score = score_class[mask]

        # processing the boxes, turn them into matrix of (?, 4)
        filter_box = [elem.reshape(-1, 4) for elem in boxes]
        filter_box = np.concatenate(filter_box)
        filter_box = filter_box[mask]

        return filter_box, box_class, box_score

    def iou(self, box1, box2):
        """
        Calculates intersection over union
        Args:
            box1: first box, list object with coordinates
            (box1_x1, box1_y1, box1_x2, box_1_y2)
            box2: second box, list object with coordinates
            (box2_x1, box2_y1, box2_x2, box2_y2)
        Returns: intersection over union
        """

        xi1 = max(box1[0], box2[0])
        yi1 = max(box1[1], box2[1])
        xi2 = min(box1[2], box2[2])
        yi2 = min(box1[3], box2[3])
        inter_width = max(xi2 - xi1, 0)
        inter_height = max(yi2 - yi1, 0)
        inter_area = inter_width * inter_height

        box1_area = (box1[3] - box1[1]) * (box1[2] - box1[0])
        box2_area = (box2[3] - box2[1]) * (box2[2] - box2[0])
        union_area = box1_area + box2_area - inter_area

        iou = inter_area / union_area

        return iou

    def nms(self, filter, thresh, scores):
        """
        Calculate remain boxes
        """
        x1 = filter[:, 0]
        y1 = filter[:, 1]
        x2 = filter[:, 2]
        y2 = filter[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            indx = np.where(ovr <= thresh)[0]
            order = order[indx + 1]

        return keep

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Non max suppression
        """
        all_classes = []
        all_scores = []
        all_filters = []

        for i in(np.unique(box_classes)):
            idx = np.where(box_classes == i)
            filter = filtered_boxes[idx]
            scores = box_scores[idx]
            classes = box_classes[idx]
            keep = self.nms(filter, self.nms_t, scores)

            filter = filter[keep]
            classes = classes[keep]
            scores = scores[keep]
            all_scores.append(scores)
            all_classes.append(classes)
            all_filters.append(filter)

        all_filters = np.concatenate(all_filters, axis=0)
        all_classes = np.concatenate(all_classes, axis=0)
        all_scores = np.concatenate(all_scores, axis=0)
        return all_filters, all_classes, all_scores

    @staticmethod
    def load_images(folder_path):
        """
        Loads an image
        """
        image_paths = glob.glob(folder_path + "/*", recursive=False)
        images = [cv2.imread(img) for img in image_paths]
        return images, image_paths