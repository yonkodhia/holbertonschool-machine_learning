#!/usr/bin/env python3
"""Yolo class"""
import tensorflow.keras as K
import numpy as np


class Yolo:
    """Yolo class"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Csonstructor
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
