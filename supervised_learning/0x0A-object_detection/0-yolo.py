#!/usr/bin/env python3
"""YOLO module"""
import tensorflow.keras as A

    class Yolo:
        """preform object detection"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):

        """
        Init function

        """
        self.model = A.models.load_model(model_path)
        with open(classes_path, 'r') as d:
            self.class_names = d.read().rstrip('\n').split('\n')
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
