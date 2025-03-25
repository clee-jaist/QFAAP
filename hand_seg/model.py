import os
import time

import numpy as np
import cv2
import tensorflow as tf

class DeepLabModel(object):
    """Class to load DeepLab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, graph_dir, input_width, input_height):
        """Creates and loads pretrained DeepLab model."""
        self.graph = tf.Graph()
        self.input_width = input_width
        self.input_height = input_height

        graph_def = None
        if self.FROZEN_GRAPH_NAME + '.pb' in os.listdir(graph_dir):
            filename = graph_dir + '/' + self.FROZEN_GRAPH_NAME + '.pb'
            graph_def = tf.compat.v1.GraphDef.FromString(
                open(filename, 'rb').read())

        if graph_def is None:
            raise RuntimeError(
                'Cannot find inference graph in folder ' + graph_dir)

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.compat.v1.Session(graph=self.graph)
    
    def __input_preprocess(self, image):
        """Image input preprocessing (BGR2RGB or BGR2LAB conversion, then resize if necessary).
        Args:
          image: A Image object read using cv2 (BGR order).

        Returns:
          image: Input preprocessed image.
        """
        target_size = (self.input_width, self.input_height)
        # print("img shape w x h: ", image.shape[0], " x ", image.shape[1])
        if not(image.shape[1] == self.input_width and image.shape[0] == self.input_height):
            # resize the input image
            image = cv2.resize(image, target_size,
                               interpolation=cv2.INTER_LANCZOS4)
            #print("image shape: ", image.shape)
        return image

    def run(self, image):
        """Runs inference on a single image.

        Args:
          image: A Image object read using cv2 (BGR order).

        Returns:
          seg_map: Segmentation map of input processed image.
        """
        image = self.__input_preprocess(image)   
        start_time = time.time()
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(image)]})
        seg_map = batch_seg_map[0]
        time_elapsed = time.time() - start_time
        print('Inference time in seconds: {}'.format(time_elapsed))
        return seg_map