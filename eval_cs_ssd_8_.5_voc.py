from __future__ import division

# import the necessary packages

from eval_utils.average_precision_evaluator_cs import Evaluator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
import tensorflow as tf

import tensorflow as tf
import math
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers  import Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Lambda
import time
import csv
import shutil
import os
import numpy as np
import matplotlib.pyplot as plt
from math import floor
from statistics import mean
import pandas as pd

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
#from imutils import paths
import random
import pickle
import cv2
import os

# Commented out IPython magic to ensure Python compatibility.
# import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, CSVLogger, LearningRateScheduler
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt

from models.depthwise_conv2d import DepthwiseConvolution2D
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
from data_generator.data_augmentation_chain_variable_input_size import DataAugmentationVariableInputSize
from data_generator.data_augmentation_chain_constant_input_size import DataAugmentationConstantInputSize
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels

# %matplotlib inline


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, BatchNormalization, ELU, Reshape, Concatenate, Activation
from tensorflow.keras.layers import Convolution2D, ZeroPadding2D, Add, Conv2DTranspose
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K


import statistics

import skimage
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr
from statistics import mean
os.environ["CUDA_VISIBLE_DEVICES"]="0"

M_over_N = 0.5  #compresion rate
w = 304
h = 304
l = 3
B = 8  #filter size

n_B = math.ceil(M_over_N * l * B**2) #number of filters
n_blocks = int(w / B)

print ("n_B = ", n_B)
print ("n_blocks = ", n_blocks)


img_height = 304  # Height of the input images
img_width = 304  # Width of the input images
img_channels = 3  # Number of color channels of the input images
sstdev = 127.5
mean_color = 127.5  # The per-channel mean of the images in the dataset
swap_channels = False  # The color channel order in the original SSD is BGR
n_classes = 20  # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
scales_voc = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88,
              1.05]  # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
scales_coco = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87,
               1.05]  # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets
scales = scales_voc

aspect_ratios = [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]]  # The anchor box aspect ratios used in the original SSD300; the order matters
two_boxes_for_ar1 = True
steps = [8, 16, 32, 64, 100, 300]  # The space between two adjacent anchor box center points for each predictor layer.
offsets = [0.5, 0.5, 0.5, 0.5, 0.5,
           0.5]  # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
clip_boxes = False  # Whether or not you want to limit the anchor boxes to lie entirely within the image boundaries
variances = [0.1, 0.1, 0.2,
             0.2]  # The variances by which the encoded target coordinates are scaled as in the original implementation
coords = 'centroids'  # Whether the box coordinates to be used as targets for the model should be in the 'centroids', 'corners', or 'minmax' format, see documentation
normalize_coords = True

#mean_color=127.5
#sstdev=127.5
#swap_channels=False

class CS_SSD:
  @staticmethod
  def build_ssd_branch(code,
          n_classes,
          mode='training',
          l2_regularization=0.0005,
          min_scale=None,
          max_scale=None,
          scales=scales,
          aspect_ratios_global=None,
          aspect_ratios_per_layer=aspect_ratios,
          two_boxes_for_ar1=True,
          steps=[8, 16, 32, 64, 100, 300],
          offsets=offsets,
          clip_boxes=False,
          variances=[0.1, 0.1, 0.2, 0.2],
          coords= coords,
          normalize_coords=True,
          # subtract_mean=None,
          # divide_by_stddev=None,
          # swap_channels=False,
          confidence_thresh=0.01,
          iou_threshold=0.45,
          top_k=200,
          nms_max_output_size=400,
          return_predictor_sizes=False):


    n_predictor_layers = 6 # The number of predictor conv layers in the network is 6 for the original SSD300.
    
    n_classes += 1 # Account for the background class.
    l2_reg = l2_regularization # Make the internal name shorter.
    

    # img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]
    img_height, img_width, img_channels = code.shape[1],code.shape[2],code.shape[3]
    print(code.shape[1])
    height = width = 304
    print(img_height, img_width, img_channels)

    ############################################################################
    # Get a few exceptions out of the way.
    ############################################################################

    if aspect_ratios_global is None and aspect_ratios_per_layer is None:
        raise ValueError("`aspect_ratios_global` and `aspect_ratios_per_layer` cannot both be None. At least one needs to be specified.")
    if aspect_ratios_per_layer:
        if len(aspect_ratios_per_layer) != n_predictor_layers:
            raise ValueError("It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == {}, but len(aspect_ratios_per_layer) == {}.".format(n_predictor_layers, len(aspect_ratios_per_layer)))

    if (min_scale is None or max_scale is None) and scales is None:
        raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")
    if scales:
        if len(scales) != n_predictor_layers+1:
            raise ValueError("It must be either scales is None or len(scales) == {}, but len(scales) == {}.".format(n_predictor_layers+1, len(scales)))
    else: # If no explicit list of scaling factors was passed, compute the list of scaling factors from `min_scale` and `max_scale`
        scales = np.linspace(min_scale, max_scale, n_predictor_layers+1)

    if len(variances) != 4:
        raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
    variances = np.array(variances)
    if np.any(variances <= 0):
        raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

    if (not (steps is None)) and (len(steps) != n_predictor_layers):
        raise ValueError("You must provide at least one step value per predictor layer.")

    if (not (offsets is None)) and (len(offsets) != n_predictor_layers):
        raise ValueError("You must provide at least one offset value per predictor layer.")

    ############################################################################
    # Compute the anchor box parameters.
    ############################################################################

    # Set the aspect ratios for each predictor layer. These are only needed for the anchor box layers.
    if aspect_ratios_per_layer:
        aspect_ratios = aspect_ratios_per_layer
    else:
        aspect_ratios = [aspect_ratios_global] * n_predictor_layers

    # Compute the number of boxes to be predicted per cell for each predictor layer.
    # We need this so that we know how many channels the predictor layers need to have.
    if aspect_ratios_per_layer:
        n_boxes = []
        for ar in aspect_ratios_per_layer:
            if (1 in ar) & two_boxes_for_ar1:
                n_boxes.append(len(ar) + 1) # +1 for the second box for aspect ratio 1
            else:
                n_boxes.append(len(ar))
    else: # If only a global aspect ratio list was passed, then the number of boxes is the same for each predictor layer
        if (1 in aspect_ratios_global) & two_boxes_for_ar1:
            n_boxes = len(aspect_ratios_global) + 1
        else:
            n_boxes = len(aspect_ratios_global)
        n_boxes = [n_boxes] * n_predictor_layers
    

    if steps is None:
        steps = [None] * n_predictor_layers
    if offsets is None:
        offsets = [None] * n_predictor_layers

    ############################################################################
    # Build the network.
    ############################################################################

    ## try fine tuning using rgb images ## 
    # x1 = Lambda(lambda c: tf.image.rgb_to_grayscale(c))(x1)
    # x1 = Lambda(grayscale_to_rgb, output_shape=(img_height, img_width, img_channels), name='grayscale_to_rgb_cheated')(x1)
    
    # conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv1_1')(x1)
    # batch1_1 = BatchNormalization(name='batch1_1')(conv1_1)
    # conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv1_2')(batch1_1)
    # batch1_2 = BatchNormalization(name='batch1_2')(conv1_2)
    # pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1')(batch1_2)

    # conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv2_1')(pool1)
    # batch2_1 = BatchNormalization(name='batch2_1')(conv2_1)
    # conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv2_2')(batch2_1)
    # batch2_2 = BatchNormalization(name='batch2_2')(conv2_2)
    # pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool2')(batch2_2)

    # conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv3_1')(code)
    # batch3_1 = BatchNormalization(name='batch3_1')(conv3_1)
    # conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv3_2')(batch3_1)
    # batch3_2 = BatchNormalization(name='batch3_2')(conv3_2)
    # conv3_3 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv3_3')(batch3_2)
    # batch3_3 = BatchNormalization(name='batch3_3')(conv3_3)
    # pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool3')(batch3_3)

    conv4_1 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4_1')(code)
    batch4_1 = BatchNormalization(name='batch4_1')(conv4_1)
    conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4_2')(batch4_1)
    batch4_2 = BatchNormalization(name='batch4_2')(conv4_2)
    conv4_3 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4_3')(batch4_2)
    batch4_3 = BatchNormalization(name='batch4_3')(conv4_3)
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool4')(batch4_3)

    conv5_1 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv5_1')(pool4)
    batch5_1 = BatchNormalization(name='batch5_1')(conv5_1)
    conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv5_2')(batch5_1)
    batch5_2 = BatchNormalization(name='batch5_2')(conv5_2)
    conv5_3 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv5_3')(batch5_2)
    batch5_3 = BatchNormalization(name='batch5_3')(conv5_3)
    pool5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='pool5')(batch5_3)

    fc6 = Conv2D(1024, (3, 3), dilation_rate=(6, 6), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='fc6')(pool5)
    batch6 = BatchNormalization(name='batch6')(fc6)

    fc7 = Conv2D(1024, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='fc7')(batch6)
    batch7 = BatchNormalization(name='batch7')(fc7)

    conv6_1 = Conv2D(256, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv6_1')(fc7)
    batch6_1 = BatchNormalization(name='batch6_1')(conv6_1)
    conv6_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv6_padding')(batch6_1)
    conv6_2 = Conv2D(512, (3, 3), strides=(2, 2), activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv6_2')(conv6_1)
    batch6_2 = BatchNormalization(name='batch6_2')(conv6_2)

    conv7_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv7_1')(batch6_2)
    batch7_1 = BatchNormalization(name='batch7_1')(conv7_1)
    conv7_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv7_padding')(batch7_1)
    conv7_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv7_2')(conv7_1)
    batch7_2 = BatchNormalization(name='batch7_2')(conv7_2)

    conv8_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv8_1')(batch7_2)
    batch8_1 = BatchNormalization(name='batch8_1')(conv8_1)
    conv8_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv8_2')(batch8_1)
    batch8_2 = BatchNormalization(name='batch8_2')(conv8_2)

    conv9_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv9_1')(batch8_2)
    batch9_1 = BatchNormalization(name='batch9_1')(conv9_1)
    conv9_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv9_2')(batch9_1)
    batch9_2 = BatchNormalization(name='batch9_2')(conv9_2)
    # print('cc5')
    # Feed conv4_3 into the L2 normalization layer
    # conv4_3_norm = L2Normalization(gamma_init=20, name='conv4_3_norm')(conv4_3)
    
    

    ### Build the convolutional predictor layers on top of the base network

    # We precidt `n_classes` confidence values for each box, hence the confidence predictors have depth `n_boxes * n_classes`
    # Output shape of the confidence layers: `(batch, height, width, n_boxes * n_classes)`
    conv4_3_norm_mbox_conf = Conv2D(n_boxes[0] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4_3_norm_mbox_conf')(conv4_3)
    batch4_3_norm_mbox_conf = BatchNormalization(name='batch4_3_norm_mbox_conf')(conv4_3_norm_mbox_conf)
    fc7_mbox_conf = Conv2D(n_boxes[1] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='fc7_mbox_conf')(fc7)
    batch7_mbox_conf = BatchNormalization(name='batch7_mbox_conf')(fc7_mbox_conf)
    conv6_2_mbox_conf = Conv2D(n_boxes[2] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv6_2_mbox_conf')(conv6_2)
    batch6_2_mbox_conf = BatchNormalization(name='batch6_2_mbox_conf')(conv6_2_mbox_conf)
    conv7_2_mbox_conf = Conv2D(n_boxes[3] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv7_2_mbox_conf')(conv7_2)
    batch7_2_mbox_conf = BatchNormalization(name='batch7_2_mbox_conf')(conv7_2_mbox_conf)
    conv8_2_mbox_conf = Conv2D(n_boxes[4] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv8_2_mbox_conf')(conv8_2)
    batch8_2_mbox_conf = BatchNormalization(name='batch8_2_mbox_conf')(conv8_2_mbox_conf)
    conv9_2_mbox_conf = Conv2D(n_boxes[5] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv9_2_mbox_conf')(batch9_2)
    batch9_2_mbox_conf = BatchNormalization(name='batch9_2_mbox_conf')(conv9_2_mbox_conf)
    # We predict 4 box coordinates for each box, hence the localization predictors have depth `n_boxes * 4`
    # Output shape of the localization layers: `(batch, height, width, n_boxes * 4)`
    conv4_3_norm_mbox_loc = Conv2D(n_boxes[0] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4_3_norm_mbox_loc')(conv4_3)
    batch4_3_norm_mbox_loc = BatchNormalization(name='batch4_3_norm_mbox_loc')(conv4_3_norm_mbox_loc)
    fc7_mbox_loc = Conv2D(n_boxes[1] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='fc7_mbox_loc')(fc7)
    batch7_mbox_loc = BatchNormalization(name='batch7_mbox_loc')(fc7_mbox_loc)
    conv6_2_mbox_loc = Conv2D(n_boxes[2] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv6_2_mbox_loc')(conv6_2)
    batch6_2_mbox_loc = BatchNormalization(name='batch6_2_mbox_loc')(conv6_2_mbox_loc)
    conv7_2_mbox_loc = Conv2D(n_boxes[3] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv7_2_mbox_loc')(conv7_2)
    batch7_2_mbox_loc = BatchNormalization(name='batch7_2_mbox_loc')(conv7_2_mbox_loc)
    conv8_2_mbox_loc = Conv2D(n_boxes[4] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv8_2_mbox_loc')(conv8_2)
    batch8_2_mbox_loc = BatchNormalization(name='batch8_2_mbox_loc')(conv8_2_mbox_loc)
    conv9_2_mbox_loc = Conv2D(n_boxes[5] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv9_2_mbox_loc')(batch9_2)
    batch9_2_mbox_loc = BatchNormalization(name='batch9_2_mbox_loc')(conv9_2_mbox_loc)

    ### Generate the anchor boxes (called "priors" in the original Caffe/C++ implementation, so I'll keep their layer names)

    # Output shape of anchors: `(batch, height, width, n_boxes, 8)`
    conv4_3_norm_mbox_priorbox = AnchorBoxes(height , width, this_scale=scales[0], next_scale=scales[1], aspect_ratios=aspect_ratios[0],
                                             two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0], this_offsets=offsets[0], clip_boxes=clip_boxes,
                                             variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv4_3_norm_mbox_priorbox')(batch4_3_norm_mbox_loc)
    fc7_mbox_priorbox = AnchorBoxes(height , width, this_scale=scales[1], next_scale=scales[2], aspect_ratios=aspect_ratios[1],
                                    two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[1], this_offsets=offsets[1], clip_boxes=clip_boxes,
                                    variances=variances, coords=coords, normalize_coords=normalize_coords, name='fc7_mbox_priorbox')(batch7_mbox_loc)
    conv6_2_mbox_priorbox = AnchorBoxes(height , width, this_scale=scales[2], next_scale=scales[3], aspect_ratios=aspect_ratios[2],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[2], this_offsets=offsets[2], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv6_2_mbox_priorbox')(batch6_2_mbox_loc)
    conv7_2_mbox_priorbox = AnchorBoxes(height , width, this_scale=scales[3], next_scale=scales[4], aspect_ratios=aspect_ratios[3],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[3], this_offsets=offsets[3], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv7_2_mbox_priorbox')(batch7_2_mbox_loc)
    conv8_2_mbox_priorbox = AnchorBoxes(height , width, this_scale=scales[4], next_scale=scales[5], aspect_ratios=aspect_ratios[4],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[4], this_offsets=offsets[4], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv8_2_mbox_priorbox')(batch8_2_mbox_loc)
    conv9_2_mbox_priorbox = AnchorBoxes(height , width, this_scale=scales[5], next_scale=scales[6], aspect_ratios=aspect_ratios[5],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[5], this_offsets=offsets[5], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv9_2_mbox_priorbox')(batch9_2_mbox_loc)

    ### Reshape

    # Reshape the class predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, n_classes)`
    # We want the classes isolated in the last axis to perform softmax on them
    conv4_3_norm_mbox_conf_reshape = Reshape((-1, n_classes), name='conv4_3_norm_mbox_conf_reshape')(batch4_3_norm_mbox_conf)
    fc7_mbox_conf_reshape = Reshape((-1, n_classes), name='fc7_mbox_conf_reshape')(batch7_mbox_conf)
    conv6_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv6_2_mbox_conf_reshape')(batch6_2_mbox_conf)
    conv7_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv7_2_mbox_conf_reshape')(batch7_2_mbox_conf)
    conv8_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv8_2_mbox_conf_reshape')(batch8_2_mbox_conf)
    conv9_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv9_2_mbox_conf_reshape')(batch9_2_mbox_conf)
    # Reshape the box predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, 4)`
    # We want the four box coordinates isolated in the last axis to compute the smooth L1 loss
    conv4_3_norm_mbox_loc_reshape = Reshape((-1, 4), name='conv4_3_norm_mbox_loc_reshape')(batch4_3_norm_mbox_loc)
    fc7_mbox_loc_reshape = Reshape((-1, 4), name='fc7_mbox_loc_reshape')(batch7_mbox_loc)
    conv6_2_mbox_loc_reshape = Reshape((-1, 4), name='conv6_2_mbox_loc_reshape')(batch6_2_mbox_loc)
    conv7_2_mbox_loc_reshape = Reshape((-1, 4), name='conv7_2_mbox_loc_reshape')(batch7_2_mbox_loc)
    conv8_2_mbox_loc_reshape = Reshape((-1, 4), name='conv8_2_mbox_loc_reshape')(batch8_2_mbox_loc)
    conv9_2_mbox_loc_reshape = Reshape((-1, 4), name='conv9_2_mbox_loc_reshape')(batch9_2_mbox_loc)
    # Reshape the anchor box tensors, yielding 3D tensors of shape `(batch, height * width * n_boxes, 8)`
    conv4_3_norm_mbox_priorbox_reshape = Reshape((-1, 8), name='conv4_3_norm_mbox_priorbox_reshape')(conv4_3_norm_mbox_priorbox)
    fc7_mbox_priorbox_reshape = Reshape((-1, 8), name='fc7_mbox_priorbox_reshape')(fc7_mbox_priorbox)
    conv6_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv6_2_mbox_priorbox_reshape')(conv6_2_mbox_priorbox)
    conv7_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv7_2_mbox_priorbox_reshape')(conv7_2_mbox_priorbox)
    conv8_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv8_2_mbox_priorbox_reshape')(conv8_2_mbox_priorbox)
    conv9_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv9_2_mbox_priorbox_reshape')(conv9_2_mbox_priorbox)

    ### Concatenate the predictions from the different layers


    # Axis 0 (batch) and axis 2 (n_classes or 4, respectively) are identical for all layer predictions,
    # so we want to concatenate along axis 1, the number of boxes per layer
    # Output shape of `mbox_conf`: (batch, n_boxes_total, n_classes)
    mbox_conf = Concatenate(axis=1, name='mbox_conf')([conv4_3_norm_mbox_conf_reshape,
                                                       fc7_mbox_conf_reshape,
                                                       conv6_2_mbox_conf_reshape,
                                                       conv7_2_mbox_conf_reshape,
                                                       conv8_2_mbox_conf_reshape,
                                                       conv9_2_mbox_conf_reshape])

    # Output shape of `mbox_loc`: (batch, n_boxes_total, 4)
    mbox_loc = Concatenate(axis=1, name='mbox_loc')([conv4_3_norm_mbox_loc_reshape,
                                                     fc7_mbox_loc_reshape,
                                                     conv6_2_mbox_loc_reshape,
                                                     conv7_2_mbox_loc_reshape,
                                                     conv8_2_mbox_loc_reshape,
                                                     conv9_2_mbox_loc_reshape])

    # Output shape of `mbox_priorbox`: (batch, n_boxes_total, 8)
    mbox_priorbox = Concatenate(axis=1, name='mbox_priorbox')([conv4_3_norm_mbox_priorbox_reshape,
                                                               fc7_mbox_priorbox_reshape,
                                                               conv6_2_mbox_priorbox_reshape,
                                                               conv7_2_mbox_priorbox_reshape,
                                                               conv8_2_mbox_priorbox_reshape,
                                                               conv9_2_mbox_priorbox_reshape])

    # The box coordinate predictions will go into the loss function just the way they are,
    # but for the class predictions, we'll apply a softmax activation layer first
    mbox_conf_softmax = Activation('softmax', name='mbox_conf_softmax')(mbox_conf)

    # Concatenate the class and box predictions and the anchors to one large predictions vector
    # Output shape of `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)
    predictions = Concatenate(axis=2, name='predictions')([mbox_conf_softmax, mbox_loc, mbox_priorbox])

    if mode == 'training':
        # model = Model(inputs=x, outputs=predictions)
        predictions = Concatenate(axis=2, name='predictions')([mbox_conf_softmax, mbox_loc, mbox_priorbox])
    elif mode == 'inference':
        predictions = DecodeDetections(confidence_thresh=confidence_thresh,
                                               iou_threshold=iou_threshold,
                                               top_k=top_k,
                                               nms_max_output_size=nms_max_output_size,
                                               coords=coords,
                                               normalize_coords=normalize_coords,
                                               img_height=height,
                                               img_width=width,
                                               name='decoded_predictions')(predictions)
        # model = Model(inputs=x, outputs=decoded_predictions)

    return predictions
  @staticmethod
  def build_decoder_branch(code):
    
    volumeSize = code.shape
    # print(code.shape)
    compressedInputs = Input(shape=volumeSize[1:])
    # print(compressedInputs.shape)
    x = Conv2D(l*B**2, kernel_size = (1,1), strides=1, activation=None, use_bias=False, name = 'D_conv')(code)
    # print(x.shape)

    #reshape - concate
    x = tf.compat.v2.keras.layers.Reshape((n_blocks,n_blocks,B,B,l), input_shape=(int(w/B),int(w/B),l*B**2), name  = 'reshape')(x)
    x = tf.compat.v2.keras.layers.Permute(dims = (1,3,2,4,5), input_shape=(n_blocks,n_blocks,B,B,l), name = 'permute')(x)
    decoded = tf.compat.v2.keras.layers.Reshape((w,h,l), input_shape=(n_blocks,B,n_blocks,B,l), name = 'decoded')(x)
    
    # print(decoded.shape)

    # build the decoder model
    # decoder = Model(compressedInputs, x, name="decoder")
    # decoder = Model(compressedInputs, output, name="decoder")
    return decoded

  @staticmethod
  def build(h,
            w,
            l):
            # mean_color=mean_color,
            # sstdev=sstdev,
            # swap_channels = False):
            # subtract_mean=None,
            # divide_by_stddev=None,
            # swap_channels=False,):
    # initialize the input shape to be "channels last" along with
    # the channels dimension itself
    # channels dimension itself
    
    
    ############################################################################
    # Define functions for the Lambda layers below.
    ############################################################################   
    ###########################""
    inputShape = (h, w, 3)
    chanDim = -1

    # define the input to the encoder
    inputs = Input(shape=inputShape)
    x = inputs

    ############################################################################
    # Build the network.
    ############################################################################

    # The following identity layer is only needed so that the subsequent lambda layers can be optional.
    # x1 = Lambda(identity_layer, output_shape=(img_height, img_width, img_channels), name='identity_layer')(x)
    # if not (subtract_mean is None):
    # x1 = Lambda(input_mean_normalization, output_shape=(img_height, img_width, img_channels), name='input_mean_normalization')(x)
    # # if not (divide_by_stddev is None):
    # x1 = Lambda(input_stddev_normalization, output_shape=(img_height, img_width, img_channels), name='input_stddev_normalization')(x1)
    # if swap_channels:
    #   x1 = Lambda(input_channel_swap, output_shape=(img_height, img_width, img_channels), name='input_channel_swap')(x1)


    #x = Lambda(lambda c: tf.image.rgb_to_grayscale(c))(x)
    encoded = Conv2D(n_B, kernel_size = (B,B), strides=(B,B), activation=None, use_bias=False, name = 'E_conv')(x)
    #print(encoded.shape)

    # build the encoder model
    # encoder = Model(inputs, x, name="encoder")

    # construct both the "ssd" and "decoder" sub-networks
    # inputs = Input(shape=inputShape)
    ssdBranch = CS_SSD.build_ssd_branch(encoded,
                    n_classes=n_classes,
                    mode='training',
                    l2_regularization=0.0005,
                    scales=scales,
                    #aspect_ratios_global=aspect_ratios,
                    aspect_ratios_per_layer=aspect_ratios,
                    two_boxes_for_ar1=two_boxes_for_ar1,
                    steps=steps,
                    offsets=offsets,
                    clip_boxes=clip_boxes,
                    variances=variances,
                    normalize_coords=normalize_coords)
                    # subtract_mean=intensity_mean,
                    # divide_by_stddev=intensity_range)
    decodedBranch = CS_SSD.build_decoder_branch(encoded)
    # create the model using our input (the batch of images) and
    # two separate outputs -- one for the clothing category
    # branch and another for the color branch, respectively
    model = Model(
      inputs=inputs,
      # outputs=[decodedBranch ],
      outputs=[ssdBranch, decodedBranch ],
      name="cs_ssd")
    # return the constructed network architecture
    return model

# initialize our FashionNet multi-output network
K.clear_session() # Clear previous models from memory.



# psnr and ssim batch wise function

def psnr_ssim_batch(original,reconstructed):
  if len(original) != len(reconstructed) :
    print("batches must have same leght")
  if original.shape != reconstructed.shape :
    print("batches must have same shapes")
  sh = original.shape
  l = len(original)
  psnr, ssim = [], []
  if sh[:-1] == 1 :
    for i in range(l):
      psnr.append(compare_psnr(original[i],reconstructed[i]))
      #  w = b[i].reshape(256,256)
      ssim.append(compare_ssim(original[i][:,:,0],reconstructed[i][:,:,0]))
  else:
    for i in range(l):
      psnr.append(compare_psnr(original[i],reconstructed[i]))
      #  w = b[i].reshape(256,256)
      ssim.append(compare_ssim(original[i],reconstructed[i],multichannel=True))
  return mean(psnr), mean(ssim)
"""

#loss curves
H = pd.read_csv('cs_ssd_8_.5/training_log.csv')
# plot the total loss, category loss, and color loss
lossNames = ["loss", "predictions_loss", "decoded_loss"]
plt.style.use("ggplot")
(fig, ax) = plt.subplots(3, 1, figsize=(13, 13))

# loop over the loss names
for (i, g) in enumerate(lossNames):
	# plot the loss for both the training and validation data
	title = "Loss for {}".format(g) if g != "loss" else "Total loss"
	ax[i].set_title(title)
	ax[i].set_xlabel("Epoch #")
	ax[i].set_ylabel("Loss")
	ax[i].plot(np.arange(0, 120), H[g].iloc[1:121], label=g)
	ax[i].plot(np.arange(0, 120), H["val_" + g].iloc[1:121],
		label="val_" + g)
	ax[i].legend()
# save the losses figure
plt.tight_layout()
# plt.savefig("{}_losses.png".format(args["plot"]))
plt.savefig("losses.png")
plt.close()
"""

"""***7. Evaluation***"""

# 1: Build the Keras model.
model_mode = 'inference'

K.clear_session() # Clear previous models from memory.

# code = Input(shape=(38,38,64))

code = Input(shape=(n_blocks,n_blocks,n_B))

ssd = CS_SSD.build_ssd_branch(code,
                    n_classes=n_classes,
                    mode=model_mode,
                    l2_regularization=0.0005,
                    scales=scales,
                    #aspect_ratios_global=aspect_ratios,
                    aspect_ratios_per_layer=aspect_ratios,
                    two_boxes_for_ar1=two_boxes_for_ar1,
                    steps=steps,
                    offsets=offsets,
                    clip_boxes=clip_boxes,
                    variances=variances,
                    normalize_coords=normalize_coords)
# 2: Load the trained weights into the model.
model = Model(inputs=code, outputs=ssd)
# TODO: Set the path of the trained weights.
weights_path = 'results/cs_ssd_8_.5/best.h5'

model.load_weights(weights_path, by_name=True)

# 3: Compile the model so that Keras won't complain the next time you load it.

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

# define the input to the encoder
x = Input(shape=(h, w, 3))
#x1 = Lambda(lambda c: tf.image.rgb_to_grayscale(c))(x)
encoded = Conv2D(n_B, kernel_size = (B,B), strides=(B,B), activation=None, use_bias=False, name = 'E_conv')(x)
encoder = Model(inputs = x, outputs = encoded, name="encoder")

loss_fn = tf.keras.losses.MeanSquaredError()

encoder.load_weights(weights_path, by_name=True)
encoder.compile(optimizer=adam, loss=loss_fn)

decoder = CS_SSD.build_decoder_branch(code)
decoder_model = Model(inputs=code, outputs=decoder)
decoder_model.compile(optimizer=adam, loss=loss_fn)
decoder_model.load_weights(weights_path, by_name=True)


"""## 2. Create a data generator for the evaluation dataset

Instantiate a `DataGenerator` that will serve the evaluation dataset during the prediction phase.
"""

dataset = DataGenerator()

# TODO: Set the paths to the dataset here.
Pascal_VOC_dataset_images_dir = 'dataset/VOCdevkit/VOC2007/JPEGImages/'
Pascal_VOC_dataset_annotations_dir = 'dataset/VOCdevkit/VOC2007/Annotations/'
Pascal_VOC_dataset_image_set_filename = 'dataset/VOCdevkit/VOC2007/ImageSets/Main/test.txt'

# The XML parser needs to now what object class names to look for and in which order to map them to integers.
classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

dataset.parse_xml(images_dirs=[Pascal_VOC_dataset_images_dir],
                  image_set_filenames=[Pascal_VOC_dataset_image_set_filename],
                  annotations_dirs=[Pascal_VOC_dataset_annotations_dir],
                  classes=classes,
                  include_classes='all',
                  exclude_truncated=False,
                  exclude_difficult=False,
                  ret=False)

"""## 3. Run the evaluation

Now that we have instantiated a model and a data generator to serve the dataset, we can set up the evaluator and run the evaluation.

The evaluator is quite flexible: It can compute the average precisions according to the Pascal VOC pre-2010 algorithm, which samples 11 equidistant points of the precision-recall curves, or according to the Pascal VOC post-2010 algorithm, which integrates numerically over the entire precision-recall curves instead of sampling a few individual points. You could also change the number of sampled recall points or the required IoU overlap for a prediction to be considered a true positive, among other things. Check out the `Evaluator`'s documentation for details on all the arguments.

In its default settings, the evaluator's algorithm is identical to the official Pascal VOC pre-2010 Matlab detection evaluation algorithm, so you don't really need to tweak anything unless you want to.

The evaluator roughly performs the following steps: It runs predictions over the entire given dataset, then it matches these predictions to the ground truth boxes, then it computes the precision-recall curves for each class, then it samples 11 equidistant points from these precision-recall curves to compute the average precision for each class, and finally it computes the mean average precision over all classes.
"""
"""
evaluator = Evaluator(encoder=encoder,
                        model=model,
                      n_classes=n_classes,
                      data_generator=dataset,
                      model_mode=model_mode)

results = evaluator(img_height=img_height,
                    img_width=img_width,
                    batch_size=8,
                    data_generator_mode='resize',
                    round_confidences=False,
                    matching_iou_threshold=0.5,
                    border_pixels='include',
                    sorting_algorithm='quicksort',
                    average_precision_mode='sample',
                    num_recall_points=11,
                    ignore_neutral_boxes=True,
                    return_precisions=True,
                    return_recalls=True,
                    return_average_precisions=True,
                    verbose=True)

mean_average_precision, average_precisions, precisions, recalls = results
"""

"""## 4. Visualize the results

Let's take a look:

print("mAP and AP according to the Pascal VOC pre-2010 algorithm (which samples 11 equidistant points of the precision-recall curves) :\n")

for i in range(1, len(average_precisions)):
    print("{:<14}{:<6}{}".format(classes[i], 'AP', round(average_precisions[i], 3)))
print()
print("{:<14}{:<6}{}".format('','mAP', round(mean_average_precision, 3)))

"""
"""
m = max((n_classes + 1) // 2, 2)
n = 2

fig, cells = plt.subplots(m, n, figsize=(n*8,m*8))
for i in range(m):
    for j in range(n):
        if n*i+j+1 > n_classes: break
        cells[i, j].plot(recalls[n*i+j+1], precisions[n*i+j+1], color='blue', linewidth=1.0)
        cells[i, j].set_xlabel('recall', fontsize=14)
        cells[i, j].set_ylabel('precision', fontsize=14)
        cells[i, j].grid(True)
        cells[i, j].set_xticks(np.linspace(0,1,11))
        cells[i, j].set_yticks(np.linspace(0,1,11))
        cells[i, j].set_title("{}, AP: {:.3f}".format(classes[n*i+j+1], average_precisions[n*i+j+1]), fontsize=16)
fig.savefig("results/cs_ssd_8_.5/precision_recall")
"""
"""## 5. Advanced use

`Evaluator` objects maintain copies of all relevant intermediate results like predictions, precisions and recalls, etc., so in case you want to experiment with different parameters, e.g. different IoU overlaps, there is no need to compute the predictions all over again every time you make a change to a parameter. Instead, you can only update the computation from the point that is affected onwards.

The evaluator's `__call__()` method is just a convenience wrapper that executes its other methods in the correct order. You could just call any of these other methods individually as shown below (but you have to make sure to call them in the correct order).

Note that the example below uses the same evaluator object as above. Say you wanted to compute the Pascal VOC post-2010 'integrate' version of the average precisions instead of the pre-2010 version computed above. The evaluator object still has an internal copy of all the predictions, and since computing the predictions makes up the vast majority of the overall computation time and since the predictions aren't affected by changing the average precision computation mode, we skip computing the predictions again and instead only compute the steps that come after the prediction phase of the evaluation. We could even skip the matching part, since it isn't affected by changing the average precision mode either. In fact, we would only have to call `compute_average_precisions()` `compute_mean_average_precision()` again, but for the sake of illustration we'll re-do the other computations, too.
"""
"""
evaluator.get_num_gt_per_class(ignore_neutral_boxes=True,
                               verbose=True,
                               ret=False)

evaluator.match_predictions(ignore_neutral_boxes=True,
                            matching_iou_threshold=0.5,
                            border_pixels='include',
                            sorting_algorithm='quicksort',
                            verbose=True,
                            ret=False)

precisions, recalls = evaluator.compute_precision_recall(verbose=True, ret=True)

average_precisions = evaluator.compute_average_precisions(mode='integrate',
                                                          num_recall_points=11,
                                                          verbose=True,
                                                          ret=True)

mean_average_precision = evaluator.compute_mean_average_precision(ret=True)

print("mAP and AP according to the Pascal VOC post-2010 algorithm (which integrates numerically over the entire precision-recall curves) :\n")

for i in range(1, len(average_precisions)):
    print("{:<14}{:<6}{}".format(classes[i], 'AP', round(average_precisions[i], 3)))
print()
print("{:<14}{:<6}{}".format('','mAP', round(mean_average_precision, 3)))



"""
# For the validation generator:
convert_to_3_channels = ConvertTo3Channels()
resize = Resize(height=img_height, width=img_width)

# 1: Set the generator for the predictions.

predict_generator = dataset.generate(batch_size=32,
                                         shuffle=False,
                                         transformations=[convert_to_3_channels,
                                                          resize],
                                         label_encoder=None,
                                         returns={'processed_images',
                                                  'filenames',
                                                  'inverse_transform',
                                                  'original_images',
                                                  'original_labels'},
                                         keep_images_without_gt=False)

fps_ssd = []
decoding_time = []
encoding_time = []

# 2: Generate samples.

# batch_images, batch_gray, batch_filenames, batch_inverse_transforms, batch_original_images, batch_original_labels = next(test_generator)
batch_images, batch_filenames, batch_inverse_transforms, batch_original_images, batch_original_labels = next(predict_generator)

batch_images = batch_images-127.5
batch_images = batch_images/127.5
"""
i = 57 # Which batch item to look at

# print("Image:", batch_filenames[i])
print()
print("Ground truth boxes:\n")
print(np.array(batch_original_labels[i]))


for i in range(32):
  t_start = time.time()
  # predict for 32
  encoded = encoder.predict(batch_images[i][np.newaxis, :, :, :])
  encoding_time.append((time.time() - t_start))
  t_start = time.time()
  predictions = model.predict(encoded)
  fps_ssd.append(1/(time.time() - t_start))
  t_start = time.time()
  # predict
  decoded = decoder_model.predict(encoded)
  decoding_time.append((time.time() - t_start))
  
print("encoded shape : " , encoded.shape)  
print("decoded shape : " , decoded.shape)


fps_ssd = fps_ssd[1:-1]
decoding_time = decoding_time[1:-1]
encoding_time = encoding_time[1:-1]


a = mean(fps_ssd)
b = mean(encoding_time)
c = mean(decoding_time)

print('mean fps = ' ,a , "\n")

print('mean encoding_time = ' , b , "\n")

print('mean decoding_time = ' , c , "\n")
"""

"""
np.set_printoptions(precision=2, suppress=True, linewidth=90)
print("Predicted boxes:\n")
print('   class   conf xmin   ymin   xmax   ymax')
print(y_pred_decoded_inv[i])
"""
"""
orig_images = [] # Store the images here.
input_images = [] # Store resized versions of the images here.

# We'll only load one image in this example.
img_path = '/home/imene/Bureau/ssd_v0/dataset/VOCdevkit/VOC2007/JPEGImages/000058.jpg'

orig_images.append(imread(img_path))
img = image.load_img(img_path, target_size=(img_height, img_width))
img = image.img_to_array(img) 
input_images.append(img)
input_images = np.array(input_images)


# some figures
encoded = encoder.predict(input_images)
decoded = decoder_model.predict(encoded)
y_pred = model.predict(encoded)

my_dpi=100

confidence_threshold = 0.5

y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

np.set_printoptions(precision=2, suppress=True, linewidth=90)
print("Predicted boxes:\n")
print('   class   conf xmin   ymin   xmax   ymax')
print(y_pred_thresh[0])

# Display the image and draw the predicted boxes onto it.

# Set the colors for the bounding boxes
colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

plt.figure(figsize=(20,12))
plt.imshow(orig_images[0])

current_axis = plt.gca()

for box in y_pred_thresh[0]:
    # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
    xmin = box[2] * orig_images[0].shape[1] / img_width
    ymin = box[3] * orig_images[0].shape[0] / img_height
    xmax = box[4] * orig_images[0].shape[1] / img_width
    ymax = box[5] * orig_images[0].shape[0] / img_height
    color = colors[int(box[0])]
    label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
    current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))  
    current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})

plt.savefig('results/cs_ssd_8_.1/images/a', dpi=my_dpi)

"""
"""
psnr, ssim = psnr_ssim_batch(batch_images,decoded)

print('fro the whole batch,,, psnr = ', psnr, 'and ssim = ', ssim)
"""
i=0
# some figures
encoded = encoder.predict(batch_images)
decoded = decoder_model.predict(encoded)
y_pred = model.predict(encoded)

my_dpi=100
fig = plt.figure(figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi)

ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
plt.imshow(decoded[i])
plt.savefig('results/cs_ssd_8_.5/images/decoded', dpi=my_dpi)

confidence_threshold = 0.5

# Perform confidence thresholding.
y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

# Convert the predictions for the original image.
y_pred_thresh_inv = apply_inverse_transforms(y_pred_thresh, batch_inverse_transforms)

np.set_printoptions(precision=2, suppress=True, linewidth=90)
print("Predicted boxes:\n")
print('   class   conf xmin   ymin   xmax   ymax')
print(y_pred_thresh_inv[i])




fig = plt.figure(figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi)

ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)


#plt.figure(figsize=(20,12))
#plt.grid(False)
z = batch_original_images[i]
z = cv2.resize(z, dsize=(img_height,img_width))

#plt.figure(figsize=(20,12))
#plt.grid(False)
plt.imshow(z)
plt.savefig('results/cs_ssd_8_.5/images/nogt', dpi=my_dpi)


# Set the colors for the bounding boxes
colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()




fig = plt.figure(figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi)

ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)

plt.imshow(z)

current_axis = plt.gca()

for box in batch_original_labels[i]:
    xmin = box[1]* img_width / batch_original_images[0].shape[1]
    ymin = box[2] * img_height / batch_original_images[0].shape[0]
    xmax = box[3]* img_width / batch_original_images[0].shape[1]
    ymax = box[4] * img_height / batch_original_images[0].shape[0]
    label = '{}'.format(classes[int(box[0])])
    current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='green', fill=False, linewidth=2))  
    #current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':'green', 'alpha':1.0})


plt.savefig('results/cs_ssd_8_.5/images/gt', dpi=my_dpi)

fig = plt.figure(figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi)

ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)

plt.imshow(decoded[i])

current_axis = plt.gca()

for box in y_pred_thresh_inv[i]:
    xmin = box[2] * img_width / batch_original_images[0].shape[1]
    ymin = box[3] * img_height / batch_original_images[0].shape[0]
    xmax = box[4] * img_width /  batch_original_images[0].shape[1]
    ymax = box[5] * img_height / batch_original_images[0].shape[0]
    color = colors[int(box[0])]
    label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
    current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))  
    #current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})

plt.savefig('results/cs_ssd_8_.5/images/recons_pred', dpi=my_dpi)

images_per_row = 16
# This is the number of features in the feature map
n_features = n_B
# The feature map has shape (1, size, size, n_features)
size = n_blocks

# We will tile the activation channels in this matrix
n_cols = n_features // images_per_row
display_grid = np.zeros((size * n_cols, images_per_row * size))
print(encoded[i].shape)
    # We'll tile each filter into this big horizontal grid
for col in range(n_cols):
    for row in range(images_per_row):
        channel_image = encoded[i, :, :, col * images_per_row + row]
        # Post-process the feature to make it visually palatable
        channel_image -= channel_image.mean()
        channel_image /= channel_image.std()
        channel_image *= 64
        channel_image += 128
        channel_image = np.clip(channel_image, 0, 255).astype('uint8')
        display_grid[col * size : (col + 1) * size,
                      row * size : (row + 1) * size] = channel_image

# Display the grid

scale = 1. / size
plt.figure(figsize=(scale * display_grid.shape[1],
                    scale * display_grid.shape[0]))
plt.title('Compressed representation')
plt.grid(False)
# plt.savefig('Csnet prediction for test image {}'.format(i))
plt.imshow(display_grid, aspect='auto', cmap='viridis')
plt.savefig('results/cs_ssd_8_.5/images/encoded', dpi=my_dpi)

# fig=plt.figure(figsize=(20,20))
nbligne = floor(n_B/8)+1
# print(nbligne)
for m in range(n_B):
    # subplot = fig.add_subplot(nbligne,min(n_B,8),m+1)
    plt.figure(figsize=(20,12))
    plt.axis("off")
    channel_image = encoded[i, :, :, m]
    # Post-process the feature to make it visually palatable
    channel_image -= channel_image.mean()
    channel_image /= channel_image.std()
    channel_image *= 64
    channel_image += 128
    channel_image = np.clip(channel_image, 0, 255).astype('uint8')
    plt.imshow(channel_image)
    plt.savefig('results/cs_ssd_8_.5/images/encoded{}'.format(m))
    # subplot.title.set_text(('Compressed Channel {}'.format(m)))

