'''
A Keras port of the original Caffe SSD300 network.

Copyright (C) 2018 Pierluigi Ferrari

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from __future__ import division
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Activation, Conv2D, MaxPooling2D, ZeroPadding2D, Reshape, Concatenate
from tensorflow.keras.layers import MaxPooling2D, BatchNormalization, ELU
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K

from keras_layers.keras_layer_AnchorBoxes_orig import AnchorBoxes
from keras_layers.keras_layer_L2Normalization import L2Normalization
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.resnet_blocks import conv_block, identity_block

def ssd_300(image_size,
            n_classes,
            mode='training',
            l2_regularization=0.0005,
            min_scale=None,
            max_scale=None,
            scales=None,
            aspect_ratios_global=None,
            aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                     [1.0, 2.0, 0.5],
                                     [1.0, 2.0, 0.5]],
            kernel_initializer: str = 'he_normal',
            two_boxes_for_ar1=True,
            steps=[8, 16, 32, 64, 100, 300],
            offsets=None,
            clip_boxes=False,
            variances=[0.1, 0.1, 0.2, 0.2],
            coords='centroids',
            normalize_coords=True,
            subtract_mean=[123, 117, 104],
            divide_by_stddev=None,
            swap_channels=[2, 1, 0],
            confidence_thresh=0.01,
            iou_threshold=0.45,
            top_k=200,
            nms_max_output_size=400,
            return_predictor_sizes=False):
   
    # resnet backbone
    n_classes += 1  # Account for the background class.
    n_predictor_layers = 6
    
    
    # Set the aspect ratios for each predictor layer. These are only needed for the anchor box layers.
    if aspect_ratios_per_layer:
        aspect_ratios = aspect_ratios_per_layer
    else:
        aspect_ratios = [aspect_ratios_global] * n_predictor_layers
        
    l2_reg = l2_regularization # Make the internal name shorter.
    #  Scales values for the anchor boxes
    if scales is None:
        scales = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05]

    # The number of anchor boxes for the given output layer.
    n_boxes = [4, 6, 6, 6, 4, 4]

    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]
    
    
    input_layer = Input(shape=(img_height, img_width, img_channels))
    
    # 1
    x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(input_layer)
    x = Conv2D(64, (7, 7), strides=(2, 2),padding='valid', kernel_initializer='he_normal',kernel_regularizer=l2(l2_reg),name='conv1')(x)
    x = BatchNormalization(axis=3, momentum=0.9, epsilon=1e-5, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a',strides=(1, 1), kernel_reg=l2_reg)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', kernel_reg=l2_reg)
    x = identity_block(x, 3, [64, 64, 256], stage=2,block='c', kernel_reg=l2_reg)

    # 3
    x = conv_block(x, 3, [128, 128, 512], stage=3,block='a', kernel_reg=l2_reg)
    x = identity_block(x, 3, [128, 128, 512], stage=3,block='b', kernel_reg=l2_reg)
    x = identity_block(x, 3, [128, 128, 512], stage=3,block='c', kernel_reg=l2_reg)
    block4_conv3 = identity_block(x, 3, [128, 128, 512], stage=3, block='d', kernel_reg=l2_reg)

    # 4
    x = conv_block(block4_conv3, 3, [256, 256, 1024],stage=4, block='a', kernel_reg=l2_reg)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', kernel_reg=l2_reg)
    x = identity_block(x, 3, [256, 256, 1024], stage=4,block='c', kernel_reg=l2_reg)
    x = identity_block(x, 3, [256, 256, 1024], stage=4,block='d', kernel_reg=l2_reg)
    x = identity_block(x, 3, [256, 256, 1024], stage=4,block='e', kernel_reg=l2_reg)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', kernel_reg=l2_reg)

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', kernel_reg=l2_reg, strides=(1, 1))
    x = identity_block(x, 3, [512, 512, 2048], stage=5,block='b', kernel_reg=l2_reg)
    fc7 = identity_block(x, 3, [512, 512, 2048],stage=5, block='c', kernel_reg=l2_reg)
 
    # additional blocks
    

    conv6_1 = Conv2D(256, (1, 1), activation='relu', padding='same',
                     kernel_initializer=kernel_initializer, kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg), name='conv6_1')(fc7)
    conv6_1 = ZeroPadding2D(padding=((1, 1), (1, 1)),
                            name='conv6_padding')(conv6_1)
    conv6_2 = Conv2D(512, (3, 3), strides=(2, 2), kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg), activation='relu', padding='valid',
                     kernel_initializer=kernel_initializer, name='conv6_2')(conv6_1)

    conv7_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg),
                     kernel_initializer=kernel_initializer, name='conv7_1')(conv6_2)
    conv7_1 = ZeroPadding2D(padding=((1, 1), (1, 1)),
                            name='conv7_padding')(conv7_1)
    conv7_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg), padding='valid',
                     kernel_initializer=kernel_initializer, name='conv7_2')(conv7_1)

    conv8_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg),
                     kernel_initializer=kernel_initializer, name='conv8_1')(conv7_2)
    conv8_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg), padding='valid',
                     kernel_initializer=kernel_initializer, name='conv8_2')(conv8_1)

    conv9_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg),
                     kernel_initializer=kernel_initializer, name='conv9_1')(conv8_2)
    conv9_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg), padding='valid',
                     kernel_initializer=kernel_initializer, name='conv9_2')(conv9_1)

    # Feed block4_conv3 into the L2 normalization layer
    block4_conv3_norm = L2Normalization(
        gamma_init=20, name='block4_conv3_norm')(block4_conv3)

    # Build the convolutional predictor layers on top of the base network
    # We predict `n_classes` confidence values for each box, hence the confidence predictors have depth `n_boxes * n_classes`
    # Output shape of the confidence layers: `(batch, height, width, n_boxes * n_classes)`
    block4_conv3_norm_mbox_conf = Conv2D(n_boxes[0] * n_classes, (3, 3), padding='same', kernel_initializer=kernel_initializer, kernel_regularizer=l2(l2_reg),                              bias_regularizer=l2(l2_reg),name='block4_conv3_norm_mbox_conf')(block4_conv3_norm)
    fc7_mbox_conf = Conv2D(n_boxes[1] * n_classes, (3, 3), padding='same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg),
                           kernel_initializer=kernel_initializer, name='fc7_mbox_conf')(fc7)
    conv6_2_mbox_conf = Conv2D(n_boxes[2] * n_classes, (3, 3), padding='same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg),
                               kernel_initializer=kernel_initializer, name='conv6_2_mbox_conf')(conv6_2)
    conv7_2_mbox_conf = Conv2D(n_boxes[3] * n_classes, (3, 3), padding='same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg),
                               kernel_initializer=kernel_initializer, name='conv7_2_mbox_conf')(conv7_2)
    conv8_2_mbox_conf = Conv2D(n_boxes[4] * n_classes, (3, 3), padding='same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg),
                               kernel_initializer=kernel_initializer, name='conv8_2_mbox_conf')(conv8_2)
    conv9_2_mbox_conf = Conv2D(n_boxes[5] * n_classes, (3, 3), padding='same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg),
                               kernel_initializer=kernel_initializer, name='conv9_2_mbox_conf')(conv9_2)

    # We predict 4 box coordinates for each box, hence the localization predictors have depth `n_boxes * 4`
    # Output shape of the localization layers: `(batch, height, width, n_boxes * 4)`
    block4_conv3_norm_mbox_loc = Conv2D(n_boxes[0] * 4, (3, 3), padding='same', kernel_initializer=kernel_initializer, kernel_regularizer=l2(l2_reg), bias_regularizer=l2               (l2_reg),name='block4_conv3_norm_mbox_loc')(block4_conv3_norm)
    fc7_mbox_loc = Conv2D(n_boxes[1] * 4, (3, 3), padding='same', kernel_initializer=kernel_initializer, kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg),
                          name='fc7_mbox_loc')(fc7)
    conv6_2_mbox_loc = Conv2D(n_boxes[2] * 4, (3, 3), padding='same', kernel_initializer=kernel_initializer, kernel_regularizer=l2(l2_reg),
        bias_regularizer=l2(l2_reg),name='conv6_2_mbox_loc')(conv6_2)
    conv7_2_mbox_loc = Conv2D(n_boxes[3] * 4, (3, 3), padding='same', kernel_initializer=kernel_initializer, kernel_regularizer=l2(l2_reg), 
        bias_regularizer=l2(l2_reg),name='conv7_2_mbox_loc')(conv7_2)
    conv8_2_mbox_loc = Conv2D(n_boxes[4] * 4, (3, 3), padding='same', kernel_initializer=kernel_initializer, kernel_regularizer=l2(l2_reg),
        bias_regularizer=l2(l2_reg),name='conv8_2_mbox_loc')(conv8_2)
    conv9_2_mbox_loc = Conv2D(n_boxes[5] * 4, (3, 3), padding='same', kernel_initializer=kernel_initializer, kernel_regularizer=l2(l2_reg),
        bias_regularizer=l2(l2_reg),name='conv9_2_mbox_loc')(conv9_2)

    # Generate the anchor (prior) boxes
    # Output shape of anchors: `(batch, height, width, n_boxes, 8)
    block4_conv3_norm_mbox_priorbox = AnchorBoxes(300, 300, this_scale=scales[0], next_scale=scales[1], aspect_ratios=aspect_ratios[0],
                                                  two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0], this_offsets=offsets[0], clip_boxes=clip_boxes,
        variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv4_3_norm_mbox_priorbox')(block4_conv3_norm_mbox_loc)
    fc7_mbox_priorbox = AnchorBoxes(300, 300, this_scale=scales[1], next_scale=scales[2], aspect_ratios=aspect_ratios[1],
                                    two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[
                                        1], this_offsets=offsets[1], clip_boxes=clip_boxes,
                                    variances=variances, coords=coords, normalize_coords=normalize_coords, name='fc7_mbox_priorbox')(fc7_mbox_loc)
    conv6_2_mbox_priorbox = AnchorBoxes(300, 300, this_scale=scales[2], next_scale=scales[3], aspect_ratios=aspect_ratios[2],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[
                                            2], this_offsets=offsets[2], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv6_2_mbox_priorbox')(conv6_2_mbox_loc)
    conv7_2_mbox_priorbox = AnchorBoxes(300, 300, this_scale=scales[3], next_scale=scales[4], aspect_ratios=aspect_ratios[3],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[
                                            3], this_offsets=offsets[3], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv7_2_mbox_priorbox')(conv7_2_mbox_loc)
    conv8_2_mbox_priorbox = AnchorBoxes(300, 300, this_scale=scales[4], next_scale=scales[5], aspect_ratios=aspect_ratios[4],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[
                                            4], this_offsets=offsets[4], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv8_2_mbox_priorbox')(conv8_2_mbox_loc)
    conv9_2_mbox_priorbox = AnchorBoxes(300, 300, this_scale=scales[5], next_scale=scales[6], aspect_ratios=aspect_ratios[5],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[
                                            5], this_offsets=offsets[5], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv9_2_mbox_priorbox')(conv9_2_mbox_loc)


    # Reshape
    # Reshape the class predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, n_classes)`
    # We want the classes isolated in the last axis to perform softmax on them
    block4_conv3_norm_mbox_conf_reshape = Reshape((-1, n_classes), name='block4_conv3_norm_mbox_conf_reshape')(block4_conv3_norm_mbox_conf)
    fc7_mbox_conf_reshape = Reshape((-1, n_classes), name='fc7_mbox_conf_reshape')(fc7_mbox_conf)
    conv6_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv6_2_mbox_conf_reshape')(conv6_2_mbox_conf)
    conv7_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv7_2_mbox_conf_reshape')(conv7_2_mbox_conf)
    conv8_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv8_2_mbox_conf_reshape')(conv8_2_mbox_conf)
    conv9_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv9_2_mbox_conf_reshape')(conv9_2_mbox_conf)

    # Reshape the box predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, 4)`
    # We want the four box coordinates isolated in the last axis to compute the smooth L1 loss
    block4_conv3_norm_mbox_loc_reshape = Reshape((-1, 4), name='block4_conv3_norm_mbox_loc_reshape')(block4_conv3_norm_mbox_loc)
    fc7_mbox_loc_reshape = Reshape((-1, 4), name='fc7_mbox_loc_reshape')(fc7_mbox_loc)
    conv6_2_mbox_loc_reshape = Reshape((-1, 4), name='conv6_2_mbox_loc_reshape')(conv6_2_mbox_loc)
    conv7_2_mbox_loc_reshape = Reshape((-1, 4), name='conv7_2_mbox_loc_reshape')(conv7_2_mbox_loc)
    conv8_2_mbox_loc_reshape = Reshape((-1, 4), name='conv8_2_mbox_loc_reshape')(conv8_2_mbox_loc)
    conv9_2_mbox_loc_reshape = Reshape((-1, 4), name='conv9_2_mbox_loc_reshape')(conv9_2_mbox_loc)

    # Reshape the anchor box tensors, yielding 3D tensors of shape `(batch, height * width * n_boxes, 8)`
    block4_conv3_norm_mbox_priorbox_reshape = Reshape((-1, 8), name='block4_conv3_norm_mbox_priorbox_reshape')(block4_conv3_norm_mbox_priorbox)
    fc7_mbox_priorbox_reshape = Reshape((-1, 8), name='fc7_mbox_priorbox_reshape')(fc7_mbox_priorbox)
    conv6_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv6_2_mbox_priorbox_reshape')(conv6_2_mbox_priorbox)
    conv7_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv7_2_mbox_priorbox_reshape')(conv7_2_mbox_priorbox)
    conv8_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv8_2_mbox_priorbox_reshape')(conv8_2_mbox_priorbox)
    conv9_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv9_2_mbox_priorbox_reshape')(conv9_2_mbox_priorbox)

    # Concatenate the predictions from the different layers

    # Axis 0 (batch) and axis 2 (n_classes or 4, respectively) are identical for all layer predictions,
    # so we want to concatenate along axis 1, the number of boxes per layer
    # Output shape of `mbox_conf`: (batch, n_boxes_total, n_classes)
    mbox_conf = Concatenate(axis=1, name='mbox_conf')([block4_conv3_norm_mbox_conf_reshape,
                                                       fc7_mbox_conf_reshape,
                                                       conv6_2_mbox_conf_reshape,
                                                       conv7_2_mbox_conf_reshape,
                                                       conv8_2_mbox_conf_reshape,
                                                       conv9_2_mbox_conf_reshape])

    # Output shape of `mbox_loc`: (batch, n_boxes_total, 4)
    mbox_loc = Concatenate(axis=1, name='mbox_loc')([block4_conv3_norm_mbox_loc_reshape,
                                                     fc7_mbox_loc_reshape,
                                                     conv6_2_mbox_loc_reshape,
                                                     conv7_2_mbox_loc_reshape,
                                                     conv8_2_mbox_loc_reshape,
                                                     conv9_2_mbox_loc_reshape])

    # Output shape of `mbox_priorbox`: (batch, n_boxes_total, 8)
    mbox_priorbox = Concatenate(axis=1, name='mbox_priorbox')([block4_conv3_norm_mbox_priorbox_reshape,
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
    predictions = Concatenate(axis=-1, name='predictions_ssd')([mbox_conf_softmax, mbox_loc, mbox_priorbox])

    if mode == 'training':
        model = Model(inputs=input_layer, outputs=predictions)
    elif mode == 'inference':
        decoded_predictions = DecodeDetections(confidence_thresh=confidence_thresh,
                                               iou_threshold=iou_threshold,
                                               top_k=top_k,
                                               nms_max_output_size=nms_max_output_size,
                                               coords=coords,
                                               normalize_coords=normalize_coords,
                                               img_height=img_height,
                                               img_width=img_width,
                                               name='decoded_predictions')(predictions)
        model = Model(inputs=input_layer, outputs=decoded_predictions)
    else:
        raise ValueError(
            "`mode` must be one of 'training' or 'inference', but received '{}'.".format(mode))

    return model