from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import torch
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import sensor_msgs.msg
import sys
import os
sys.path.append(os.path.abspath(".."))
from utils.utils import *

def default_loader(image, resize_shape, change_scale = False):
    trans = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
    ])
    # image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (h_original, w_original) = image.shape
    image = cv2.resize(image, dsize=(resize_shape,resize_shape), interpolation=cv2.INTER_CUBIC)


    np_image_data = np.asarray(image)
    image_tensor = trans(np_image_data)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.permute(1,0,2,3)
    
    return image_tensor

def get_gt_tensor(this_gt, size):
    this_gt = this_gt +180
    gt_tensor_self = torch.zeros(size,size)
    angle_convert = this_gt*size/360
    angle_index = angle_convert//1 + (angle_convert%1+0.5)//1
    if angle_index.long() == size:
        angle_index = size-1
        gt_tensor_self[angle_index,0] = 1
    else:
        gt_tensor_self[angle_index.long(),0] = 1
    # print("angle_index", angle_index)

    return gt_tensor_self

def cv_bridge( img_msg):
        """ cv_bridge does not support python3 and this is extracted from the
            cv_bridge file to convert the msg::Img to np.ndarray
        """
        color_msg = img_msg
        # print("encoding=====", img_msg.encoding)
        #set different dtype based on different encoding type
        if 'C' in img_msg.encoding:
            map_dtype = {'U': 'uint', 'S': 'int', 'F': 'float'}
            dtype_str, n_channels_str = img_msg.encoding.split('C')
            n_channels = int(n_channels_str)
            dtype = np.dtype(map_dtype[dtype_str[-1]] + dtype_str[:-1])
        elif img_msg.encoding == 'bgr8' or img_msg.encoding == 'rgb8':
            n_channels = 3
            dtype = np.dtype('uint8')
        elif img_msg.encoding == 'mono8':
            n_channels = 1
            dtype = np.dtype('uint8')


        dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')

        if n_channels == 1:
            img1 = np.ndarray(shape=(img_msg.height, img_msg.width),
                           dtype=dtype, buffer=img_msg.data)
        else:
            if(type(img_msg.data) == str):
                img1 = np.ndarray(shape=(img_msg.height, img_msg.width, n_channels),
                               dtype=dtype, buffer=img_msg.data.encode())
            else:
                img1 = np.ndarray(shape=(img_msg.height, img_msg.width, n_channels),
                               dtype=dtype, buffer=img_msg.data)

        # img1 = np.ndarray(shape=(img_msg.height, img_msg.width, n_channels),
        #                 dtype=dtype)#, buffer=img_msg.data)
        img1 = np.squeeze(img1)
        if img_msg.is_bigendian == (sys.byteorder == 'little'):
            img1 = img1.byteswap().newbyteorder()

        # convert RGB to BGR
        if img_msg.encoding == 'rgb8':
            img0 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        elif img_msg.encoding == 'mono8':
            img0 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        else:
            img0 = img1
        return img0

numpy_type_to_cvtype = {'uint8': '8U', 'int8': '8S', 'uint16': '16U',
                                    'int16': '16S', 'int32': '32S', 'float32': '32F',
                                    'float64': '64F'}
numpy_type_to_cvtype.update(dict((v, k) for (k, v) in numpy_type_to_cvtype.items()))

cvdepth_to_numpy_depth = {cv2.CV_8U: 'uint8', cv2.CV_8S: 'int8', cv2.CV_16U: 'uint16',
                                       cv2.CV_16S: 'int16', cv2.CV_32S:'int32', cv2.CV_32F:'float32',
                                       cv2.CV_64F: 'float64'}

cvtype_to_name = {}

def dtype_with_channels_to_cvtype2(dtype, n_channels):
    return '%sC%d' % (numpy_type_to_cvtype[dtype.name], n_channels)

def cvtype2_to_dtype_with_channels(cvtype):
    from cv_bridge.boost.cv_bridge_boost import CV_MAT_CNWrap, CV_MAT_DEPTHWrap
    return cvdepth_to_numpy_depth[CV_MAT_DEPTHWrap(cvtype)], CV_MAT_CNWrap(cvtype)

# def encoding_to_cvtype2(encoding):
#     from cv_bridge.boost.cv_bridge_boost import getCvType

#     try:
#         return getCvType(encoding)
#     except RuntimeError as e:
#         raise CvBridgeError(e)

def cv2_to_imgmsg(cvim, encoding = "passthrough"):
    """
    Convert an OpenCV :cpp:type:`cv::Mat` type to a ROS sensor_msgs::Image message.
    :param cvim:      An OpenCV :cpp:type:`cv::Mat`
    :param encoding:  The encoding of the image data, one of the following strings:
        * ``"passthrough"``
        * one of the standard strings in sensor_msgs/image_encodings.h
    :rtype:           A sensor_msgs.msg.Image message
    :raises CvBridgeError: when the ``cvim`` has a type that is incompatible with ``encoding``
    If encoding is ``"passthrough"``, then the message has the same encoding as the image's OpenCV type.
    Otherwise desired_encoding must be one of the standard image encodings
    This function returns a sensor_msgs::Image message on success, or raises :exc:`cv_bridge.CvBridgeError` on failure.
    """
    
    if not isinstance(cvim, (np.ndarray, np.generic)):
        raise TypeError('Your input type is not a numpy array')
    img_msg = sensor_msgs.msg.Image()
    img_msg.height = cvim.shape[0]
    img_msg.width = cvim.shape[1]
    if len(cvim.shape) < 3:
        cv_type = dtype_with_channels_to_cvtype2(cvim.dtype, 1)
    else:
        cv_type = dtype_with_channels_to_cvtype2(cvim.dtype, cvim.shape[2])
    if encoding == "passthrough":
        img_msg.encoding = cv_type
    else:
        img_msg.encoding = encoding
        # Verify that the supplied encoding is compatible with the type of the OpenCV image
        # if cvtype_to_name[encoding_to_cvtype2(encoding)] != cv_type:
        #     raise CvBridgeError("encoding specified as %s, but image has incompatible type %s" % (encoding, cv_type))
    if cvim.dtype.byteorder == '>':
        img_msg.is_bigendian = True
    img_msg.data = cvim.tostring()
    img_msg.step = len(img_msg.data) // img_msg.height

    return img_msg
# Software License Agreement (BSD License)
#
# Copyright (c) 2011, Willow Garage, Inc.
# Copyright (c) 2016, Tal Regev.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# import sensor_msgs.msg

# class CvBridgeError(TypeError):
#     """
#     This is the error raised by :class:`cv_bridge.CvBridge` methods when they fail.
#     """
#     pass


# class CvBridge(object):
#     """
#     The CvBridge is an object that converts between OpenCV Images and ROS Image messages.
#        .. doctest::
#            :options: -ELLIPSIS, +NORMALIZE_WHITESPACE
#            >>> import cv2
#            >>> import numpy as np
#            >>> from cv_bridge import CvBridge
#            >>> br = CvBridge()
#            >>> dtype, n_channels = br.encoding_as_cvtype2('8UC3')
#            >>> im = np.ndarray(shape=(480, 640, n_channels), dtype=dtype)
#            >>> msg = br.cv2_to_imgmsg(im)  # Convert the image to a message
#            >>> im2 = br.imgmsg_to_cv2(msg) # Convert the message to a new image
#            >>> cmprsmsg = br.cv2_to_compressed_imgmsg(im)  # Convert the image to a compress message
#            >>> im22 = br.compressed_imgmsg_to_cv2(msg) # Convert the compress message to a new image
#            >>> cv2.imwrite("this_was_a_message_briefly.png", im2)
#     """

#     def __init__(self):
#         import cv2
#         self.cvtype_to_name = {}
#         self.cvdepth_to_numpy_depth = {cv2.CV_8U: 'uint8', cv2.CV_8S: 'int8', cv2.CV_16U: 'uint16',
#                                        cv2.CV_16S: 'int16', cv2.CV_32S:'int32', cv2.CV_32F:'float32',
#                                        cv2.CV_64F: 'float64'}

#         for t in ["8U", "8S", "16U", "16S", "32S", "32F", "64F"]:
#             for c in [1, 2, 3, 4]:
#                 nm = "%sC%d" % (t, c)
#                 self.cvtype_to_name[getattr(cv2, "CV_%s" % nm)] = nm

#         self.numpy_type_to_cvtype = {'uint8': '8U', 'int8': '8S', 'uint16': '16U',
#                                         'int16': '16S', 'int32': '32S', 'float32': '32F',
#                                         'float64': '64F'}
#         self.numpy_type_to_cvtype.update(dict((v, k) for (k, v) in self.numpy_type_to_cvtype.items()))

#     def dtype_with_channels_to_cvtype2(self, dtype, n_channels):
#         return '%sC%d' % (self.numpy_type_to_cvtype[dtype.name], n_channels)

#     def cvtype2_to_dtype_with_channels(self, cvtype):
#         from cv_bridge.boost.cv_bridge_boost import CV_MAT_CNWrap, CV_MAT_DEPTHWrap
#         return self.cvdepth_to_numpy_depth[CV_MAT_DEPTHWrap(cvtype)], CV_MAT_CNWrap(cvtype)

#     def encoding_to_cvtype2(self, encoding):
#         from cv_bridge.boost.cv_bridge_boost import getCvType

#         try:
#             return getCvType(encoding)
#         except RuntimeError as e:
#             raise CvBridgeError(e)

#     def encoding_to_dtype_with_channels(self, encoding):
#         return self.cvtype2_to_dtype_with_channels(self.encoding_to_cvtype2(encoding))

#     def compressed_imgmsg_to_cv2(self, cmprs_img_msg, desired_encoding = "passthrough"):
#         """
#         Convert a sensor_msgs::CompressedImage message to an OpenCV :cpp:type:`cv::Mat`.
#         :param cmprs_img_msg:   A :cpp:type:`sensor_msgs::CompressedImage` message
#         :param desired_encoding:  The encoding of the image data, one of the following strings:
#            * ``"passthrough"``
#            * one of the standard strings in sensor_msgs/image_encodings.h
#         :rtype: :cpp:type:`cv::Mat`
#         :raises CvBridgeError: when conversion is not possible.
#         If desired_encoding is ``"passthrough"``, then the returned image has the same format as img_msg.
#         Otherwise desired_encoding must be one of the standard image encodings
#         This function returns an OpenCV :cpp:type:`cv::Mat` message on success, or raises :exc:`cv_bridge.CvBridgeError` on failure.
#         If the image only has one channel, the shape has size 2 (width and height)
#         """
#         import cv2
#         import numpy as np

#         str_msg = cmprs_img_msg.data
#         buf = np.ndarray(shape=(1, len(str_msg)),
#                           dtype=np.uint8, buffer=cmprs_img_msg.data)
#         im = cv2.imdecode(buf, cv2.IMREAD_ANYCOLOR)

#         if desired_encoding == "passthrough":
#             return im

#         from cv_bridge.boost.cv_bridge_boost import cvtColor2

#         try:
#             res = cvtColor2(im, "bgr8", desired_encoding)
#         except RuntimeError as e:
#             raise CvBridgeError(e)

#         return res

#     def imgmsg_to_cv2(self, img_msg, desired_encoding = "passthrough"):
#         """
#         Convert a sensor_msgs::Image message to an OpenCV :cpp:type:`cv::Mat`.
#         :param img_msg:   A :cpp:type:`sensor_msgs::Image` message
#         :param desired_encoding:  The encoding of the image data, one of the following strings:
#            * ``"passthrough"``
#            * one of the standard strings in sensor_msgs/image_encodings.h
#         :rtype: :cpp:type:`cv::Mat`
#         :raises CvBridgeError: when conversion is not possible.
#         If desired_encoding is ``"passthrough"``, then the returned image has the same format as img_msg.
#         Otherwise desired_encoding must be one of the standard image encodings
#         This function returns an OpenCV :cpp:type:`cv::Mat` message on success, or raises :exc:`cv_bridge.CvBridgeError` on failure.
#         If the image only has one channel, the shape has size 2 (width and height)
#         """
#         import cv2
#         import numpy as np
#         dtype, n_channels = self.encoding_to_dtype_with_channels(img_msg.encoding)
#         dtype = np.dtype(dtype)
#         dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
#         if n_channels == 1:
#             im = np.ndarray(shape=(img_msg.height, img_msg.width),
#                            dtype=dtype, buffer=img_msg.data)
#         else:
#             if(type(img_msg.data) == str):
#                 im = np.ndarray(shape=(img_msg.height, img_msg.width, n_channels),
#                                dtype=dtype, buffer=img_msg.data.encode())
#             else:
#                 im = np.ndarray(shape=(img_msg.height, img_msg.width, n_channels),
#                                dtype=dtype, buffer=img_msg.data)
#         # If the byt order is different between the message and the system.
#         if img_msg.is_bigendian == (sys.byteorder == 'little'):
#             im = im.byteswap().newbyteorder()

#         if desired_encoding == "passthrough":
#             return im

#         from cv_bridge.boost.cv_bridge_boost import cvtColor2

#         try:
#             res = cvtColor2(im, img_msg.encoding, desired_encoding)
#         except RuntimeError as e:
#             raise CvBridgeError(e)

#         return res

#     def cv2_to_compressed_imgmsg(self, cvim, dst_format = "jpg"):
#         """
#         Convert an OpenCV :cpp:type:`cv::Mat` type to a ROS sensor_msgs::CompressedImage message.
#         :param cvim:      An OpenCV :cpp:type:`cv::Mat`
#         :param dst_format:  The format of the image data, one of the following strings:
#            * from http://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html
#            * from http://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#Mat imread(const string& filename, int flags)
#            * bmp, dib
#            * jpeg, jpg, jpe
#            * jp2
#            * png
#            * pbm, pgm, ppm
#            * sr, ras
#            * tiff, tif
#         :rtype:           A sensor_msgs.msg.CompressedImage message
#         :raises CvBridgeError: when the ``cvim`` has a type that is incompatible with ``format``
#         This function returns a sensor_msgs::Image message on success, or raises :exc:`cv_bridge.CvBridgeError` on failure.
#         """
#         import cv2
#         import numpy as np
#         if not isinstance(cvim, (np.ndarray, np.generic)):
#             raise TypeError('Your input type is not a numpy array')
#         cmprs_img_msg = sensor_msgs.msg.CompressedImage()
#         cmprs_img_msg.format = dst_format
#         ext_format = '.' + dst_format
#         try:
#             cmprs_img_msg.data = np.array(cv2.imencode(ext_format, cvim)[1]).tostring()
#         except RuntimeError as e:
#             raise CvBridgeError(e)

#         return cmprs_img_msg

#     def cv2_to_imgmsg(self, cvim, encoding = "passthrough"):
#         """
#         Convert an OpenCV :cpp:type:`cv::Mat` type to a ROS sensor_msgs::Image message.
#         :param cvim:      An OpenCV :cpp:type:`cv::Mat`
#         :param encoding:  The encoding of the image data, one of the following strings:
#            * ``"passthrough"``
#            * one of the standard strings in sensor_msgs/image_encodings.h
#         :rtype:           A sensor_msgs.msg.Image message
#         :raises CvBridgeError: when the ``cvim`` has a type that is incompatible with ``encoding``
#         If encoding is ``"passthrough"``, then the message has the same encoding as the image's OpenCV type.
#         Otherwise desired_encoding must be one of the standard image encodings
#         This function returns a sensor_msgs::Image message on success, or raises :exc:`cv_bridge.CvBridgeError` on failure.
#         """
#         import cv2
#         import numpy as np
#         if not isinstance(cvim, (np.ndarray, np.generic)):
#             raise TypeError('Your input type is not a numpy array')
#         img_msg = sensor_msgs.msg.Image()
#         img_msg.height = cvim.shape[0]
#         img_msg.width = cvim.shape[1]
#         if len(cvim.shape) < 3:
#             cv_type = self.dtype_with_channels_to_cvtype2(cvim.dtype, 1)
#         else:
#             cv_type = self.dtype_with_channels_to_cvtype2(cvim.dtype, cvim.shape[2])
#         if encoding == "passthrough":
#             img_msg.encoding = cv_type
#         else:
#             img_msg.encoding = encoding
#             # Verify that the supplied encoding is compatible with the type of the OpenCV image
#             if self.cvtype_to_name[self.encoding_to_cvtype2(encoding)] != cv_type:
#                 raise CvBridgeError("encoding specified as %s, but image has incompatible type %s" % (encoding, cv_type))
#         if cvim.dtype.byteorder == '>':
#             img_msg.is_bigendian = True
#         img_msg.data = cvim.tostring()
#         img_msg.step = len(img_msg.data) // img_msg.height

#         return img_msg