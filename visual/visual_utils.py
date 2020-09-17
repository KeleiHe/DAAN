#    Copyright 2020 Wen Ji & Kelei He (hkl@nju.edu.cn)
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import cv2
import numpy as np


def cal_nomalize_cam(attri_mask, image_path):
    image = cv2.imread(image_path, 1)
    image = np.float32(cv2.resize(image, (256, 256))) / 255
    image = image[16:240, 16:240]
    attri_mask = np.maximum(attri_mask, 0)
    attri_mask = cv2.resize(attri_mask, (224, 224))
    attri_mask = attri_mask - np.min(attri_mask)
    attri_mask = attri_mask / np.max(attri_mask)
    hm = cv2.applyColorMap(np.uint8(255 * attri_mask), cv2.COLORMAP_JET)
    hm = np.float32(hm) / 255
    cam = hm + np.float32(image)
    cam = cam / np.max(cam)
    return cam