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

import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import numpy as np
from PIL import Image
import cv2
from skimage import io
from utils import cross_domain_param as param


class NormalizeImageDict(object):

    def __init__(self, image_keys, normalizeRange=True):
        self.image_keys = image_keys
        self.normalizeRange = normalizeRange
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, sample):
        for key in self.image_keys:
            if self.normalizeRange:
                sample[key] /= 255
            sample[key] = self.normalize(sample[key])
        return sample


class MyDataSet(data.Dataset):
    NumFileList = 0

    def __init__(self, usedata, transform1=None, transform2=None):
        self.data_path = usedata.dir_path
        self.face_data_path = './dataset/gan/'
        self.parse = usedata.parse
        if self.parse == 'Photo':
            self.fake_data_path = os.path.join(self.face_data_path, 'P2C')
        elif self.parse == 'Caricature':
            self.fake_data_path = os.path.join(self.face_data_path, 'C2P')
        self.name_list = usedata.names
        self.anna_list = usedata.annas
        self.visual_list = usedata.visuals
        self.usedata = usedata
        self.num_attribute = usedata.num_attribute
        self.transform1 = transform1
        self.transform2 = transform2

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        imgname = self.name_list[idx]
        attribute = self.anna_list[idx]
        visual = self.visual_list[idx]

        class_labels = [0 for _ in param.exclution_groups]
        index = 0
        for i, x in enumerate(param.exclution_groups):
            sub = attribute[index:index + x]
            sub_v = np.array(visual[index:index + x])
            if np.sum(sub_v) == 0:
                class_labels[i] = -1
            else:
                if x > 1:
                    try:
                        label = sub.index(1)
                        class_labels[i] = label
                    except ValueError as e:
                        print(e)
                else:
                    class_labels[i] = attribute[index]
            index = index + x

        imgname = self.usedata.getPath(imgname)
        img_path = os.path.join(self.data_path, imgname)
        fake_img_path = os.path.join(self.fake_data_path, imgname)
        try:
            input = io.imread(img_path)
            fake_input = io.imread(fake_img_path)
        except FileNotFoundError:
            img_path = img_path[:-4] + '.png'
            input = io.imread(img_path)
            fake_img_path = fake_img_path[:-4] + '.png'
            fake_input = io.imread(fake_img_path)
        if input.ndim < 3:
            input = cv2.cvtColor(input, cv2.COLOR_GRAY2RGB)
        if fake_input.ndim < 3:
            fake_input = cv2.cvtColor(fake_input, cv2.COLOR_GRAY2RGB)

        if self.transform1:
            PILinput = Image.fromarray(input)
            tranPILinput = self.transform1(PILinput)
            inp = np.asarray(tranPILinput)

            fake_PILinput = Image.fromarray(fake_input)
            fake_tranPILinput = self.transform1(fake_PILinput)
            fake_inp = np.asarray(fake_tranPILinput)
        else:
            inp = cv2.resize(input, (224, 224))
            fake_inp = cv2.resize(fake_input, (224, 224))

        sample = {
            'name': imgname,
            'image': inp,
            'attribute': attribute,
            'visual': visual,
            'class_label': class_labels,
            'fake_image': fake_inp
        }
        # print(sample)
        if self.transform2:
            sample = self.transform2(sample)
        return sample


class ToTensorDict(object):
    # Convert ndarrays in sample to Tensors.
    def __call__(self, sample):
        imgname = sample['name']
        image = sample['image']
        attribute = sample['attribute']
        visual = sample['visual']
        class_label = sample['class_label']
        fake_image = sample['fake_image']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        fake_image = fake_image.transpose((2, 0, 1))
        img = torch.from_numpy(image)
        fake_img = torch.from_numpy(fake_image)
        attri = torch.from_numpy(np.asarray(attribute))
        vis = torch.from_numpy(np.asarray(visual))
        cls_lab = torch.from_numpy(np.asarray(class_label))
        return {
            'name': imgname,
            'image': img.float(),
            'attribute': attri.float(),
            'visual': vis.float(),
            'class_label': cls_lab.float(),
            'fake_image': fake_img.float()
        }
