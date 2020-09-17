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

import argparse
import torch
import os
parser = argparse.ArgumentParser(description='cross domain face attributes Training')
parser.add_argument('--source_dataset', dest='source_dataset',
                    help='which dataset to use, CelebA or  WebCaricature or Caricature or Photo for source set',
                    default='Photo', type=str)
parser.add_argument('--target_dataset', dest='target_dataset',
                    help='which dataset to use, CelebA or  WebCaricature or Caricature or Photo for target set',
                    default='Caricature', type=str)
parser.add_argument('--modelType', dest='modelType', help='basic, seperate, multiTask or cross_model',
                    default='basic', type=str)

parser.add_argument('-k', default=5, type=int, metavar='N',
                    help='k folds cross validation')
parser.add_argument('--lr', '--learning-rate', default=5e-2, type=list, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--device', default='3,2', type=str, metavar='G',
                    help='gpu device')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device
ROOT_PATH = './attention_consistency_result'
args.testname = '5cv_Label_Feat_Attention_ConsisModel-shareWeight_lr_5e-2_1_2e-2_1e-1_1e-1'
data_save_dir = ROOT_PATH + '/new_result_cross_model/' + str(args.source_dataset) + str(2) \
                + str(args.target_dataset) + "/cv_dataset/"
result_dir = ROOT_PATH + '/new_result_cross_model/' + str(args.source_dataset) + str(2) \
             + str(args.target_dataset)+'/'+args.testname+'/'

exclution_groups = [1, 3, 3, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 2, 1, 2, 2, 1, 2, 1, 1, 2, 2, 1, 4, 1, 2, 1, 3, 1, 1]

exclution_attris = [
        'gender',
        'race',
        'age',
        'wrinkle',
        'makeup',
        'bald',
        'largeForehead',
        'face',
        'narrowEye',
        'sleepEye',
        'slantEye',
        'sharpEye',
        'flabbyEye',
        'sizeOfEye',
        'underEyePuffiness',
        'sizeOfNose',
        'highFlatNose',
        'hookNose',
        'wideNarrowNose',
        'toothy',
        'smile',
        'sizeOfMouth',
        'thickThinLips',
        'doubleChin',
        'eyebrowShape',
        'bushyEyebrows',
        'thickThinEyebrows',
        'mustache',
        'beardShape',
        'whiskers',
        'highCheekbone'
    ]

separate_attributes = ['Women',
                       'Asian',  # Asian and Indian
                       'White',
                       'Black',
                       'Youth',  # teenager and youth
                       'Middle',
                       'Old',
                       'Wrinkle',
                       'MakeUp',
                       'Bald',
                       'LargeForehead',
                       'RoundFace',
                       'DiamondFace',
                       'OvalFace',
                       'SquareShapeFace',  # SquareFace and RectangularFace
                       'NarrowEye',
                       'SleepyEye',
                       'SlantEye',
                       'SharpEye',
                       'FlabbyEye',
                       'BigEye',
                       'SmallEye',
                       'UnderEyePuffiness',
                       'BigNose',
                       'SmallNose',
                       'HighNose',
                       'FlatNose',
                       'HookNose',
                       'WideNose',
                       'NarrowNose',
                       'Toothy',
                       'Smile',
                       'BigMouth',
                       'SmallMouth',
                       'ThickLips',
                       'ThinLips',
                       'DoubleChin',
                       'ArchedEyebrows',
                       'FlatEyebrow',
                       'SlantedEyebrows',
                       'UpsideDownSlantedEyebrows',
                       'BushyEyebrows',
                       'ThickEyebrows',
                       'ThinEyebrows',
                       'Mustache',
                       'Goatee',
                       'Whiskers',
                       'OtherBeard&NoBeard',
                       'HighCheekbones',
                       'SquareJaw']

count_attributes = len(separate_attributes)

separate_groups = [1] * count_attributes

def prepare_device():
    n_gpu = torch.cuda.device_count()
    device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
    list_ids = list(range(n_gpu))
    return device, list_ids

