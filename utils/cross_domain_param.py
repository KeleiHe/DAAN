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
parser.add_argument('--modelType', dest='cross_model', default='cross_model', type=str)

parser.add_argument('-k', default=5, type=int, metavar='N',
                    help='k folds cross validation')
parser.add_argument('-b', '--batch-size', default=40, type=int, metavar='N',
                    help='mini-batch size (default: 50)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='numer of total epochs to run')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful to restarts)')
parser.add_argument('--lr', '--learning-rate', default=[5e-2], type=list, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--lr_D', '--learning-rate-D', default=1e-4, type=float,
                    help='initial learning rate of discriminator')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='Weight decay (default: 1e-4)')

parser.add_argument('--task_weight', default=1, type=float, metavar='M',
                    help='task_weight')
parser.add_argument('--label_consis_weight', default=0.02, type=float, metavar='M',
                    help='label_consis_weight')
parser.add_argument('--feat_consis_weight', default=0.1, type=float, metavar='M',
                    help='feat_consis_weight')
parser.add_argument('--attention_consis_weight', default=0.1, type=float, metavar='M',
                    help='attention_consis_weight')

parser.add_argument('--device', default='0,1,2,3', type=str, metavar='G',
                    help='gpu device')

parser.add_argument('-ce', '--celoss', dest='celoss', action='store_false',
                    help='whether to use Cross Entropy Loss')
parser.add_argument('-hm', '--hmloss', type=str, default='Vanilla',
                    help="choose the GAN objective.")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
isDug = False
args.testname = '5cv_Feat_ConsisModel-shareWeight_lr_5e-2_1_1e-2'
ROOT_PATH = './attention_consistency_result'
data_save_dir = ROOT_PATH + '/new_result_' + str(args.modelType) + "/" + str(args.source_dataset) + str(2) \
                + str(args.target_dataset) + "/cv_dataset/"
result_dir = ROOT_PATH + '/new_result_' + str(args.modelType) + "/" + str(args.source_dataset) + str(2) \
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


def prepare_device():
    n_gpu = torch.cuda.device_count()
    device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
    list_ids = list(range(n_gpu))
    return device, list_ids

