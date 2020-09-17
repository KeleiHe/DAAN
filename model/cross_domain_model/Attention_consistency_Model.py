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
from torch.nn import functional as F
from torch import nn
from torch.autograd import Variable
import torchvision.models as models
from utils import cross_domain_param as param
args = param.args
device, device_ids = param.prepare_device()


class ExclusionClassifier(torch.nn.Module):
    def __init__(self, num_classes):
        super(ExclusionClassifier, self).__init__()
        if args.celoss and num_classes == 1:
            if num_classes > 1:
                self.num_classes = num_classes
            else:
                self.num_classes = num_classes + 1
        else:
            self.num_classes = num_classes
        self.fc = nn.Linear(2048, self.num_classes)

    def forward(self, x, gap_x):
        feat = x
        N, C, H, W = feat.shape
        gap_x = gap_x.view(gap_x.size(0), -1)
        out = self.fc(gap_x)
        params = list(self.fc.parameters())
        fc_weights = params[-2].data
        fc_weights = fc_weights.view(1, self.num_classes, C, 1, 1)
        fc_weights = Variable(fc_weights, requires_grad=False)

        # attention
        feat = feat.unsqueeze(1)  # N * 1 * C * H * W
        hm = feat * fc_weights
        hm = hm.sum(2)  # N * self.num_classes * H * W
        return out, hm


class FeatureExtraction(torch.nn.Module):
    def __init__(self):
        super(FeatureExtraction, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.extractor = nn.Sequential(*list(self.resnet.children())[:-2])
        self.gap = list(self.resnet.children())[-2]

    def forward(self, img):
        feature = self.extractor(img)
        gap_feature = self.gap(feature)
        return feature, gap_feature


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.attri_count = param.exclution_groups
        for i, n in enumerate(self.attri_count):
            exec('self.{} = ExclusionClassifier({})'.format(param.exclution_attris[i], n))

    def forward(self, x, gap_x, vis=False):
        result = x.new(x.shape[0], 0).to(device)
        heatmap = x.new(x.shape[0], 0, x.shape[-2], x.shape[-1]).to(device)
        for i, n in enumerate(self.attri_count):
            res, hm = eval('self.{}(x, gap_x)'.format(param.exclution_attris[i]))
            res = res.view(x.size(0), -1)  # flatten
            if torch.cuda.is_available():
                res = res.to(device)
                hm = hm.to(device)
            if vis is False:
                result = torch.cat((result, res), dim=1)
                heatmap = torch.cat((heatmap, hm), dim=1)
            else:
                res = F.softmax(res)
                if n == 1:
                    result = torch.cat((result, res[:, 1:]), dim=1)
                    heatmap = torch.cat((heatmap, hm[:, 1:, :, :]), dim=1)
                else:
                    result = torch.cat((result, res), dim=1)
                    heatmap = torch.cat((heatmap, hm), dim=1)
        return result, heatmap


class AttentionConsisNet(nn.Module):
    def __init__(self):
        super(AttentionConsisNet, self).__init__()
        self.sharedNet = FeatureExtraction()
        self.classifier = Classifier()
        # self.classifier_t = Classifier()

    def forward(self, img_s, gen_img_s, img_t, gen_img_t):
        feature_s, gap_feature_s = self.sharedNet(img_s)
        gen_feature_s, gap_gen_feature_s = self.sharedNet(gen_img_s)
        feature_t, gap_feature_t = self.sharedNet(img_t)
        gen_feature_t, gap_gen_feature_t = self.sharedNet(gen_img_t)

        output_s, heatmap_s = self.classifier(feature_s, gap_feature_s)
        gen_output_s, gen_heatmap_s = self.classifier(gen_feature_s, gap_gen_feature_s)
        output_t, heatmap_t = self.classifier(feature_t, gap_feature_t)
        gen_output_t, gen_heatmap_t = self.classifier(gen_feature_t, gap_gen_feature_t)

        return (output_s, gap_feature_s, heatmap_s), \
               (gen_output_s, gap_gen_feature_s, gen_heatmap_s), \
               (output_t, gap_feature_t, heatmap_t), \
               (gen_output_t, gap_gen_feature_t, gen_heatmap_t)

