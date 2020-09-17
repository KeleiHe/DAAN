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

import torch.nn as nn
from model.cross_domain_model.Attention_consistency_Model import AttentionConsisNet
from dataloader import *
import os
from utils import cam_param as param
from dataset.WebCaricature import WebCaricature
from torch.utils.data import SubsetRandomSampler
from visual.visual_utils import cal_nomalize_cam

args = param.args
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
device, device_ids = param.prepare_device()
result_dir = param.result_dir
load_model_dir =  param.result_dir
data_save_dir = param.data_save_dir
k = 0
exclution_groups = param.exclution_groups
exclution_attris = param.exclution_attris
attributes = param.separate_attributes


def ori_show_cam_on_image(heatmap, fake_heatmap, imageNames, preds, fake_preds, labels):
    labels = labels.detach().cpu().numpy()
    preds = preds.detach().cpu().numpy()
    fake_preds = fake_preds.detach().cpu().numpy()
    for i in range(num_attribute):
        sub_attri = param.separate_attributes[i]
        sub_label = labels[0, i]
        sub_pred = preds[0, i]
        sub_fake_pred = fake_preds[0, i]
        attri_heatmap = heatmap[0, i]
        fake_attri_heatmap = fake_heatmap[0, i]

        save_dir = os.path.join(result_dir, 'heatmap', model_name[:-4], parse, sub_attri)
        attri_mask = attri_heatmap.detach().cpu().numpy()
        fake_attri_mask = fake_attri_heatmap.detach().cpu().numpy()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        cam = cal_nomalize_cam(attri_mask, image_path)
        fake_cam = cal_nomalize_cam(fake_attri_mask, fake_image_path)
        cam = np.uint8(255 * cam)
        fake_cam = np.uint8(255 * fake_cam)
        padding = np.zeros((40, cam.shape[1], cam.shape[2]))
        save_cam = np.concatenate((cam, padding), 0)
        save_cam = np.concatenate((save_cam, fake_cam), 0)
        pred = round(sub_pred, 2)
        fake_pred = round(sub_fake_pred, 2)
        cv2.imwrite(save_dir + '/' + imageNames[:-4] + '_' + str(pred) + '_' + str(fake_pred) + '_' + str(sub_label) + '.jpg',
                    save_cam)


def get_attentionnet_cam(model, input):
    model = model.module
    extractor = model.sharedNet.to(device)
    classifier = model.classifier.to(device)

    feature_map, feature_gap = extractor(input)
    N, C, H, W = feature_map.shape
    result, heatmap = classifier(feature_map, feature_gap, vis=True)
    return result, heatmap


if __name__ == '__main__':
    modelType = args.modelType
    source_data = WebCaricature('all_data', modelType, parse=args.source_dataset)
    target_data = WebCaricature('all_data', modelType, parse=args.target_dataset)
    source_dataset = MyDataSet(usedata=source_data,
                               transform1=transforms.Compose([
                                   transforms.Resize(256),
                                   transforms.CenterCrop(224)
                               ]),
                               transform2=transforms.Compose([
                                   ToTensorDict(),
                                   NormalizeImageDict(['image','fake_image'])
                               ]))
    target_dataset = MyDataSet(usedata=target_data,
                               transform1=transforms.Compose([
                                   transforms.Resize(256),
                                   transforms.CenterCrop(224)
                               ]),
                               transform2=transforms.Compose([
                                   ToTensorDict(),
                                   NormalizeImageDict(['image','fake_image'])
                               ]))
    attributes = source_data.attributes
    num_attribute = source_dataset.num_attribute
    assert (source_dataset.num_attribute == target_dataset.num_attribute)
    print("count of classes:{}".format(num_attribute))
    k = args.k
    i = 0
    model_name = 'best_DataParallelFold ' + str(i) + '_LR0.05_F1.pth'
    target_train_index = np.load(os.path.join(data_save_dir, "target_train_indices_" + str(i + 1) + ".npy"))
    target_val_index = np.load(os.path.join(data_save_dir, "target_val_indices_" + str(i + 1) + ".npy"))
    target_train_subset = torch.utils.data.Subset(target_dataset, target_train_index)
    target_val_subset = torch.utils.data.Subset(target_dataset, target_val_index)
    source_dataLoader = data.DataLoader(source_dataset, batch_size=1,
                                        shuffle=False)
    target_train_dataLoader = torch.utils.data.DataLoader(target_train_subset, batch_size=1,
                                                          shuffle=False)
    target_val_dataLoader = torch.utils.data.DataLoader(target_val_subset, batch_size=1,
                                                        shuffle=False)
    model = AttentionConsisNet()
    model.to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use ", torch.cuda.device_count(), " GPUs!")
        model = nn.DataParallel(model, device_ids=device_ids)
    checkpoint_name = os.path.join(load_model_dir, model_name)
    checkpoint = torch.load(checkpoint_name, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print(model)

    for (dataLoader, parse) in [(source_dataLoader, 'source_Set'),(target_train_dataLoader, 'target_train_Set'),
                                (target_val_dataLoader, 'target_val_Set')]:
        print(parse)
        for i_batch, sample_batched in enumerate(dataLoader):
            inputs = sample_batched['image'].to(device)
            fake_inputs = sample_batched['fake_image'].to(device)
            labels = sample_batched['attribute'].to(device)
            imageNames = sample_batched['name'][0]
            print(imageNames)
            if parse == 'source_Set':
                image_path = source_dataset.data_path + '/' + str(imageNames)
                fake_image_path = source_dataset.fake_data_path + '/' + str(imageNames)
            else:
                image_path = target_dataset.data_path + '/' + str(imageNames)
                fake_image_path = target_dataset.fake_data_path + '/' + str(imageNames)
            preds, heatmaps = get_attentionnet_cam(model, inputs)
            fake_preds, fake_heatmaps = get_attentionnet_cam(model, fake_inputs)
            ori_show_cam_on_image(heatmaps, fake_heatmaps, imageNames, preds, fake_preds, labels)