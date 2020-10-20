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

from dataset.WebCariA import WebCariA
from dataloader import *
from sklearn.model_selection import KFold
from utils import cross_domain_param as param


args = param.args
global save_dir
save_dir = param.data_save_dir


def add(x):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    path = os.path.join(save_dir, "setting.txt")
    with open(path, "a+") as outfile:
        outfile.write(x + "\n")
    print(x)


def k_fold_dataset_split(k, dataset):
    kf = KFold(n_splits=k, shuffle=True)
    kf.get_n_splits(dataset)
    return kf


if __name__ == '__main__':
    modelType = args.modelType
    source_data = WebCaricature('all_data', modelType, parse=args.source_dataset)
    target_data = WebCaricature('all_data', modelType, parse=args.target_dataset)
    source_dataset = MyDataSet(usedata=source_data,
                               transform1=transforms.Compose([
                                   transforms.RandomHorizontalFlip(),
                                   transforms.Resize(256),
                                   transforms.CenterCrop(224)
                               ]),
                               transform2=transforms.Compose([
                                   ToTensorDict(),
                                   NormalizeImageDict(['image'])
                               ]))
    target_dataset = MyDataSet(usedata=target_data,
                               transform1=transforms.Compose([
                                   transforms.RandomHorizontalFlip(),
                                   transforms.Resize(256),
                                   transforms.CenterCrop(224)
                               ]),
                               transform2=transforms.Compose([
                                   ToTensorDict(),
                                   NormalizeImageDict(['image'])
                               ]))
    source_num_attribute = source_dataset.num_attribute
    target_num_attribute = target_dataset.num_attribute
    add("count of classes:{}".format(source_num_attribute))
    len_source_dataset = len(source_dataset)
    len_target_dataset = len(target_dataset)
    add('size of source set : {} '.format(len_source_dataset))
    add('size of target set : {} '.format(len_target_dataset))
    assert (source_num_attribute == target_num_attribute)
    k = args.k
    kf = k_fold_dataset_split(k, target_dataset)
    for i, (target_train_index, target_val_index) in enumerate(kf.split(target_dataset)):
        add('fold {} :'.format(i+1))
        np.save(os.path.join(save_dir, "target_train_indices_" + str(i+1) + ".npy"), np.array(target_train_index))
        np.save(os.path.join(save_dir, "target_val_indices_" + str(i+1) + ".npy"), np.array(target_val_index))
        len_target_train_index = len(target_train_index)
        len_target_val_index = len(target_val_index)
        add('size of train in target set : {} '.format(len_target_train_index))
        add('size of val in target set : {} '.format(len_target_val_index))
        add('target train index')
        add(str(target_train_index))
        add('target val index')
        add(str(target_val_index))
