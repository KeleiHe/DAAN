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

from sklearn.metrics import f1_score, accuracy_score
import numpy as np


def cal_attribute_performance(num, label, prediction, visual_label):
    accuracy = np.zeros(num)
    f1 = np.zeros(num)
    prediction = prediction.transpose(0, 1).cpu().numpy()
    label = label.transpose(0, 1).cpu().numpy()
    visual_label = visual_label.transpose(0, 1).cpu().numpy()
    for i in range(num):
        cal_prediction = np.delete(prediction[i], np.where(visual_label[i] == 0))
        cal_label = np.delete(label[i], np.where(visual_label[i] == 0))
        accuracy[i] = accuracy_score(y_true=cal_label, y_pred=cal_prediction)
        f1[i] = f1_score(y_true=cal_label, y_pred=cal_prediction, average='macro')
    return accuracy, f1


def cal_class_performance(num, label, prediction):
    accuracy = np.zeros(num)
    f1 = np.zeros(num)
    prediction = prediction.transpose(0, 1).cpu().numpy()
    label = label.transpose(0, 1).cpu().numpy()
    for i in range(num):
        cal_prediction = np.delete(prediction[i], np.where(label[i] == -1))
        cal_label = np.delete(label[i], np.where(label[i] == -1))
        accuracy[i] = accuracy_score(y_true=cal_label, y_pred=cal_prediction)
        f1[i] = f1_score(y_true=cal_label, y_pred=cal_prediction, average='macro')
    return accuracy, f1
