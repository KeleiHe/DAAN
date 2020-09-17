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

import os
from utils import webCari_param


class WebCaricature:
    def __init__(self, dataType, modelType, parse, des_attri=None):
        self.dir_path = "./dataset/" + str(parse)
        self.dataType = dataType
        self.parse = parse
        self.des_attri = des_attri
        if self.dataType == 'train':
            if self.parse == 'Caricature':
                self.subPath = 'CariTrain'
            elif self.parse == 'Photo':
                self.subPath = 'PhotoTrain'
            else:
                self.subPath = 'WebCariTrain'

        elif self.dataType == 'val':
            if self.parse == 'Caricature':
                self.subPath = 'CariVal'
            elif self.parse == 'Photo':
                self.subPath = 'PhotoVal'
            else:
                self.subPath = 'WebCariVal'

        elif self.dataType == 'test':
            if self.parse == 'Caricature':
                self.subPath = 'CariTest'
            elif self.parse == 'Photo':
                self.subPath = 'PhotoTest'
            else:
                self.subPath = 'WebCariTest'

        elif self.dataType == 'all_data':
            if self.parse == 'Caricature':
                self.subPath = 'all_cari_data'
            elif self.parse == 'Photo':
                self.subPath = 'all_photo_data'
            else:
                self.subPath = 'all_WebCari_data'

        else:
            print("Caricature error, please select a dataType from: train, val, github")
            exit(1)
        self.modelType = modelType
        self.dir_path = os.path.join(self.dir_path, self.subPath)
        self.attributes = ['Women',
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

        self.names, self.annas, self.visuals, self.num_attribute = self.getImgNameAndAnnas()
        print(parse+"dataset, images: ", len(self.names), " type for: ", self.dataType, "   num_attribute: ",
              self.num_attribute)

    def getImgNameAndAnnas(self):
        names = []
        annas = []
        visuals = []
        file = self.subPath+".txt"
        file_v = self.subPath+"_V.txt"
        fileList = open(os.path.join(self.dir_path, file)).readlines()
        fileVList = open((os.path.join(self.dir_path, file_v))).readlines()
        if webCari_param.isDug:
            fileList = fileList[:10]
            fileVList = fileVList[:10]

        newFileList = []
        newfileVList = []
        for file in fileList:
            file = file.strip().split(' ')
            for i in range(1, len(file)):
                file[i] = int(file[i])
            newLine = []
            if file[52] == 0 and file[53] == 0 and file[54] == 0:
                file.insert(55, 1)
            else:
                file.insert(55, 0)
            file[2] = file[5] + file[2]
            file[9] = file[8] + file[9]
            file[20] = file[19] + file[20]
            file[55] = file[53] + file[55]
            for i in range(len(file)):
                if i not in [5, 6, 7, 8, 19, 48, 53]:
                    newLine.append(file[i])
            newFileList.append(newLine)

        for file in fileVList:
            file = file.strip().split(' ')
            for i in range(1, len(file)):
                file[i] = int(file[i])
            newVLine = []
            if file[52] == 0 and file[53] == 0 and file[54] == 0:
                file.insert(55, 0)
            else:
                file.insert(55, 1)
            for i in range(len(file)):
                if i not in [5, 6, 7, 8, 19, 48, 53]:
                    newVLine.append(file[i])
            newfileVList.append(newVLine)

        if self.modelType == 'seperate':
            num_attribute = 1
            attribute = self.des_attri
            print("des_attribute", attribute)
            if attribute not in self.attributes:
                print("error: ", attribute, "is not in this dataset, please write a correct attribute in param")
                exit(1)
            for line in newFileList:
                names.append(line[0])
                attributes = line[1::]
                index = self.attributes.index(attribute)
                annas.append([int(attributes[index])])
            for line in newfileVList:
                attributes_v = line[1::]
                index = self.attributes.index(attribute)
                visuals.append([int(attributes_v[index])])
        else:
            for line in newFileList:
                names.append(line[0])
                annas.append([int(x) for x in line[1::]])
            for line in newfileVList:
                visuals.append([int(x) for x in line[1::]])
            self.attributes = self.attributes
            num_attribute = len(self.attributes)
        return names, annas, visuals, num_attribute

    def getPath(self, name):
        name = name.replace(' ', '_')
        name = name.replace('._', '_')
        name = name.replace('-', '_')
        name = name + ".jpg"
        return name



