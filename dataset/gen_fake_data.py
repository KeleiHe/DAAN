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

from dataset.WebCaricature import WebCaricature
from dataloader import *
from model.GAN.UGATIT import UGATIT
from utils.gan import *
from utils import cross_domain_param as param
args = param.args
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
device, device_ids = param.prepare_device()
result_dir = './dataset/gan/'


def gan(source_dataLoader, target_dataLoader, translator):
    translator.genA2B.eval(), translator.genB2A.eval()
    for i, samples_A in enumerate(source_dataLoader):
        real_A = samples_A['image'].to(device)
        real_A_name = samples_A['name']
        fake_A2B, _, fake_A2B_heatmap = translator.genA2B(real_A)
        for i in range(real_A.shape[0]):
            A2B = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[i]))),
                                  RGB2BGR(tensor2numpy(denorm(fake_A2B[i])))), 0)
            B = RGB2BGR(tensor2numpy(denorm(fake_A2B[i])))

            dir1 = os.path.join(result_dir, 'gan', 'test', 'P2C')
            dir2 = os.path.join(result_dir, 'gan', 'Generate', 'P2C')
            if not os.path.exists(dir1):
                os.makedirs(dir1)
            if not os.path.exists(dir2):
                os.makedirs(dir2)
            cv2.imwrite(os.path.join(dir1, str(real_A_name[i])), A2B * 255.0)
            cv2.imwrite(os.path.join(dir2, str(real_A_name[i])), B * 255.0)
            print('SAVE P2C!')

    for i, samples_B in enumerate(target_dataLoader):
        real_B = samples_B['image'].to(device)
        real_B_name = samples_B['name']
        fake_B2A, _, fake_B2A_heatmap = translator.genB2A(real_B)
        for i in range(real_B.shape[0]):
            B2A = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                  RGB2BGR(tensor2numpy(denorm(fake_B2A[0])))), 0)
            A = RGB2BGR(tensor2numpy(denorm(fake_B2A[i])))
            dir1 = os.path.join(result_dir, 'gan', 'test', 'C2P')
            dir2 = os.path.join(result_dir, 'gan', 'Generate', 'C2P')
            if not os.path.exists(dir1):
                os.makedirs(dir1)
            if not os.path.exists(dir2):
                os.makedirs(dir2)
            cv2.imwrite(os.path.join(dir1, str(real_B_name[i])), B2A * 255.0)
            cv2.imwrite(os.path.join(dir2, str(real_B_name[i])), A * 255.0)
            print('SAVE C2P!')


def main():
    modelType = args.modelType
    source_data = WebCaricature('all_data', modelType, parse=args.source_dataset)
    target_data = WebCaricature('all_data', modelType, parse=args.target_dataset)
    source_dataset = MyDataSet(usedata=source_data,
                               transform1=transforms.Compose([
                                   transforms.Resize(256)
                               ]),
                               transform2=transforms.Compose([
                                   ToTensorDict(),
                                   NormalizeImageDict(['image'])
                               ]))
    target_dataset = MyDataSet(usedata=target_data,
                               transform1=transforms.Compose([
                                   transforms.Resize(256)
                               ]),
                               transform2=transforms.Compose([
                                   ToTensorDict(),
                                   NormalizeImageDict(['image'])
                               ]))
    source_dataLoader = data.DataLoader(source_dataset, batch_size=1,
                                        shuffle=False,
                                        num_workers=args.workers
                                        )
    target_dataLoader = data.DataLoader(target_dataset, batch_size=1,
                                        shuffle=False,
                                        num_workers=args.workers)

    '''For Generating the images of the other domain'''
    translator = UGATIT()
    translator.build_model()
    gan(source_dataLoader, target_dataLoader, translator)


if __name__ == '__main__':
        main()
