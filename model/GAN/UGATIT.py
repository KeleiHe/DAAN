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

import sys
import os
from model.GAN.UGATIT_Networks import *
from utils.gan import *
from glob import glob
from utils import cross_domain_param as param
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
args = param.args
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
device, device_ids = param.prepare_device()


class UGATIT(object):
    def __init__(self):
        self.light = True
        self.load_dir = './attention_consistency_result/result_cross_model/gan/'
        self.result_dir = param.result_dir
        self.n_res = 4
        self.ch = 56
        self.img_size = 224

    def build_model(self):
        """ Define Generator, Discriminator """
        self.genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light).to(device)
        self.genB2A = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light).to(device)

        """ Define Rho clipper to constraint the value of rho in AdaILN and ILN"""
        self.Rho_clipper = RhoClipper(0, 1)

        model_list = glob(os.path.join(self.load_dir, 'model', '*.pt'))
        if not len(model_list) == 0:
            model_list.sort()
            iter = int(model_list[-1].split('_')[-1].split('.')[0])
            self.load(os.path.join(self.load_dir, 'model'), iter)
            print(" [*] Load SUCCESS")
        else:
            print(" [*] Load FAILURE")
            return

    def load(self, dir, step):
        params = torch.load(os.path.join(dir, 'params_%07d.pt' % step))
        self.genA2B.load_state_dict(params['genA2B'])
        self.genB2A.load_state_dict(params['genB2A'])
        if torch.cuda.device_count() > 1:
            print("Let's use " + str(torch.cuda.device_count()) + " GPUs to  generate the images!")
            self.genA2B = nn.DataParallel(self.genA2B, device_ids=device_ids)
            self.genB2A = nn.DataParallel(self.genB2A, device_ids=device_ids)

    def generate_imgs(self, fold, epoch, A_data, B_data):
        self.genA2B.eval(), self.genB2A.eval()
        real_A = A_data['image'].to(device)
        real_A_name = A_data['name']
        fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
        if epoch % 10 == 0:
            for i in range(real_A.shape[0]):
                A2B = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[i]))),
                                      RGB2BGR(tensor2numpy(denorm(fake_A2B[i])))), 0)
                B = RGB2BGR(tensor2numpy(denorm(fake_A2B[i])))

                dir1 = os.path.join(self.result_dir,  'gan', 'test', 'A2B', str(fold), str(epoch))
                dir2 = os.path.join(self.result_dir,  'gan', 'Generate', 'A2B', str(fold), str(epoch))
                if not os.path.exists(dir1):
                    os.makedirs(dir1)
                if not os.path.exists(dir2):
                    os.makedirs(dir2)
                cv2.imwrite(os.path.join(dir1, str(real_A_name[i])), A2B * 255.0)
                cv2.imwrite(os.path.join(dir2, str(real_A_name[i])), B * 255.0)

        real_B = B_data['image'].to(device)
        real_B_name = B_data['name']
        fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)
        if epoch % 10 == 0:
            for i in range(real_B.shape[0]):
                B2A = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                      RGB2BGR(tensor2numpy(denorm(fake_B2A[0])))), 0)
                A = RGB2BGR(tensor2numpy(denorm(fake_B2A[i])))
                dir1 = os.path.join(self.result_dir, 'gan', 'test', 'B2A', str(fold), str(epoch))
                dir2 = os.path.join(self.result_dir, 'gan', 'Generate', 'B2A', str(fold), str(epoch))
                if not os.path.exists(dir1):
                    os.makedirs(dir1)
                if not os.path.exists(dir2):
                    os.makedirs(dir2)
                cv2.imwrite(os.path.join(dir1, str(real_B_name[i])), B2A * 255.0)
                cv2.imwrite(os.path.join(dir2, str(real_B_name[i])), A * 255.0)

        return fake_A2B, fake_B2A
