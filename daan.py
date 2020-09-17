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

from os.path import exists, join, basename, dirname
from os import makedirs
import shutil
import torch.nn as nn
import torch.optim as optim
from dataset.WebCaricature import WebCaricature
from dataloader import *
from model.cross_domain_model.Attention_consistency_Model import AttentionConsisNet
from model.cross_domain_model.Discriminator import AttentionDiscriminator2D, AttentionDiscriminator
from model.cross_domain_model.Discriminator import FeatureDiscriminator
import warnings
from torch.autograd import Variable
from utils import cross_domain_param as param, performance
from torch.utils.tensorboard import SummaryWriter

AttD = AttentionDiscriminator

warnings.filterwarnings('ignore')
args = param.args
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
device, device_ids = param.prepare_device()
global result_dir, data_save_dir
data_save_dir = param.data_save_dir
result_dir = param.result_dir


def add(x):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    path = os.path.join(result_dir, "TrainLog.txt")
    with open(path, "a+") as outfile:
        outfile.write(x + "\n")
    print(x)


def save_checkpoint(state, is_best, file):
    model_dir = dirname(file)
    model_fn = basename(file)
    # make dir if needed (should be non-empty)
    if model_dir != '' and not exists(model_dir):
        makedirs(model_dir)
    torch.save(state, file)
    if is_best:
        shutil.copyfile(file, join(model_dir, 'best_' + model_fn))
        add("save best model to {}.".format(file))


def train(epoch, iteration, model, AttentionD_S, AttentionD_T,  FeatureD_S, FeatureD_T, loss_fn, lab_consis_loss,
          feature_loss, heatmap_loss, optimizer, optimizer_AttentionD_S, optimizer_AttentionD_T, optimizer_FeatureD_S,
          optimizer_FeatureD_T, source_train_dataLoader, target_train_dataLoader):

    model.train()
    AttentionD_S.train()
    AttentionD_T.train()
    FeatureD_S.train()
    FeatureD_T.train()

    src_iter = iter(source_train_dataLoader)
    tgt_iter = iter(target_train_dataLoader)

    add('iteration: {}'.format(iteration))

    source_label_loss = 0
    target_label_loss = 0
    label_consistency_loss = 0
    source_feature_adv_loss = 0
    target_feature_adv_loss = 0
    source_attention_adv_loss = 0
    target_attention_adv_loss = 0
    AttentionD_S_loss = 0
    AttentionD_T_loss = 0
    FeatureD_S_loss = 0
    FeatureD_T_loss = 0

    for it in range(iteration):
        try:
            src_data = src_iter.next()
        except Exception as err:
            src_iter = iter(source_train_dataLoader)
            src_data = src_iter.next()

        try:
            tgt_data = tgt_iter.next()
        except Exception as err:
            tgt_iter = iter(target_train_dataLoader)
            tgt_data = tgt_iter.next()

        src_img = src_data['image'].to(device)
        src_labels = src_data['attribute'].to(device)
        src_class_labels = src_data['class_label'].to(device)
        tgt_img = tgt_data['image'].to(device)
        fake_S2T_img = src_data['fake_image'].to(device)
        fake_T2S_img = tgt_data['fake_image'].to(device)

        optimizer.zero_grad()
        optimizer_AttentionD_S.zero_grad()
        optimizer_AttentionD_T.zero_grad()
        optimizer_FeatureD_S.zero_grad()
        optimizer_FeatureD_T.zero_grad()

        # fix Discriminator, train the backbone model
        for params in AttentionD_S.parameters():
            params.requires_grad = False
        for params in AttentionD_T.parameters():
            params.requires_grad = False
        for params in FeatureD_S.parameters():
            params.requires_grad = False
        for params in FeatureD_T.parameters():
            params.requires_grad = False

        (src_pred, src_feat, src_hm), (gen_src_pred, gen_src_feat, gen_src_hm),\
        (tgt_pred, tgt_feat, tgt_hm), (gen_tgt_pred, gen_tgt_feat, gen_tgt_hm) = \
            model(src_img, fake_T2S_img, tgt_img, fake_S2T_img)

        if args.celoss:
            index = 0
            src_class_labels = src_class_labels.long()
            src_losses = torch.zeros(len(param.exclution_groups)).to(device)
            gen_tgt_losses = torch.zeros(len(param.exclution_groups)).to(device)
            label_consis_losses = torch.zeros(len(param.exclution_groups)).to(device)
            for i, x in enumerate(param.exclution_groups):
                if x > 1:
                    sub_src_pred = src_pred[:, index:index + x]
                    sub_tgt_pred = tgt_pred[:, index:index + x]
                    sub_gen_src_pred = gen_src_pred[:, index:index + x]
                    sub_gen_tgt_pred = gen_tgt_pred[:, index:index + x]
                    src_losses[i] = loss_fn(sub_src_pred, src_class_labels[:, i])
                    gen_tgt_losses[i] = loss_fn(sub_gen_tgt_pred, src_class_labels[:, i])
                else:
                    sub_src_pred = src_pred[:, index:index + 2]
                    sub_tgt_pred = tgt_pred[:, index:index + 2]
                    sub_gen_src_pred = gen_src_pred[:, index:index + 2]
                    sub_gen_tgt_pred = gen_tgt_pred[:, index:index + 2]
                    src_losses[i] = loss_fn(sub_src_pred, src_class_labels[:, i])
                    gen_tgt_losses[i] = loss_fn(sub_gen_tgt_pred, src_class_labels[:, i])

                    index = index + 1

                label_consis_losses[i] = lab_consis_loss(sub_gen_src_pred, sub_tgt_pred)
                index = index + x
            src_att_loss = torch.mean(src_losses)
            gen_tgt_loss = torch.mean(gen_tgt_losses)
            label_consis_loss = torch.mean(label_consis_losses)
        else:
            src_labels_F = src_labels.float()
            src_att_loss = loss_fn(src_pred, src_labels_F)
            gen_tgt_loss = loss_fn(gen_tgt_pred, src_labels_F)
            label_consis_loss = lab_consis_loss(gen_src_pred, tgt_pred)
        src_att_loss = args.task_weight * src_att_loss
        gen_tgt_loss = args.task_weight * gen_tgt_loss
        label_consis_loss = args.label_consis_weight * label_consis_loss
        loss = src_att_loss + gen_tgt_loss + label_consis_loss
        loss.backward(retain_graph=True)
        source_label_loss += src_att_loss.data.cpu().numpy()
        target_label_loss += gen_tgt_loss.data.cpu().numpy()
        label_consistency_loss += label_consis_loss.data.cpu().numpy()

        FeatureD_S_out = FeatureD_S(gen_src_feat)
        FeatureD_T_out = FeatureD_T(tgt_feat)
        loss_adv_feature_src = feature_loss(FeatureD_S_out,
                                           Variable(torch.FloatTensor(FeatureD_S_out.data.size()).fill_(
                                                   source_label)).to(device))

        loss_adv_feature_tgt = feature_loss(FeatureD_T_out,
                                           Variable(torch.FloatTensor(FeatureD_T_out.data.size()).fill_(
                                                   source_label)).to(device))
        loss_adv_feature_src = args.feat_consis_weight * loss_adv_feature_src
        loss_adv_feature_tgt = args.feat_consis_weight * loss_adv_feature_tgt
        loss = loss_adv_feature_src + loss_adv_feature_tgt
        loss.backward(retain_graph=True)
        source_feature_adv_loss += loss_adv_feature_src.data.cpu().numpy()
        target_feature_adv_loss += loss_adv_feature_tgt.data.cpu().numpy()

        AttentionD_S_out = AttentionD_S(gen_src_hm)
        AttentionD_T_out = AttentionD_T(tgt_hm)
        loss_adv_attention_src = heatmap_loss(AttentionD_S_out,
                                              Variable(torch.FloatTensor(AttentionD_S_out.data.size()).fill_(
                                                  source_label)).to(device))

        loss_adv_attention_tgt = heatmap_loss(AttentionD_T_out,
                                              Variable(torch.FloatTensor(AttentionD_T_out.data.size()).fill_(
                                                  source_label)).to(device))
        loss_adv_attention_src = args.attention_consis_weight * loss_adv_attention_src
        loss_adv_attention_tgt = args.attention_consis_weight * loss_adv_attention_tgt
        loss = loss_adv_attention_src + loss_adv_attention_tgt
        loss.backward()
        source_attention_adv_loss += loss_adv_attention_src.data.cpu().numpy()
        target_attention_adv_loss += loss_adv_attention_tgt.data.cpu().numpy()

        # train the Discriminator
        for params in AttentionD_S.parameters():
            params.requires_grad = True
        for params in AttentionD_T.parameters():
            params.requires_grad = True
        for params in FeatureD_S.parameters():
            params.requires_grad = True
        for params in FeatureD_T.parameters():
            params.requires_grad = True

        src_feat = src_feat.detach()
        gen_tgt_feat = gen_tgt_feat.detach()
        FeatureD_S_out = FeatureD_S(src_feat)
        FeatureD_T_out = FeatureD_T(gen_tgt_feat)
        loss_FeatureD_S = feature_loss(FeatureD_S_out,
                                      Variable(torch.FloatTensor(FeatureD_S_out.data.size()).fill_(
                                              source_label)).to(device))
        loss_FeatureD_T = feature_loss(FeatureD_T_out,
                                      Variable(torch.FloatTensor(FeatureD_T_out.data.size()).fill_(
                                              source_label)).to(device))
        loss_FeatureD_S.backward()
        loss_FeatureD_T.backward()

        src_hm = src_hm.detach()
        gen_tgt_hm = gen_tgt_hm.detach()
        AttentionD_S_out = AttentionD_S(src_hm)
        AttentionD_T_out = AttentionD_T(gen_tgt_hm)
        loss_AttentionD_S = heatmap_loss(AttentionD_S_out,
                                         Variable(torch.FloatTensor(AttentionD_S_out.data.size()).fill_(
                                             source_label)).to(device))
        loss_AttentionD_T = heatmap_loss(AttentionD_T_out,
                                         Variable(torch.FloatTensor(AttentionD_T_out.data.size()).fill_(
                                             source_label)).to(device))
        loss_AttentionD_S.backward()
        loss_AttentionD_T.backward()

        gen_src_feat = gen_src_feat.detach()
        tgt_feat = tgt_feat.detach()
        FeatureD_S_gen_out = FeatureD_S(gen_src_feat)
        FeatureD_T_gen_out = FeatureD_T(tgt_feat)
        loss_FeatureD_S_gen = feature_loss(FeatureD_S_gen_out,
                                          Variable(torch.FloatTensor(FeatureD_S_gen_out.data.size()).fill_(
                                              target_label)).to(device))
        loss_FeatureD_T_gen = feature_loss(FeatureD_T_gen_out,
                                          Variable(torch.FloatTensor(FeatureD_T_gen_out.data.size()).fill_(
                                              target_label)).to(device))
        loss_FeatureD_S_gen.backward()
        loss_FeatureD_T_gen.backward()

        gen_src_hm = gen_src_hm.detach()
        tgt_hm = tgt_hm.detach()
        AttentionD_S_gen_out = AttentionD_S(gen_src_hm)
        AttentionD_T_gen_out = AttentionD_T(tgt_hm)
        loss_AttentionD_S_gen = heatmap_loss(AttentionD_S_gen_out,
                                             Variable(torch.FloatTensor(AttentionD_S_gen_out.data.size()).fill_(
                                                 target_label)).to(device))
        loss_AttentionD_T_gen = heatmap_loss(AttentionD_T_gen_out,
                                             Variable(torch.FloatTensor(AttentionD_T_gen_out.data.size()).fill_(
                                                 target_label)).to(device))
        loss_AttentionD_S_gen.backward()
        loss_AttentionD_T_gen.backward()

        FeatureD_S_loss += (loss_FeatureD_S + loss_FeatureD_S_gen).data.cpu().numpy()
        FeatureD_T_loss += (loss_FeatureD_T + loss_FeatureD_T_gen).data.cpu().numpy()
        AttentionD_S_loss += (loss_AttentionD_S + loss_AttentionD_S_gen).data.cpu().numpy()
        AttentionD_T_loss += (loss_AttentionD_T + loss_AttentionD_T_gen).data.cpu().numpy()

        optimizer.step()
        optimizer_FeatureD_S.step()
        optimizer_FeatureD_T.step()
        optimizer_AttentionD_S.step()
        optimizer_AttentionD_T.step()
        total_loss = src_att_loss + gen_tgt_loss + label_consis_loss + loss_adv_feature_src + loss_adv_feature_tgt \
                     + loss_adv_attention_src + loss_adv_attention_tgt + FeatureD_S_loss + FeatureD_T_loss \
                     + AttentionD_S_loss + AttentionD_T_loss
        if it % 20 == 0:
            print('-' * 100)
            add('\n\n Train iter: {} [({:.0f}%)]\t Total Loss:{}\t Src Label Loss: {:.6f}\t Tgt Label Loss: {:.6f}\t '
                'Label_Consis_Loss:{:.6f}\n\n Src_Adv_Feature_Loss:{:.6f}\t Tgt_Adv_Feature_Loss:{:.6f}\t '
                'Src_Adv_Attention_Loss:{:.6f}\t Tgt_Adv_Attention_Loss:{:.6f}\n\n Src_Feat_D_Loss:{:.6f}\t '
                'Tgt_Feat_D_Loss:{:.6f}\t Src_Attention_D_Loss:{:.6f}\t Tgt_Attention_D_Loss:{:.6f}\t '
                .format(it, 100. * it / iteration, total_loss.item(), src_att_loss.item(), gen_tgt_loss.item(),
                        label_consis_loss.item(), loss_adv_feature_src.item(), loss_adv_feature_tgt.item(),
                        loss_adv_attention_src.item(), loss_adv_attention_tgt.item(), loss_FeatureD_S.item(),
                        loss_FeatureD_T.item(), loss_AttentionD_S_gen.item(), loss_AttentionD_T_gen.item()))
    source_label_loss /= iteration
    target_label_loss /= iteration
    label_consistency_loss /= iteration
    source_feature_adv_loss /= iteration
    target_feature_adv_loss /= iteration
    source_attention_adv_loss /= iteration
    target_attention_adv_loss /= iteration
    AttentionD_S_loss /= iteration
    AttentionD_T_loss /= iteration
    FeatureD_S_loss /= iteration
    FeatureD_T_loss /= iteration
    all_loss = source_label_loss + target_label_loss + label_consistency_loss + source_feature_adv_loss \
               + target_feature_adv_loss + source_attention_adv_loss + target_attention_adv_loss + AttentionD_S_loss \
               + AttentionD_T_loss + FeatureD_S_loss + FeatureD_T_loss

    add('Train: Average Source Label Loss: {:.6f}'.format(source_label_loss))
    writer.add_scalar('Average Source Label Loss/train', source_label_loss, epoch)
    add('Train: Average Target Label Loss: {:.6f}'.format(target_label_loss))
    writer.add_scalar('Average Target Label Loss/train', target_label_loss, epoch)
    add('Train: Average Label Consistenct Loss: {:.6f}'.format(label_consistency_loss))
    writer.add_scalar('Average Label Consistenct Loss/train', label_consistency_loss, epoch)

    add('Train: Average Source Adv feature Loss: {:.6f}'.format(source_feature_adv_loss))
    writer.add_scalar('Average Source Adv feature Loss/train', source_feature_adv_loss, epoch)
    add('Train: Average Target Adv feature Loss: {:.6f}'.format(target_feature_adv_loss))
    writer.add_scalar('Average Target Adv feature Loss/train', target_feature_adv_loss, epoch)

    add('Train: Average Source Adv attention Loss: {:.6f}'.format(source_attention_adv_loss))
    writer.add_scalar('Average Source Adv attention Loss/train', source_attention_adv_loss, epoch)
    add('Train: Average Target Adv attention Loss: {:.6f}'.format(target_attention_adv_loss))
    writer.add_scalar('Average Target Adv attention Loss/train', target_attention_adv_loss, epoch)

    add('Train: Average Source Feature Discriminator Loss: {:.6f}'.format(FeatureD_S_loss))
    writer.add_scalar('Average Source Feature Discriminator Loss/train', FeatureD_S_loss, epoch)
    add('Train: Average Target Feature Discriminator Loss: {:.6f}'.format(FeatureD_T_loss))
    writer.add_scalar('Average Target Feature Discriminator Loss/train', FeatureD_T_loss, epoch)

    add('Train: Average Source Attention Discriminator Loss: {:.6f}'.format(AttentionD_S_loss))
    writer.add_scalar('Average Source Attention Discriminator Loss/train', AttentionD_S_loss, epoch)
    add('Train: Average Target Attention Discriminator Loss: {:.6f}'.format(AttentionD_T_loss))
    writer.add_scalar('Average Target Attention Discriminator Loss/train', AttentionD_T_loss, epoch)
    add('Train: Average Loss: {:.6f}'.format(all_loss))
    writer.add_scalar('Average Loss/train', all_loss, epoch)
    return all_loss


def test(epoch, model, loss_fn, dataLoader):
    model.eval()
    test_loss = 0
    resultPre = None
    resultLabel = None
    visualLabel = None
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataLoader):
            img = sample_batched['image'].to(device)
            labels = sample_batched['attribute'].to(device)
            visual_labels = sample_batched['visual'].to(device)
            class_labels = sample_batched['class_label'].to(device)
            _, _, (pred, _, heatmap), _ = model(img, img, img, img)
            if args.celoss:
                index = 0
                m_index = 0
                res = labels.new_tensor(torch.zeros(labels.shape))
                class_labels = class_labels.long()
                losses = torch.zeros(len(param.exclution_groups)).to(device)
                for i, x in enumerate(param.exclution_groups):
                    if x > 1:
                        sub_pred = pred[:, index:index + x]
                        sub_res = sub_pred.new_tensor(torch.zeros(sub_pred.shape))
                        losses[i] = loss_fn(sub_pred, class_labels[:, i])
                        max_index = torch.argmax(sub_pred, dim=1)
                        max_index = torch.unsqueeze(max_index, dim=1)
                        max_index = max_index.long()
                        one_hot = sub_res.scatter_(1, max_index, 1)
                        res[:, m_index:m_index + x] = one_hot
                    else:
                        sub_pred = pred[:, index:index + 2]
                        losses[i] = loss_fn(sub_pred, class_labels[:, i])
                        max_index = torch.argmax(sub_pred, dim=1)
                        max_index = torch.unsqueeze(max_index, dim=1)
                        max_index = max_index.long()
                        res[:, m_index:m_index + x] = max_index
                        index = index + 1

                    index = index + x
                    m_index = m_index + x
                att_loss = torch.mean(losses)
            else:
                labels_F = labels.float()
                att_loss = loss_fn(pred, labels_F)
                res = torch.where(pred < 0.5, torch.full_like(labels, 0), torch.full_like(labels, 1))

            labels_I = labels.int()
            visual_labels_I = visual_labels.int()
            if resultPre is None:
                resultPre = res
                resultLabel = labels_I
                visualLabel = visual_labels_I
            else:
                resultPre = torch.cat((resultPre, res), 0)
                resultLabel = torch.cat((resultLabel, labels_I), 0)
                visualLabel = torch.cat((visualLabel, visual_labels_I), 0)
            test_loss += att_loss.data.cpu().numpy()
    test_loss /= len(dataLoader)
    accuracy, f1 = performance.cal_attribute_performance(num_attribute, resultLabel, resultPre, visualLabel)
    f1 = np.array(f1)
    f1_mean = np.mean(f1)
    accuracy = np.array(accuracy)
    accuracy_mean = np.mean(accuracy)

    add('Val: Average loss: {:.6f} \tAverage Accuracy :{}; \tAverage F1: {:.4f}'.format(test_loss, accuracy_mean, f1_mean))
    writer.add_scalar('Average Loss/Val', test_loss, epoch)
    writer.add_scalar('Average Accuracy/Val', accuracy_mean, epoch)
    writer.add_scalar('Average F1/Val', f1_mean, epoch)
    return test_loss, accuracy, f1, accuracy_mean, f1_mean


def adjust_lr(optimizer, epoch, maxepoch, init_lr, power):
    lr = init_lr * (1 - epoch / maxepoch) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def main(lr):
    add('Original LR : {}'.format(lr))
    add(str(args))
    modelType = args.modelType
    add(str(modelType) + " Attention Consistency model")

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
                                   NormalizeImageDict(['image', 'fake_image'])
                               ]))
    target_dataset = MyDataSet(usedata=target_data,
                               transform1=transforms.Compose([
                                   transforms.Resize(256),
                                   transforms.CenterCrop(224)
                               ]),
                               transform2=transforms.Compose([
                                   ToTensorDict(),
                                   NormalizeImageDict(['image', 'fake_image'])
                               ]))
    global tgt_data_path
    tgt_data_path = target_data.dir_path
    global num_attribute
    attributes = source_data.attributes
    num_attribute = source_dataset.num_attribute
    assert (source_dataset.num_attribute == target_dataset.num_attribute)
    add("count of classes:{}".format(num_attribute))
    # labels for adversarial training
    global source_label, target_label
    source_label = 0
    target_label = 1

    k = args.k
    k_fold_accuracy = []
    k_fold_f1 = []
    k_fold_test_loss = []
    k_fold_accuracy_mean = []
    k_fold_f1_mean = []
    for i in range(k):
        add('fold {} :'.format(i + 1))
        global writer
        writer = SummaryWriter(log_dir='new_runs/'+args.testname+'/fold' + str(i))
        target_train_index = np.load(os.path.join(data_save_dir, "target_train_indices_" + str(i+1) + ".npy"))
        target_val_index = np.load(os.path.join(data_save_dir, "target_val_indices_" + str(i+1) + ".npy"))
        model = AttentionConsisNet()
        # init D
        AttentionD_S = AttD()
        AttentionD_T = AttD()
        FeatureD_S = FeatureDiscriminator()
        FeatureD_T = FeatureDiscriminator()
        model.to(device)
        AttentionD_S.to(device)
        AttentionD_T.to(device)
        FeatureD_S.to(device)
        FeatureD_T.to(device)
        if torch.cuda.device_count() > 1:
            add("Let's use " + str(torch.cuda.device_count()) + " GPUs!")
            model = nn.DataParallel(model, device_ids=device_ids)
            AttentionD_S = nn.DataParallel(AttentionD_S, device_ids=device_ids)
            AttentionD_T = nn.DataParallel(AttentionD_T, device_ids=device_ids)
            FeatureD_S = nn.DataParallel(FeatureD_S, device_ids=device_ids)
            FeatureD_T = nn.DataParallel(FeatureD_T, device_ids=device_ids)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer_AttentionD_S = optim.Adam(AttentionD_S.parameters(), lr=args.lr_D, betas=(0.9, 0.99))
        optimizer_AttentionD_T = optim.Adam(AttentionD_T.parameters(), lr=args.lr_D, betas=(0.9, 0.99))
        optimizer_FeatureD_S = optim.Adam(FeatureD_S.parameters(), lr=args.lr_D, betas=(0.9, 0.99))
        optimizer_FeatureD_T = optim.Adam(FeatureD_T.parameters(), lr=args.lr_D, betas=(0.9, 0.99))
        if args.celoss:
            add("Let's use CrossEntropy loss!")
            att_loss = nn.CrossEntropyLoss(ignore_index=-1)
        else:
            add("Let's use MSE loss!")
            att_loss = nn.MSELoss()
        if args.hmloss == 'Vanilla':
            heatmap_loss = nn.BCEWithLogitsLoss()
            feature_loss = nn.BCEWithLogitsLoss()
        elif args.hmloss == 'LS':
            heatmap_loss = nn.MSELoss()
            feature_loss = nn.MSELoss()
        lab_consis_loss = nn.MSELoss()

        target_train_subset = torch.utils.data.Subset(target_dataset, target_train_index)
        target_val_subset = torch.utils.data.Subset(target_dataset, target_val_index)
        source_dataLoader = data.DataLoader(source_dataset, batch_size=args.batch_size,
                                            drop_last=True,
                                            shuffle=True,
                                            num_workers=args.workers
                                            )
        target_train_dataLoader = torch.utils.data.DataLoader(target_train_subset, batch_size=args.batch_size,
                                                              drop_last=True,
                                                              shuffle=True,
                                                              num_workers=args.workers)
        target_val_dataLoader = torch.utils.data.DataLoader(target_val_subset, batch_size=args.batch_size,
                                                            shuffle=True,
                                                            num_workers=args.workers)

        global len_source_dataset, len_target_dataset, len_target_train_index, len_target_val_index
        len_source_dataset = len(source_dataset)
        len_target_dataset = len(target_dataset)
        len_target_train_index = len(target_train_index)
        len_target_val_index = len(target_val_index)
        add('size of source set : {} '.format(len_source_dataset))
        add('size of target set : {} '.format(len_target_dataset))
        add('size of train in target set : {} '.format(len_target_train_index))
        add('size of val in target set : {} '.format(len_target_val_index))

        len_source_dataLoader = len(source_dataLoader)
        len_target_train_dataLoader = len(target_train_dataLoader)
        print(len_source_dataLoader)
        print(len_target_train_dataLoader)

        iteration = len_source_dataLoader

        best_test_loss = float("inf")
        best_acc_mean = -1.0
        best_f1_mean = -1.0

        print('Starting training...')
        start_epoch = args.start_epoch
        end_epoch = args.epochs
        for epoch in range(start_epoch, end_epoch + 1):
            add('epoch : {}'.format(epoch))
            train_loss = train(epoch, iteration, model, AttentionD_S, AttentionD_T, FeatureD_S, FeatureD_T, att_loss,
                               lab_consis_loss, feature_loss, heatmap_loss, optimizer, optimizer_AttentionD_S,
                               optimizer_AttentionD_T,optimizer_FeatureD_S, optimizer_FeatureD_T,
                               source_dataLoader, target_train_dataLoader)
            test_loss, accuracy, f1, accuracy_mean, f1_mean = test(epoch, model, att_loss, target_val_dataLoader)

            for param_group in optimizer.param_groups:
                add("lr:" + str(param_group['lr']))
            if epoch % 1 == 0:
                lr_now = adjust_lr(optimizer, epoch, end_epoch + 1, lr, power=0.75)
                add("current learning rate : {}".format(lr_now))

            is_best_loss = test_loss < best_test_loss
            best_test_loss = min(test_loss, best_test_loss)
            checkpoint_name_loss = os.path.join(
                result_dir, str(model.__class__.__name__) + "Fold " + str(i) + "_LR" + str(lr) + "_LOSS.pth")
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_test_loss': best_test_loss,
                'optimizer': optimizer.state_dict(),
            }, is_best_loss, checkpoint_name_loss)

            is_best_acc = accuracy_mean > best_acc_mean
            if is_best_acc:
                best_acc = accuracy
            best_acc_mean = max(accuracy_mean, best_acc_mean)
            checkpoint_name_acc = os.path.join(
                result_dir, str(model.__class__.__name__) + "Fold " + str(i) + "_LR" + str(lr) + "_ACC.pth")
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_test_loss': best_test_loss,
                'optimizer': optimizer.state_dict(),
            }, is_best_acc, checkpoint_name_acc)

            is_best_f1 = f1_mean > best_f1_mean
            if is_best_f1:
                best_f1 = f1
            best_f1_mean = max(f1_mean, best_f1_mean)
            checkpoint_name_f1 = os.path.join(
                result_dir, str(model.__class__.__name__) + "Fold " + str(i) + "_LR" + str(lr) + "_F1.pth")
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_test_loss': best_test_loss,
                'optimizer': optimizer.state_dict(),
            }, is_best_f1, checkpoint_name_f1)

        k_fold_accuracy.append(best_acc)
        k_fold_f1.append(best_f1)
        k_fold_test_loss.append(best_test_loss)
        k_fold_accuracy_mean.append(best_acc_mean)
        k_fold_f1_mean.append(best_f1_mean)

        add('Best Accuracy : \n{};\nBest F1: \n{}'
            .format(best_acc, best_f1))
        add('Best github loss: {:.4f};   Best Average Accuracy :{};   Best Average F1: {:.4f}'
            .format(best_test_loss, best_acc_mean, best_f1_mean))
        writer.close()
    k_fold_test_loss = np.array(k_fold_test_loss)
    k_fold_accuracy = np.array(k_fold_accuracy)
    k_fold_f1 = np.array(k_fold_f1)
    k_fold_accuracy_mean = np.array(k_fold_accuracy_mean)
    k_fold_f1_mean = np.array(k_fold_f1_mean)

    k_fold_test_loss = np.mean(k_fold_test_loss)
    k_fold_accuracy = np.mean(k_fold_accuracy, axis=0)
    k_fold_f1 = np.mean(k_fold_f1, axis=0)
    k_fold_accuracy_mean = np.mean(k_fold_accuracy_mean)
    avg_k_fold_f1_mean = np.mean(k_fold_f1_mean)
    add('accuracy:')
    for k in range(num_attribute):
        accuracy_rate = float(k_fold_accuracy[k])
        add(attributes[k] + " right rate is %0.04f" % accuracy_rate)
    add('\n')
    add('f1:')
    for k in range(num_attribute):
        f1_rate = float(k_fold_f1[k])
        add(attributes[k] + " right rate is %0.04f" % f1_rate)
    add('avg_test_loss : {};     avg_test_accuracy : {};     avg_test_f1 : {}'
        .format(k_fold_test_loss, k_fold_accuracy_mean, avg_k_fold_f1_mean))
    print('Done!')


if __name__ == '__main__':
    for lr in args.lr:
        main(lr)
