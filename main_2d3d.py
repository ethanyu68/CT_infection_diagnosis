import argparse
import random
import skimage.measure as measure

import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset.dataset_2d import DatasetFromHdf5 as DatasetFromHdf5_2D
from dataset.dataset_3d import DatasetFromHdf5 as DatasetFromHdf5_3D

from model_2d3d import _3D_DenseNet16, _3D_CNN
from model_2d3d import dense_cbam, DenseNet

from utils import *

import h5py
# Training settings
parser = argparse.ArgumentParser(description="Pytorch remoteCT classification")
parser.add_argument("--batchsize2d", type=int, default=48, help="Training batch size")
parser.add_argument("--batchsize3d", type=int, default=8, help="Training batch size")


parser.add_argument("--num_iter_toprint", type=int, default=10, help="Training patch size")
parser.add_argument("--patchsize", type=int, default=512, help="Training patch size")
parser.add_argument("--path_data", default="./data/PIH_NPIH/ct_2D3D_32.h5", type=str, help="Training datapath")#SynthesizedfromN18_256s64
parser.add_argument("--nEpochs", type=int, default=150, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.0002, help="Learning Rate, Default=0.1")
parser.add_argument("--lr_reduce", type=float, default=0.5, help="rate of reduction of learning rate, Default=0.4")
parser.add_argument("--block_config2d", type=int, default=(4,6,4,3), help="block configuration")
parser.add_argument("--block_config3d", type=int, default=(4,6,4,3), help="block configuration")


parser.add_argument("--b2", type=float, default=0.5, help="rate of reduction of learning rate, Default=0.4")
parser.add_argument("--b3", type=float, default=0.5, help="rate of reduction of learning rate, Default=0.4")
parser.add_argument("--w_cam2d3d", type=float, default=0.1, help="rate of reduction of learning rate, Default=0.4")
parser.add_argument("--w_3dmask", type=float, default=0.5, help="rate of reduction of learning rate, Default=0.4")
parser.add_argument("--w_2dmask", type=float, default=0.5, help="rate of reduction of learning rate, Default=0.4")


parser.add_argument("--masksize", type=int, default=512, help="Mask size")
parser.add_argument("--start_fold", type=int, default=4, help="number of folds in cross-validation")
parser.add_argument("--end_fold", type=int, default=6, help="number of folds in cross-validation")
parser.add_argument("--num_folds", type=int, default=5, help="number of folds in cross-validation")
parser.add_argument("--in_channels", type=int, default=1, help="number of channels in input")
parser.add_argument("--out_classes", type=int, default=1, help="number of classes in output")
parser.add_argument("--only3D", type=int, default=0, help="1:only use 3D data, 0: 2D + 3D")
parser.add_argument("--only2D", type=int, default=1, help="1:only use 3D data, 0: 2D + 3D")
parser.add_argument("--step", type=int, default=25, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default=5")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--aug", type=int,default=1, help="Use aug?")

parser.add_argument("--resume2d", default=0, type=int, help="Path to checkpoint, Default=None")
parser.add_argument("--resume3d", default="./model/dense_2d3d_fold1of5/model_epoch_3200.pth", type=str, help="Path to checkpoint, Default=None")
parser.add_argument("--start_epoch", default=1, type = int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--clip", type=float, default=0.01, help="Clipping Gradients, Default=0.01")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default=1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default=0.9")
parser.add_argument("--weight_decay", "--wd", default=1e-6, type=float, help="Weight decay, Default=1e-4")
parser.add_argument("--pretrained", default="", type=str, help='path to pretrained model_files, Default=None')
parser.add_argument("--activation", default="no_relu", type=str, help='activation relu')
parser.add_argument("--ID", default="2d3d", type=str, help='ID for training')
parser.add_argument("--label", default="pih", type=str, help='ID for training')
parser.add_argument("--model", default="dense_cbam", type=str, help="choices: densenet/raresnet/resnet")
parser.add_argument("--alter", action="store_true", help="alternation for training??")


def main():
    global opt, idxv_tb_patient, CE2D_tr_epoch, CE3D_tr_epoch, df, cam_bank2D, cam_bank3D
    opt = parser.parse_args()
    print(opt)
    # experiment setting
    setting_df = pd.DataFrame({'model': opt.model,
                       'block_config2d': opt.block_config2d,
                       'block_config3d': opt.block_config3d,
                       'data': opt.path_data,
                       'lr': opt.lr,
                       'step': opt.step,
                       'batchsize2d':opt.batchsize2d,
                       'batchsize3d': opt.batchsize3d,
                       'augmentation': opt.aug,
                       'loss': 'CE + 0.00001CAM loss'
                       })
    # resume models
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


    opt.cuda = True
    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    cudnn.benchmark = True
    # loss function
    CEloss = nn.CrossEntropyLoss().cuda()
    # h5data
    h5data = h5py.File(opt.path_data, 'r')
    ####################################################################################################################
    # 5 fold cross-validation begins
    for fold in range(opt.start_fold, opt.end_fold):
        CE2D_tr_epoch = [0.7]
        CE3D_tr_epoch = [0.7]
        print("===> Fold-{}/{}".format(fold, opt.num_folds))
        print("===> Building model")
        model2d = DenseNet(block_config=opt.block_config2d)
        #model2d = dense_cbam(block_config=opt.block_config2d)
        model3d = _3D_DenseNet16(block_config=opt.block_config3d)
        #model3d = _3D_CNN()
        model2d = torch.nn.DataParallel(model2d).cuda()
        model3d = torch.nn.DataParallel(model3d).cuda()
        print("===> Setting Optimizer")
        optimizer2d = optim.Adam(model2d.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        optimizer3d = optim.Adam(model3d.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

        # path for saving model_files of this fold
        path_data = opt.path_data
        name_data = path_data.split('/')[-1][:-3]
        save_path2d = os.path.join('.', "model", "Dense2d3d_2D_{}_{}_fold{}of{}".format(opt.label, name_data, fold, opt.num_folds))
        save_path3d = os.path.join('.', "model", "Dense2d3d_3D_{}_{}_fold{}of{}".format(opt.label, name_data, fold, opt.num_folds))
        if not os.path.exists(save_path2d):
            os.makedirs(save_path2d)
        if not os.path.exists(save_path3d):
            os.makedirs(save_path3d)
        setting_df.to_csv(os.path.join(save_path3d, 'exp_setting.csv'))

        train_set_2D = DatasetFromHdf5_2D(opt.path_data, fold, opt.num_folds, only_PIH=False, augmentation=opt.aug)
        training_data_loader2D = DataLoader(dataset=train_set_2D, num_workers=opt.threads, batch_size=opt.batchsize2d,
                                              shuffle=True)
        train_set_3D = DatasetFromHdf5_3D(opt.path_data, fold, opt.num_folds, only_PIH=False, augmentation=opt.aug)
        training_data_loader3D = DataLoader(dataset=train_set_3D, num_workers=opt.threads, batch_size=opt.batchsize3d,
                                            shuffle=True)

        if opt.label == 'pih':
            num_scans = h5data['Hydrocephalus'].shape[0]
            idx_val = np.arange((fold - 1) * num_scans//opt.num_folds, fold * num_scans//opt.num_folds).tolist()
            idx_tr = np.delete(np.arange(num_scans), idx_val).tolist()
            idxv_tb_patient = h5data['index_table_patient'][idx_tr]
        elif opt.label == 'paeni':
            pos_pih = np.where(h5data['Hydrocephalus'][...] == 1)[0]
            num_scans = pos_pih.shape[0]
            idx_val = np.arange((fold - 1) * num_scans//opt.num_folds, fold * num_scans//opt.num_folds).tolist()
            idx_tr = np.delete(np.arange(num_scans), idx_val).tolist()
            idxv_tb_patient = h5data['index_table_patient'][idx_tr]
        else:
            raise Exception('opt.label is incorrect')

        acc2d_epoch, acc3d_epoch = [], []
        CE2d_epoch, CE3d_epoch = [], []
        cam_bank3D = torch.zeros([num_scans, 14, 14], dtype=torch.float32).cuda()
        cam_bank2D = torch.zeros([num_scans, 14, 14], dtype=torch.float32).cuda()
        for epoch in range(opt.start_epoch, opt.nEpochs + 1):
            acc2d, sen2d, spec2d, CE2d, cam2d = eval2d(model2d, h5data, fold, opt)
            acc3d, sen3d, spec3d, CE3d = eval3d(model3d, h5data, fold, opt)
            result2d = [acc2d, sen2d, spec2d, CE2d, acc2d_epoch, CE2d_epoch]
            result3d = [acc3d, sen3d, spec3d, CE3d, acc3d_epoch, CE3d_epoch]
            save_bestepoch(model2d, model3d, save_path2d, save_path3d, result2d, result3d, epoch, fold)
            train2d(training_data_loader2D, optimizer2d, model2d, epoch, CEloss, opt)
            cam_bank3D = torch.zeros([num_scans, 14, 14], dtype=torch.float32).cuda()
            train3d(training_data_loader3D, optimizer3d, model3d, epoch, CEloss, opt)
            cam_bank2D = torch.zeros([num_scans, 14, 14], dtype=torch.float32).cuda()

            print("===> Train 2D")




def train3d(training_data_loader, optimizer, model3d, epoch, CEloss, opt):
    lr = adjust_learning_rate(epoch-1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    model3d.train()
    loss_tr = []
    for iteration, batch in enumerate(training_data_loader, 1):
        input_3d, mask, pih_npih, cmv, paeni, idxv  = batch
        bsz = idxv.shape[0]
        cam2D = torch.zeros([bsz, 1, 14, 14]).float().cuda()
        input_3d = input_3d.cuda()
        #input_3d = augmentation(input_3d)
        # form the tables of label and select data to train
        # input_data: N x C x H x W; info: N x 6
        # out: N x num_classes; cam: N x num_classes x dim_feature_map x dim_feature_map
        # out is unnormalized
        out_3d, cam = model3d(input_3d)
        if opt.label == 'pih':
            gt = pih_npih.view(-1).cuda()
        elif opt.label == 'paeni':
            gt = paeni.view(-1).cuda()
        else:
            raise ValueError('opt.label is wrong')
        CE = CEloss(out_3d, gt)  # CE input1: (N, num_classes) input2: (N,)
        diff_cam = torch.abs((cam) * (1 - mask.cuda()))
        loss_cam = torch.sum((diff_cam) ** 2)
        for p in range(bsz):
            # save cam to cam bank 3D
            cam_bank3D[idxv[p]] = cam_bank3D[idxv[p]] + cam[p, gt[p]].detach()
            # fetch cam from cam bank 2D
            cam2D[p, 0] = cam_bank2D[idxv[p]]
        maxv, _ = cam2D.view(bsz, -1).max(1)
        # mask_att: 8x1x14x14
        mask_att = 1/(1 + torch.exp(-10*(cam2D - 0.5*maxv.view(-1, 1, 1, 1))))
        mask_att[mask_att < 0.8] = 0
        # mean_att: 8x2
        mean_att = torch.sum(mask_att * cam, (2,3))/torch.sum(mask_att, (2,3))
        # sum_att: 8x1
        sum_att = torch.log(torch.exp(mean_att[:, 0]) + torch.exp(mean_att[:, 1]))
        CE_att = torch.mean((gt.T - 1) * mean_att[:, 0] - gt.T * mean_att[:, 1] + sum_att)
        if epoch <0:
            loss = CE + 0.05*CE_att + 0.00001*loss_cam
        else:
            loss = CE + 0.00001*loss_cam

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_tr.append(loss.cpu().detach().numpy())
        if (iteration % opt.num_iter_toprint == 0):
            print("===> Epoch[{}]({}/{},lr:{:.8f}): CE:{:.6f}, CE_att:{:.6f}" "".format(epoch, iteration,
                                                    len(training_data_loader),lr, CE, CE_att))
    CE3D_tr_epoch.append(np.mean(loss_tr))


def train2d(training_data_loader, optimizer, model, epoch,  CEloss, opt):
    lr = adjust_learning_rate(epoch-1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    model.train()
    loss_tr = []

    for iteration, batch in enumerate(training_data_loader, 1):
        input_data, mask, pih_npih, cmv, paeni, idxv = batch
        bsz = idxv.shape[0]
        cam3D = torch.zeros([bsz, 1, 14, 14]).float().cuda()
        # form the tables of label and select data to train
        # input_data: N x C x H x W; info: N x 6
        # out: N x num_classes; cam: N x num_classes x dim_feature_map x dim_feature_map
        # out is unnormalized

        if opt.label == 'paeni':
            gt = paeni.cuda()
            out, cam = model(input_data.cuda())
        elif opt.label == 'cmv':
            gt = cmv.cuda()
            out, cam = model(input_data.cuda())
        else:
            gt = pih_npih.cuda()
            out, cam = model(input_data.cuda())
        CE = CEloss(out, gt.view(-1).cuda())
        diff_cam = torch.abs((cam) * (1 - mask.cuda()))
        loss_cam = torch.sum((diff_cam) ** 2)
        abs_cam =  torch.sum(abs(cam))

        for p in range(bsz):
            # save cam to cam bank 2D
            cam_bank2D[idxv[p]] = cam_bank2D[idxv[p]] + cam[p, gt[p], 1:-1, 1:-1].detach()
            # fetch cam from cam bank 3D
            cam3D[p, 0] = cam_bank3D[idxv[p]]
        maxv, _ = cam3D.view(bsz, -1).max(1)
        # mask_att: 8x1x14x14
        mask = mask[:,:,1:-1,1:-1].cuda()
        mask_att = mask*(1 - 1/(1 + torch.exp(-10*cam3D)))
        mask_att[mask_att<0.8] = 0
        # mean_att: 8x2
        mean_att = torch.sum(mask_att * cam[:, :, 1:-1, 1:-1], (2,3))/torch.sum(mask_att, (2,3))
        # sum_att: 8x1
        sum_att = torch.log(torch.exp(mean_att[:, 0]) + torch.exp(mean_att[:, 1]))
        CE_att = torch.mean((gt.T-1)*mean_att[:, 0] - gt.T*mean_att[:, 1] + sum_att)
        if epoch <0:
            loss = CE - 0.05*CE_att + 0.00001*loss_cam
        else:
            loss = CE + 0.00001*loss_cam

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_tr.append(CE.cpu().detach().numpy())
        if iteration%opt.num_iter_toprint == 0:
             print("===> Epoch[{}]({}/{},lr:{:.8f}): CE:{:.6f}, CE_att:{:.4f}".format(epoch, iteration,
                                                    len(training_data_loader), lr, CE, CE_att))
    CE2D_tr_epoch.append(np.mean(loss_tr))



def save_bestepoch(model2d, model3d, save2d_path, save3d_path, result2d, result3d, epoch, fold):
    '''
    :param model2d:
    :param model3d:
    :param save2d_path:
    :param save3d_path:
    :param result2d: [acc2d, sen2d, spec2d, CE2d, acc2d_epoch, CE2d_epoch]
    :param result3d:
    :param epoch:
    :param fold:
    :return:
    '''
    acc2d, sen2d, spec2d, CE2d, acc2d_epoch, CE2d_epoch = result2d[0], result2d[1],result2d[2], result2d[3],result2d[4], \
                                                          result2d[5]
    acc3d, sen3d, spec3d, CE3d, acc3d_epoch, CE3d_epoch = result3d[0], result3d[1], result3d[2], result3d[3], result3d[4],\
                                                          result3d[5]
    acc2d_epoch.append(acc2d)
    CE2d_epoch.append(CE2d)
    acc3d_epoch.append(acc3d)
    CE3d_epoch.append(CE3d)
    print("===> 2D: Accuracy/Sensitivity/Specificify of fold-{} epoch-{}: {:.4f}/{:.4f}/{:.4f}".format(fold, epoch - 1,
                                                                                                       acc2d, sen2d, spec2d))
    print("===> 3D: Accuracy/Sensitivity/Specificify of fold-{} epoch-{}: {:.4f}/{:.4f}/{:.4f}".format(fold, epoch - 1,
                                                                                                       acc3d, sen3d,
                                                                                                       spec3d))
    print("===> 2D: Error CE:{:.4f}".format(CE2d))
    print("===> 3D: Error CE:{:.4f}".format(CE3d))
    max2d_acc = np.max(acc2d_epoch)
    acc2d_best_epoch = np.where(np.array(acc2d_epoch) == max2d_acc)[0][-1]
    print("===> 2D: Highest Accuracy:ep-{}-{:4f}".format(acc2d_best_epoch, max2d_acc))
    max3d_acc = np.max(acc3d_epoch)
    acc3d_best_epoch = np.where(np.array(acc3d_epoch) == max3d_acc)[0][-1]
    print("===> 3D: Highest Accuracy:ep-{}-{:4f}".format(acc3d_best_epoch, max3d_acc))
    save_checkpoint(model2d, epoch - 1, save2d_path)
    save_checkpoint(model3d, epoch - 1, save3d_path)
    if epoch > 5 :
        save_fig(acc2d_epoch, acc3d_epoch, CE2d_epoch, CE3d_epoch, save2d_path, fold)


def save_fig(acc2d_epoch, acc3d_epoch, CE2d_epoch, CE3d_epoch, save_path, fold):
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(acc2d_epoch)
    plt.plot(acc3d_epoch)
    plt.legend(['2D:val accuracy', '3D:val accuracy'])
    max2d_acc = np.max(acc2d_epoch)
    acc2d_best_epoch = np.where(acc2d_epoch == max2d_acc)[-1][0]
    max3d_acc = np.max(acc3d_epoch)
    acc3d_best_epoch = np.where(acc3d_epoch == max3d_acc)[-1][0]
    plt.scatter(acc2d_best_epoch, max2d_acc)
    plt.scatter(acc3d_best_epoch, max3d_acc)
    plt.annotate('ep{}\n{:.4f}'.format(acc2d_best_epoch, max2d_acc*100), (acc2d_best_epoch + 1, max2d_acc))
    plt.annotate('ep{}\n{:.4f}'.format(acc3d_best_epoch, max3d_acc*100), (acc3d_best_epoch + 1, max3d_acc))

    plt.subplot(3, 1, 2)
    plt.plot(CE2d_epoch)
    plt.plot(CE2D_tr_epoch)
    plt.legend(['2D:val CE', '2D:tr CE'])
    min2d_CE = np.max(CE2d_epoch)
    CE2d_best_epoch = np.where(CE2d_epoch ==min2d_CE)[-1][0]
    plt.scatter(CE2d_best_epoch, min2d_CE)
    plt.annotate('ep{}\n{:.4f}'.format(CE2d_best_epoch, min2d_CE * 100), (CE2d_best_epoch + 1, min2d_CE))

    plt.subplot(3, 1, 3)
    plt.plot(CE3d_epoch)
    plt.plot(CE3D_tr_epoch)
    plt.legend(['3D:val CE', '3D:tr CE'])
    min3d_CE = np.max(CE3d_epoch)
    CE3d_best_epoch = np.where(CE3d_epoch == min3d_CE)[-1][0]
    plt.scatter(CE3d_best_epoch, min3d_CE)
    plt.annotate('ep{}\n{:.4f}'.format(CE3d_best_epoch, min3d_CE * 100), (CE3d_best_epoch + 1, min3d_CE))


    plt.savefig(os.path.join(save_path, 'Loss_acc_fold{}'.format(fold)))
    plt.close()


def save_checkpoint(model, epoch, save_path):
    model_out_path = os.path.join(save_path, "model_epoch_{}.pth".format(epoch))
    state = {"epoch": epoch, "model": model}
    # check path status
    if not os.path.exists("model/"):
        os.makedirs("model/")
    # save model_files
    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def adjust_learning_rate(epoch):
    lr = opt.lr * (opt.lr_reduce ** (epoch // opt.step))
    return lr


def eval2d(model, h5file, fold, opt):
    model.eval()
    ######################################################################################
    id_patient = h5file['index_table_patient'][...]
    id_slice = h5file['index_table_slice'][...]
    data2d = h5file['data2d']
    if opt.label == 'paeni':
        pos_pih = np.where(h5file['Hydrocephalus'][...].astype(int) == 1)[0]
        num_data = len(pos_pih)
        num_val = num_data // opt.num_folds
        idx_val = pos_pih[np.arange((fold - 1) * num_val, fold * num_val)]
        GT = torch.tensor(h5file['paeni_04052021'][idx_val]).ravel().cuda()
    elif opt.label == 'pih':
        num_data = id_patient.shape[0]
        num_val = num_data//opt.num_folds
        idx_val = np.arange((fold- 1)*num_val, (fold)*num_val)
        GT = torch.tensor([h5file['Hydrocephalus'][idx_val]]).ravel().cuda().float()

    output = torch.zeros([num_val,2]).cuda()
    cam = torch.zeros([num_val, 2, 16, 16]).cuda()
    for i in range(num_val):
        id = id_patient[idx_val[i]]
        loc_slices = np.nonzero(id_slice == id)[0]
        slices = torch.tensor(data2d[list(loc_slices)]).cuda()
        with torch.no_grad():
            out_p, cam_p = model(slices)
        cam[i] = torch.mean(cam_p, 0)
        output[i] = torch.mean(out_p, 0)

    p1 = torch.exp(output[:, 1]) / (torch.exp(output[:, 0]) + torch.exp(output[:, 1]))
    CE = torch.mean(- GT.ravel() * torch.log(p1) - (1 - GT.ravel()) * torch.log(1 - p1))
    results = 1 - (abs(torch.argmax(output, 1) - GT))
    accuracy = torch.mean(results).cpu().numpy()

    sensitivity = torch.sum(((results == 1) & (GT == 1)))/torch.sum(GT == 1)
    specificity = torch.sum(((results == 1) & (GT == 0)))/torch.sum(GT == 0)


    return accuracy, sensitivity.cpu().numpy(), specificity.cpu().numpy(), CE.cpu().numpy(), cam


def eval3d(model3d, h5data, fold, opt):
    model3d.eval()

    idxv_tb_patient = h5data['index_table_patient']
    pih_npih = h5data['Hydrocephalus'][...]
    num_patients = idxv_tb_patient.shape[0]
    if opt.label == 'paeni':
        pos_pih = np.where(pih_npih == 1)[0]
        num_val = len(pos_pih) // opt.num_folds
        indices_in_all = pos_pih[np.arange((fold - 1) * num_val, fold * num_val).tolist()]
        label_val = h5data['paeni_04052021'][indices_in_all]
    elif opt.label == 'pih':
        num_val = num_patients//opt.num_folds
        indices_in_all = np.arange((fold - 1) * num_val, fold * num_val).tolist()
        label_val = pih_npih[indices_in_all]

    data3d = h5data['data3d']

    out3d = np.zeros([num_val, 2])
    for i in range(num_val):
        idx = indices_in_all[i]
        input3d = torch.zeros([6,1,16,448,448]).cuda()
        for s in range(6):
            input3d_s = data3d[idx, :, 5+s:5+s+16, 32:-32, 32:-32]
            input3d[s,:,:,:,:] = torch.tensor(input3d_s, dtype=torch.float32).cuda()

        with torch.no_grad():
            out3d_i, cam = model3d(input3d)
        out3d[i, :] = np.mean(out3d_i.cpu().numpy(), 0)
    pred3d = np.argmax(out3d, 1)
    prob = np.exp(out3d[:, 1])/(np.exp(out3d[:, 0]) + np.exp(out3d[:, 1]))
    CE = np.mean(- label_val.ravel() * np.log(prob) - (1- label_val.ravel()) * np.log(1 - prob))
    correct = 1 - abs(pred3d - label_val.T)

    accuracy3d = np.sum(correct) /  num_val

    sens1, spec1 = sensitivity_specificify(correct.ravel(), label_val.ravel())

    return  accuracy3d, sens1, spec1, CE



def sensitivity_specificify(correct, labels):
    sensitivity = np.sum((correct == 1) & (labels == 1))/np.sum(labels == 1)
    specificity = np.sum((correct == 1) & (labels == 0))/np.sum(labels == 0)
    return sensitivity, specificity



if __name__ == "__main__":
    main()
