import numpy as np
import math, os
import pdb
import torch
import shutil

def checkdirctexist(dirct):
	if not os.path.exists(dirct):
		os.makedirs(dirct)

def save_exp(root_dir, exp_name, codes):
    exp_dir = os.path.join(root_dir, 'experiments', exp_name)

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    if not os.path.exists(os.path.join(exp_dir, 'results')):
        os.makedirs(os.path.join(exp_dir, 'results'))
    if not os.path.exists(os.path.join(exp_dir, 'codes')):
        os.makedirs(os.path.join(exp_dir, 'codes'))
    if not os.path.exists(os.path.join(exp_dir, 'loss')):
        os.makedirs(os.path.join(exp_dir, 'loss'))
    if not os.path.exists(os.path.join(exp_dir, 'loss')):
        os.makedirs(os.path.join(exp_dir, 'loss'))

    shutil.copy(os.path.join(root_dir, codes['main']), os.path.join(exp_dir, 'codes', codes['main']))
    shutil.copy(os.path.join(root_dir, codes['model']), os.path.join(exp_dir, 'codes', codes['model']))
    shutil.copy(os.path.join(root_dir, codes['dataset']), os.path.join(exp_dir, 'codes', codes['dataset']))
    return exp_dir
    

    
def sensitivity_specificify(correct, labels):
    sensitivity = torch.sum((correct == 1) & (labels == 1)).float()/torch.sum(labels == 1).float()
    specificity = torch.sum((correct == 1) & (labels == 0)).float()/torch.sum(labels == 0).float()
    return sensitivity, specificity


def PSNR_self(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = (pred -gt)
    pred_np = pred.cpu().data.numpy()
    gt_np = pred.cpu().data.numpy()
    rmse = math.sqrt(np.mean(imdff.cpu().data.numpy() ** 2))
    if rmse == 0:
        return 100
    return 20.0 * math.log10(1/rmse)


def adjust_learning_rate(epoch, opt):
    lr = opt.lr * (opt.lr_reduce ** (epoch // opt.step))
    return lr


def save_checkpoint(model, epoch, save_path):
    model_out_path = os.path.join(save_path, "model_epoch_{}.pth".format(epoch))
    state = {"epoch": epoch, "model": model}
    # check path status
    if not os.path.exists("model/"):
        os.makedirs("model/")
    # save model_files
    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))



