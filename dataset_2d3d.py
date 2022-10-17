from tracemalloc import start
import torch.utils.data as data
import torch
import h5py, cv2
import numpy as np
from skimage.transform import rotate
import scipy.ndimage as ndimage
import torch
import skimage.measure as measure
import matplotlib.pyplot as plt
import pandas as pd


class DatasetFromHdf5(data.Dataset):
    '''
       dataloader for single scan
       file_path: the path to the training h5 file
       fold: the number of fold in k-fold training configuration
       num_folds: the number of total folds

       outputs: data_scan, downsampled mask, information vector(6 binary digits), pih_npih(0/1), cmv(0/1), paeni(0/1)

       structure:
       data2d   idxv_tb_slice   idxv_tb_patient   data3d            info
       slice1   2001            2001              3d scans of 2001  info of 2001
       slice2   2001            2002              3d scans of 2002  info of 2002
       slice3   2001            2003              3d scans of 2003  info of 2003
         :       :                :
         :       :
       slice29  2002            2029              3d scans of 2029  info of 2029
       slice30  2002            2030              3d scans of 2030  info of 2030
         :
         :
    '''
    def __init__(self, opt, fold):
        super(DatasetFromHdf5, self).__init__()
        self.table = pd.read_csv(opt.path_table)
        hfdata = h5py.File(opt.path_data, 'r')
        hfmask = h5py.File(opt.path_mask, 'r')
        self.data = hfdata.get('image')
        self.mask = hfmask.get('mask')
        
        self.pos_pih = np.where(self.table['PIH'] == 1)[0]

        self.num_patients = len(self.table) # number of patients
        self.num_PIHpatients = len(self.pos_pih)  # number of patients
        self.num_eachfold = self.num_patients // opt.num_folds
        self.num_PIH_eachfold = self.num_PIHpatients // opt.num_folds

        # patient number for validataion
        if opt.label == 'paeni':
            self.val_index = np.arange((fold - 1) * self.num_PIH_eachfold, fold * self.num_PIH_eachfold)
            self.tr_index = self.pos_pih[np.delete(np.arange(self.num_PIHpatients), self.val_index)]
            
        else:
            self.val_index = np.arange((fold - 1) * self.num_eachfold, fold * self.num_eachfold)
            self.tr_index = np.delete(np.arange(self.num_patients), self.val_index)


        low_tr_idx = np.random.permutation(self.tr_index.shape[0])[:int(opt.tr_ratio * self.tr_index.shape[0])]
        self.tr_patient_idx = self.tr_index[low_tr_idx]
        # get number of scans to train at each epoch
        self.num_scans = self.tr_patient_idx.shape[0]

    def __getitem__(self, index):
        while True:
            table_idx = self.tr_patient_idx[index]
            # look up the position of this slice in the data2d array 
            length = self.table['end_index'][table_idx] - self.table['start_index'][table_idx]
            if length < 17:
                index = np.random.randint(self.num_scans)
            else:
                break

        r = np.random.randint(length - 16)
        start_idx = self.table['start_index'][table_idx] + r
        # fetch the info
        pih_npih = torch.from_numpy(np.array(self.table['PIH'][table_idx])).long()
        paeni = torch.from_numpy(np.array(self.table['Paeni'][table_idx])).long()
        # 3dimage
        input3d16 = self.data[start_idx:start_idx + 16,  0, 32:-32, 32:-32]
        r_2d = np.random.randint(7, 10)
        mask = self.mask[start_idx+r_2d, 0, 32:-32, 32:-32]
        #mask[mask>0] = 1
 #======# augmentation # ========================================================================================================================

        D, H, W = input3d16.shape
        # resize
        r1 = np.random.randint(3)
        r = np.random.randint(32)
        if r1 == 1:
            for i in range(D):
                input3d16[i] = cv2.resize(input3d16[i, r: H - r, r: W - r], (H, W))
            #input2d = cv2.resize(input2d[r: H - r, r: W - r], (H, W))
            mask = cv2.resize(mask[r: H - r, r: W - r], (H, W))
        mask = measure.block_reduce(mask, (32, 32))
        mask[mask > 0] = 1
        # rotation
        r1 = np.random.randint(3)
        if r1 == 1:
            r = np.random.randint(-3,3)
            for i in range(D):
                input3d16[i, :, :] = rotate(input3d16[i], angle=r*5, mode='wrap')
            #input2d = rotate(input2d, angle=r * 5, mode='wrap')
            mask = rotate(mask, angle=r * 5, mode='wrap')
        # flipping
        r1 = np.random.randint(3)
        if r1 == 1:
            r = np.random.randint(2)
            for i in range(D):
                input3d16[i, :, :] = np.flip(input3d16[i], axis=r)
            #input2d = np.flip(input2d, axis=r)
            mask = np.flip(mask, axis=r)
        # noise
        #r1 = np.random.randint(3)
        #if r1 == 1:
        #    r = np.random.randint(3)
        #    input3d16 = np.random.normal(0, 0.005 * r, [D, H, W]) + input3d16
            #input2d = np.random.normal(0, 0.005 * r, [H, W]) + input2d

        # 3D 16
        # 3D 8
        #idx_slice8 = np.arange(8) + np.random.randint(8)
        #input3d8 = input3d16[idx_slice8]
        # 2D
        
        input2d = input3d16[r_2d:r_2d+1, :, :]
        #
        #input3d = input3d[np.newaxis, :, :, :]
        input3d16 = input3d16[np.newaxis, :, :, :]
        #input3d8 = input3d8[np.newaxis, :, :, :]
        mask = mask[np.newaxis, :, :]
        #for k in range(D):
        #    cv2.imwrite('./_2D3D/experiments/cbar_noCBAR_deep2D/results/tr_idx{}_{}.jpg'.format(index, k), input3d16[0, k, :, :]*255)
            

        return torch.from_numpy(input3d16.copy()).float(),torch.from_numpy(input2d.copy()).float(), torch.from_numpy(mask.copy()).float(), pih_npih, paeni

    def __len__(self):
        return self.num_scans




class DatasetFromHdf5_val(data.Dataset):
    '''
       dataloader for single scan
       file_path: the path to the training h5 file
       fold: the number of fold in k-fold training configuration
       num_folds: the number of total folds

       outputs: data_scan, downsampled mask, information vector(6 binary digits), pih_npih(0/1), cmv(0/1), paeni(0/1)

       structure:
       data2d   idxv_tb_slice   idxv_tb_patient   data3d            info
       slice1   2001            2001              3d scans of 2001  info of 2001
       slice2   2001            2002              3d scans of 2002  info of 2002
       slice3   2001            2003              3d scans of 2003  info of 2003
         :       :                :
         :       :
       slice29  2002            2029              3d scans of 2029  info of 2029
       slice30  2002            2030              3d scans of 2030  info of 2030
         :
         :
    '''
    def __init__(self, opt, fold):
        super(DatasetFromHdf5_val, self).__init__()
        self.table = pd.read_csv(opt.path_table)
        hfdata = h5py.File(opt.path_data, 'r')
        hfmask = h5py.File(opt.path_mask, 'r')
        self.data = hfdata.get('image')
        self.mask = hfmask.get('mask')
        
        self.pos_pih = np.where(self.table['PIH'] == 1)[0]

        self.num_patients = len(self.table) # number of patients
        self.num_PIHpatients = len(self.pos_pih)  # number of patients
        self.num_eachfold = self.num_patients // opt.num_folds
        self.num_PIH_eachfold = self.num_PIHpatients // opt.num_folds

        # patient number for validataion
        if opt.label == 'paeni':
            self.val_index = np.arange((fold - 1) * self.num_PIH_eachfold, fold * self.num_PIH_eachfold)
        else:
            self.val_index = np.arange((fold - 1) * self.num_eachfold, fold * self.num_eachfold)        
        self.patient_idx = self.val_index

        # get number of scans to train at each epoch
        self.num_scans = self.patient_idx.shape[0]

    def __getitem__(self, index):
        while True:
            table_idx = self.patient_idx[index]
            # look up the position of this slice in the data2d array 
            length = self.table['end_index'][table_idx] - self.table['start_index'][table_idx]
            if length < 17:
                index = np.random.randint(self.num_scans)
            else:
                break

        # fetch the info
        pih_npih = torch.from_numpy(np.array(self.table['PIH'][table_idx])).long()
        paeni = torch.from_numpy(np.array(self.table['Paeni'][table_idx])).long()
        # 3dimage
        start_idx = self.table['start_index'][table_idx]
        end_idx = self.table['end_index'][table_idx]
        input3d_batch = []
        input2d_batch = []

        for k in range(start_idx, end_idx):
            cv2.imwrite('./_2D3D/experiments/cbar_noCBAR_deep2D/results/start{}_end{}_{}.jpg'.format(start_idx, end_idx, k), self.data[k, 0, :, :]*255)
            

        for l in range(start_idx, end_idx - 16):
            input3d16 = self.data[l:l + 16,  0, 32:-32, 32:-32]
            for k in range(7,10):
                input2d = input3d16[k]
                input3d_batch.append(input3d16)
                input2d_batch.append(input2d)
        input3d_batch = np.array(input3d_batch)[:, None, :, :, :]
        input2d_batch = np.array(input2d_batch)[:, None, :, :]

        return torch.from_numpy(input3d_batch).float(),\
                torch.from_numpy(input2d_batch).float(), \
                    pih_npih, paeni

    def __len__(self):
        return self.num_scans


