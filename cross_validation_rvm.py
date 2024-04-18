# this is the train on one day
# to save the memory usage, the data is loaded subject by subject
import torch
import time
import numpy as np
import h5py
import datetime
import os
import copy
import csv
import pickle
import torch.nn as nn
import mne

from utils_cog import *
from train_model_hr import *
from pathlib import Path
from eventinfo import *

from datetime import datetime
from scipy.io import loadmat


class CrossValidation:
    def __init__(self, arg):
        # init all the parameters here
        # arg is a dictionary containing parameter settings
        self.arg = arg
        self.data = None
        self.label = None
        self.model = None
        self.data_path = os.path.join(os.getcwd(), 'data_processed')
        self.data_dir = arg.get('data_dir') 
        self.label_type = arg.get('label_type')
        self.cv = arg.get('cross_validation')

        if self.arg.get('cross_validation') == '10-fold':
            record_file = 'n_fold_result' + datetime.now().strftime("%m-%d-%Y-%H-%M-%S") + '.txt'
            self.arg['record_file'] = record_file

        elif self.arg.get('cross_validation') == 'n-fold':
            record_file = 'n_fold_result' + datetime.now().strftime("%m-%d-%Y-%H-%M-%S") + '.txt'
            self.arg['record_file'] = record_file
            
        elif self.arg.get('cross_validation') == 'leave_one_subject_out':
            record_file = 'leaveOut_result.txt'
            self.arg['record_file'] = record_file
        # Save to log file for checking

        file = open(record_file, 'a')
        file.write("\n" + str(datetime.now()) +
                   "\nTrain:Parameter setting for " + str(arg.get('model')) +
                   "\n ----| classification:" + str(arg.get('class')) + "\n" +
                   "\n ----| segmentation:" + str(arg.get('segmentation')) +
                   "\n ----| segment length :" + str(arg.get('window')) +
                   "\n ----| segment shift :" + str(arg.get('shift')) + "\n" +
                   "\n ----| channel group :" + str(arg.get('channel_list')) + "\n" +
                   "\n ----| model type :" + str(arg.get('model')) +
                   "\n ----| model keyword :" + str(arg.get('model_name')) + "\n")

        file.close()

    # Load the data subject by subject  
    def load_data_per_subject(self, sub):
        file = 'sub' + str(sub) + '.hdf'
        subject_path = os.path.join(self.data_path, file)

        subject = h5py.File(subject_path, 'r')
        label = np.array(subject['label'])
        data = np.array(subject['data'])
        #   data: trial x segment x 1 x channel x data
        #   label: trial x segment
        self.arg['input_shape'] = data[0, 0, 0, :, :].shape
        # print('data:' + str(data.shape) + ' label:' + str(label.shape))
        return data, label
    
    # Load the data subject by subject  
    def load_data_per_subject_mat(self, sub):
        #file = 'CDsub' + str(sub) + '.mat'
        # file = 'preprocessedNRMAT' + '.mat'
        # file = 'CORRECTEDOPENCLOSEHR20CH' + str(sub) + '.mat'# testHRsub resttaskHR44CH_9Bandsub
        if sub < 9:
            file = 'sub' + '0'+ str(sub) + '.mat'
        if sub > 9:
            file = 'sub'+ str(sub) + '.mat'    
        
        subject_path = os.path.join(self.arg.get('data_dir'), file)        

        # print('loading data from subject ' + str(sub+1) + ' ..')

        data = loadmat(subject_path)['data']
        
        rlabel = np.array(data['y'][0][0],dtype='int64').T #np.array(subject['label'])
        rdata =  np.array(np.expand_dims(np.moveaxis(data['x'][0][0],[2,0,1],[0,1,2]), axis=(1,2)),dtype='float32') #np.array(subject['data'])
        #rdata = rdata[:, :,:, :, -6800:] NEDED FOR MUSE 2
        #   data: trial x segment x 1 x channel x data
        #   label: trial x segment
        self.arg['input_shape'] = rdata[0, 0, 0, :, :].shape
        # print('data:' + str(rdata.shape) + ' label:' + str(rlabel.shape))
        
        #rdata = band_pass(rdata, [4, 40, 2], self.arg['sampling_rate'], True, 1, False, True)
        # rdata = bandpassfilter_cheby2_sos(rdata, [0.3, 4], self.arg['sampling_rate'], [0.2, 5], 2)

        
        seg_data = []
        seg_label = []
        for t in range(rdata.shape[0]):
            x, y = self.split_w_overlap_muse(rdata[t], rlabel[t], self.arg['window']*self.arg['sampling_rate'], self.arg['shift']*self.arg['sampling_rate'])
            seg_data.append(x[None])
            seg_label.append(y[None])
        rdata = np.vstack(seg_data)
        rlabel = np.vstack(seg_label)
        
        # return self.normalize_trial(rdata), rlabel    
        return rdata, rlabel      
    
    def load_data_per_subject_eeg(self,sub):
        filename = 'SUB-' + str(sub+1) + '/sub0' + str(sub+1) + '.vhdr'

        ## Ximing
        print("LOADING XIMING")
        filename = 'SUB-' + str(sub) + '/pilot_calib_sess1.vhdr'
        
        print(filename)

        subject_path = os.path.join(self.arg.get('data_dir'), filename)        

        raw = mne.io.read_raw_brainvision(subject_path)
        # raw = mne.io.read_raw_eeglab(subject_path, preload=True)

        eeg, times = raw[:]  # full data
        event = mne.events_from_annotations(raw)  # event list
        fsamp = int(raw.info['sfreq'])

        ## 50 subjects
        # extype = 'hr'
        ## ximing
        extype = 'roc'

        if extype == 'hr':
            ev = eventlist_HRmi(event, fsamp)  # eventlist_HRrest1
            taskdur = 4  # task duration in seconds
            # evrest = eventlist_HRrest(event,fsamp)
            # evrest1 = eventlist_HRrest1(event,fsamp)
        elif extype == 'roc':
            ev = eventlist_ROC(event, fsamp)  # eventlist_ROC
            taskdur = 4  # task duration in seconds
        elif extype == 'motiv':
            ev = eventlist_grazmi(event, fsamp)
            taskdur = 4  # task duration in seconds
        elif extype == 'cirg':
            ev = eventlist_cirgmi(event, fsamp)
            taskdur = 4  # task duration in seconds

        rawdata = dict({'eeg': [], 'event': []})
        rawdata['eeg'] = eeg
        rawdata['event'] = event

        sampdur = int(fsamp * taskdur)
        # prepdur = int(fsamp * 0)

        data = dict({'X': [], 'y': [], 'yclass': []})
        # print(ev)
        for i, t in enumerate(ev['code']):
            if (ev['label'][i] != 'non'):
                    data['X'].append(eeg[:, ev['sampstart'][i]:ev['sampstart'][i] + sampdur])
                    # data['X'].append(eeg[:,ev['sampstart'][i]-prepdur :ev['sampstart'][i]+ sampdur])
                    data['y'].append(t)
                    data['yclass'].append(ev['label'][i])

        # print(data['y'])
        # print(np.array(data['X']).shape)
        # print(len(data['X']))
        # print(len(data['y']))
        data['X'] = np.stack(data['X'])

        data['s'] = fsamp
        data['c'] = raw.info['ch_names']
        data['rawfile'] = subject_path  # 2 is close, 1 is open, 0 is rest


        X_Data = np.array(data['X'])
        Y_Data = [np.array(data['y'])]

        count = 0
        for data_point in Y_Data[0]:
            if data_point == 2:
                Y_Data[0][count] = 1
            count += 1
        print(Y_Data)

        ### No preprocessing
        # X_Data = np.delete(X_Data, (21), axis=1)
        # X_Data = np.delete(X_Data, (10), axis=1)
        # X_Data = X_Data[:,:,::4]
        # X_Data_array = X_Data
        
        ### Preprocess
        from preprocess import preprocess_data
        preprocess_args = {
            'sampling_frequency': 1000,
            'decimate_factor': 4,
            'chan_set': 'all',
            'plot_flag': False,
            'verbose': False
        }
        preprocess_instance = preprocess_data(args=preprocess_args)
        print(X_Data.shape)
        X_Data_array = []
        for i in range(0,len(X_Data)):
            X_Data_new = preprocess_instance.process(X_Data[i])
            X_Data_new = X_Data_new[np.newaxis,:,:]
            if len(X_Data_array) == 0:
                X_Data_array = X_Data_new
            else:
                X_Data_array = np.concatenate((X_Data_array,X_Data_new),axis=0)
            # print(X_Data_array.shape)

        return X_Data_array, Y_Data

    def load_data_per_subject_mat_fbcnet_2(self, sub):
        file = 'CORRECTEDRESTOPENHR61CH' + str(sub) + '.mat'
        print(file)
        # file = 'CORRECTEDOPENCLOSEHR61CH' + str(sub) + '.mat'
        #restOPENHR60CH9bands WORKED restOPENHR60CH9bands newRESTOPENHR60CHLOWbands
        # if sub < 9:restOPENHR60CH9bands0 restOPENHR60CHLOWbands
        # file = 'sub' + '0'+ str(sub) + '.mat'
        # if sub > 9:
        # file = 'sub'+ str(sub) + '.mat'    
        subject_path = os.path.join(self.arg.get('data_dir'), file)        

        # print('loading data from subject ' + str(sub+1) + ' ..')

        data = loadmat(subject_path)['data']

        # print(data[0][0][0].shape)
        # print(data[0][0][1])

        data_sub = data[0][0][0]
        data_label = data[0][0][1]

        count = 0
        for data_point in data_label[0]:
            if data_point == 1:
                data_label[0][count] = 2
            count += 1

        data_sub = np.moveaxis(data_sub,2,0)

        return data_sub,data_label

    def load_data_per_subject_mat_fbcnet(self, sub):
        #file = 'CDsub' + str(sub) + '.mat'
        #file = 'preprocessedNRMAT' + '.mat'
        # file = 'CORRECTEDRESTOPENHR61CH' + str(sub) + '.mat'# testHRsub%resttaskHR61CH_9Bandsub  resttaskHR44CH newRESTOPENHR60CHLOWbands
        # file = 'RESTOPENHR61CH_PREPROCESSED_NTR' + str(sub) + '.mat'
        file = 'CORRECTEDRESTCLOSEHR61CH' + str(sub) + '.mat'
        print(file)
        # file = 'CORRECTEDOPENCLOSEHR61CH' + str(sub) + '.mat'
        #restOPENHR60CH9bands WORKED restOPENHR60CH9bands newRESTOPENHR60CHLOWbands
        # if sub < 9:restOPENHR60CH9bands0 restOPENHR60CHLOWbands
        # file = 'sub' + '0'+ str(sub) + '.mat'
        # if sub > 9:
        # file = 'sub'+ str(sub) + '.mat'    
        subject_path = os.path.join(self.arg.get('data_dir'), file)        

        # print('loading data from subject ' + str(sub+1) + ' ..')

        data = loadmat(subject_path)['data']

        # print(data[0][0][0].shape)
        # print(data[0][0][1])

        data_sub = data[0][0][0]
        data_label = data[0][0][1]

        data_sub = np.moveaxis(data_sub,2,0)

        return data_sub,data_label

        # import matplotlib.pyplot as plt
        # for i in range(90,100):
        #     plt.imshow(data[0][0][0][:,:,i], cmap='hot', interpolation='nearest')
        #     plt.show()
        
        rlabel = np.array(data['y'][0][0],dtype='int64').T #np.array(subject['label'])
        rdata =  np.array(np.expand_dims(np.moveaxis(data['x'][0][0],[2,1,0,3],[0,1,2,3]), axis=(1,2)),dtype='float32') #np.array(subject['data'])
        #rdata = rdata[:, :,:, :, -6800:] NEDED FOR MUSE 2
        #   data: trial x segment x 1 x channel x data x band
        #   label: trial x segment
        self.arg['input_shape'] = rdata[0, 0, 0, :, :,:].shape
        print('data:' + str(rdata.shape) + ' label:' + str(rlabel.shape))
        
        fbdata = copy.deepcopy(rdata)
        fblabel = copy.deepcopy(rlabel)
        nwin = int(self.arg['window']*self.arg['sampling_rate'])
        nseg = int((rdata.shape[-2]-self.arg['window']*self.arg['sampling_rate'])/(self.arg['shift']*self.arg['sampling_rate']) +1)
        OL=(self.arg['window']-self.arg['shift'])*self.arg['sampling_rate']
        nseg = int(np.fix(int((rdata.shape[-2]-OL)/((self.arg['window']*self.arg['sampling_rate'])-OL))))
        #fix((nsamp-noverlap)/(window-noverlap));
        fbdata = np.repeat(fbdata[:,:,:,:,:nwin,:],nseg,axis=1)
        
        for f in range(rdata.shape[-1]):       
            seg_data = []
            seg_label = []
            for t in range(rdata.shape[0]):
                x, y = self.split_w_overlap_muse(rdata[t,:,:,:,:,f], rlabel[t], 
                                                 self.arg['window']*self.arg['sampling_rate'], self.arg['shift']*self.arg['sampling_rate'])
                seg_data.append(x[None])
                seg_label.append(y[None])
        
            fbdata[:,:,:,:,:,f] = np.vstack(seg_data)
            fblabel = np.vstack(seg_label)
        
        # return self.normalize_trial(rdata), rlabel    
        return fbdata, fblabel 
    
    def split_w_overlap_muse(self,data, label, win, shift):
        sdata = []
        slabel = []
        for i in range(0, data.shape[-1]-win+1, int(shift)): 
            tmp = data[:,:,:,i:i+win]
            sdata.append(tmp)
            #if np.sum(np.sum(tmp==0,2)[0,0]!=0) > 0.02*win: # remove if any channel is zero
            if np.sum(np.sum(tmp==0,2)[0,0]==4) > 0.02*win: # only remove if all channels are zero
                slabel.append(np.array([-1.0]))
            else: 
                slabel.append(label)
            
        
        return np.vstack(sdata), np.vstack(slabel)[:,0]
    
   
    def split_balance_class(self, data, label, random):
        # Data dimention: segment x 1 x channel x data
        # Label dimention: segment x 1
        np.random.seed(0)
        # data : segments x 1 x channel x data
        # label : segments

        index_0 = np.where(label == 0)[0]
        index_1 = np.where(label == 1)[0]

        # for class 0
        index_randm_0 = copy.deepcopy(index_0)

        # for class 1
        index_randm_1 = copy.deepcopy(index_1)

        if random == True:
            np.random.shuffle(index_randm_0)
            np.random.shuffle(index_randm_1)

        index_train = np.concatenate((index_randm_0[:int(len(index_randm_0) * 0.8)],
                                      index_randm_1[:int(len(index_randm_1) * 0.8)]),
                                     axis=0)
        index_val = np.concatenate((index_randm_0[int(len(index_randm_0) * 0.8):],
                                    index_randm_1[int(len(index_randm_1) * 0.8):]),
                                   axis=0)

        # get validation
        val = data[index_val]
        val_label = label[index_val]

        train = data[index_train]
        train_label = label[index_train]

        return train, train_label, val, val_label
    
    def split_balance_class_eq(self, data, label, random):
        # Data dimention: segment x 1 x channel x data
        # Label dimention: segment x 1
        np.random.seed(0)
        # data : segments x 1 x channel x data
        # label : segments

        index_0 = np.where(label == 0)[0]
        index_1 = np.where(label == 1)[0]

        # for class 0
        index_randm_0 = copy.deepcopy(index_0)

        # for class 1
        index_randm_1 = copy.deepcopy(index_1)
        
        index_eq = min(len(index_randm_0), len(index_randm_1))
        
        # equal samples in both classes
        index_randm_0 = index_randm_0[:index_eq]
        index_randm_1 = index_randm_1[:index_eq]

        if random == True:
            np.random.shuffle(index_randm_0)
            np.random.shuffle(index_randm_1)

        index_train = np.concatenate((index_randm_0[:int(len(index_randm_0) * 0.8)],
                                      index_randm_1[:int(len(index_randm_1) * 0.8)]),
                                     axis=0)
        index_val = np.concatenate((index_randm_0[int(len(index_randm_0) * 0.8):],
                                    index_randm_1[int(len(index_randm_1) * 0.8):]),
                                   axis=0)

        # get validation
        val = data[index_val]
        val_label = label[index_val]

        train = data[index_train]
        train_label = label[index_train]

        return train, train_label, val, val_label

    def split_balance_class2(self, data, label, random):
        # Data dimention: segment x 1 x channel x data
        # Label dimention: segment x 1
        np.random.seed(0)
        # data : segments x 1 x channel x data
        # label : segments

        index_0 = np.where(label == 0)[0]
        index_1 = np.where(label == 1)[0]

        # for class 0
        index_randm_0 = copy.deepcopy(index_0)

        # for class 1
        index_randm_1 = copy.deepcopy(index_1)

        if random == True:
            np.random.shuffle(index_randm_0)
            np.random.shuffle(index_randm_1)

        index_train = np.concatenate((index_randm_0[:],
                                      index_randm_1[:]),
                                     axis=0)

        # get validation

        train = data[index_train]
        train_label = label[index_train]

        return train, train_label

    def normalize(self, train, test):
        # data: sample x 1 x channel x data
        mean = 0
        std = 0

        self.arg['input_shape'] = train[0, 0, :, :].shape

        print('train shape: ' + str(train.shape))
        print('test shape: ' + str(test.shape))
        print('input_shape: ' + str(self.arg.get('input_shape')))

        for channel in range(self.arg.get('input_shape')[0]):
            mean = np.mean(train[:, :, channel, :])
            std = np.std(train[:, :, channel, :])
            train[:, :, channel, :] = (train[:, :, channel, :] - mean) / (std)
            test[:, :, channel, :] = (test[:, :, channel, :] - mean) / (std)
        return train, test

    def normalize_trial(self, data):
        # data: trial x segment x  1 x channel x time
        
        mean = np.tile(np.mean(data[:,:,:,:,:,None], axis = 4),(1,1,1,1,data.shape[4]))
        std = np.tile(np.std(data[:,:,:,:,:,None], axis = 4),(1,1,1,1,data.shape[4]))
        
        norm_data = (data - mean)/(std)       
        
        return norm_data    

    def cross_validation(self):
        # do cross-validation subject by subject
        record_file = self.arg.get('record_file')
        fold = self.arg.get('fold')
        ACC = []
        
        # start loop
        for sub in self.arg.get('test_subject'): #sub in range(int(self.arg.get('testsub'))): #
        
            # datapath = self.arg.get('data_dir') + '/sub' + str(sub) + '.mat'
            
            #data, label = self.load_data_per_subject_mat_fbcnet(sub) 
            data, label = self.load_data_per_subject_mat(sub)
            
            # datapath = 'D:/OneDrive - Nanyang Technological University/bci_datasets/KoreaMI/s2.mat'
            # data, label = self.load_data_per_subject_ku(datapath, 4, np.array(range(62))) 
            
            # # data:  trial x segment x 1 x channels x data
            # # label:  trial x segment x 1
            if self.arg.get('model')=='FBCNet':
                self.arg['input_shape'] = data[0, 0, 0, :, :,:].shape
            else:            
                self.arg['input_shape'] = data[0, 0, 0, :, :].shape
                
            file = open(record_file, 'a')
            file.write('Subject: ' + str(sub) + '\nData size: ' + str(data.shape) + 
                       '\nLabel size: ' + str(label.shape) + '\n')
            file.close()
            
            acc = self.n_fold(fold, data, label, sub, record_file)
    
            ACC.append(acc)
            file = open(record_file, 'a')
            file.write('Subject' + str(sub) + ' mAcc:' + str(acc) + '\n')
            file.close()
        
        # end loop
        
        ACC = np.stack(ACC, axis=0)
        mAcc = np.mean(ACC)
        std = np.std(ACC)
        file = open(record_file, 'a')
        file.write("\n>>>Final mAcc:" + str(mAcc) + ' std:' + str(std) + ' <<<\n')
        file.close()
        print('-->>>-- Final mAcc:{} std:{}'.format(mAcc, std))
        
        file_save(record_file)

    def n_fold(self, n, data, label, sub_code, record_file):
        print('Using {}-fold cross-validation...'.format(n))
        #assert self.cv == str(n) + '-fold', 'CV type is wrong!' --> doesnt matter
        '''
        This is the function to achieve 10-fold cross-validation
        Each fold contains 6 trials 
        
        Note : all the acc and std will be logged into the n_fold_result.txt
               
               The txt file is located at the same location as the python script
        
        '''
        tm = TrainModel(self.arg)

        save_path = Path(os.getcwd())
        if not os.path.exists(save_path / Path('N_fold/history')):
            os.makedirs(save_path / Path('N_fold/history'))
        # Data dimension: trials x segments x 1 x channel x data
        # Label dimension: trials x segments
        shape_data = data.shape
        trial = shape_data[0]
        fold = n
        num_trial = int(trial / fold)
        # Train and evaluate the model subject by subject
        ACC = []
        # sindex = np.array(range(0,trial,fold)) 
        
        index = np.arange(trial)
        np.random.seed(0)
        np.random.shuffle(index)
        
        nseg = data.shape[1]
        fraction = 0.2

        for j in range(fold):
            # Split the data into training set and test set
            # One session(contains 2 trials) is test set
            # The rest are training set
            index_test = index[num_trial * j:num_trial * (j + 1)]
            # index_train = np.delete(index, index_test)
            index_train = np.delete(index, np.asarray(range(num_trial * j,num_trial * (j + 1))))
            
            index_val = index_train[:num_trial]
            index_train = index_train[num_trial:]
            
            print(index_test,index_val,index_train)

            # data_train = data[index_train, int(nseg*fraction):, :, :, :]
            # label_train = label[index_train, int(nseg*fraction):]
            
            # data_val = data[index_train, :int(nseg*fraction), :, :, :]
            # label_val = label[index_train, :int(nseg*fraction)]

            data_train = data[index_train, :, :, :, :]
            label_train = label[index_train,:]
            
            data_val = data[index_val, :, :, :, :]
            label_val = label[index_val, :]

            data_test = data[index_test, :, :, :, :]
            label_test = label[index_test, :]            
            
            # # training ->segment x 1 x channel x data
            data_train = np.concatenate(data_train, axis=0)
            label_train = np.concatenate(label_train, axis=0)
            
            data_val = np.concatenate(data_val, axis=0)
            label_val = np.concatenate(label_val, axis=0)

            data_test = np.concatenate(data_test, axis=0)
            label_test = np.concatenate(label_test, axis=0)                     
            
            # data_train,label_train = remove_empty(data_train,label_train,record_file, 'TR')
            # data_val, label_val = remove_empty(data_val, label_val,record_file, 'VA')
            # data_test,label_test = remove_empty(data_test,label_test,record_file, 'EV')
            
            # data_train, label_train, data_val, label_val = self.split_balance_class1(data_train, label_train, True)
            
            # Split the training set into training set and validation set
            # data_train, label_train, data_val, label_val = self.split_balance_class_eq(data_train, label_train, True)
        
            
            # Prepare the data format for training the model
            data_train = torch.from_numpy(data_train).float()
            label_train = torch.from_numpy(label_train).long()

            data_val = torch.from_numpy(data_val).float()
            label_val = torch.from_numpy(label_val).long()

            data_test = torch.from_numpy(data_test).float()
            label_test = torch.from_numpy(label_test).long()

            # Check the dimension of the training, validation and test set
            print('Training:', data_train.size(), label_train.size())
            print('Validation:', data_val.size(), label_val.size())
            print('Test:', data_test.size(), label_test.size())

            # Get the accuracy of the model
            ACC_fold = tm.get_result(train_data=data_train,
                                     train_label=label_train,
                                     test_data=data_test,
                                     test_label=label_test,
                                     val_data=data_val,
                                     val_label=label_val,
                                     subject=sub_code,
                                     fold=j,
                                     cv_type="n-fold",
                                     model_name=self.arg.get('model_name'))

            ACC.append(ACC_fold)
            
            # Log the results per fold

            file = open(record_file,'a')
            file.write('Subject:'+ str(sub_code) +' fold:'+ str(j) + ' ACC:' + str(ACC_fold) + '\n')
            file.close()
            
        ACC = np.array(ACC)
        mAcc = np.mean(ACC)
        std = np.std(ACC)

        print("Subject:{} -- mAcc: {:4.2f} std: {:4.2f}".format(str(sub_code), mAcc, std))
        calculate_statistics(tm.y_true, tm.y_pred, tm.y_hat, record_file)
        
        # import csv
        # f=open("people.csv","a",newline="")
        # tup1=mAcc
        # writer=csv.writer(f) 
        # writer.writerow(tup1)
        # f.close;
        
        # file = open('result_sub{}_model{}_{}.txt'.format(subject, self.arg.get('model'), date_time), 'a')
        # file.write(' MeanACC:' + str(mAcc) + ' Std:' + str(std) + '\n') 

        return mAcc

    def leave_one_subject_out(self):
        """
        leave one subject out
        """
        date_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        record_file = self.arg.get('record_file')
        assert self.cv == 'leave_one_subject_out', 'CV type is wrong!'
        save_path = Path(os.getcwd())
        if self.arg.get('model') == 'EEGNet':
            print('------------Training model for EEGNet--------------')
            if not os.path.exists(save_path / Path('Leave_one_subject_out_EEG/history')):
                os.makedirs(save_path / Path('Leave_one_subject_out_EEG/history'))
            data_file = os.path.join(save_path, 'Leave_one_subject_out_EEG', 'data.npy')
        elif self.arg.get('model') == 'TSception_Revised':
            print('------------Training model for TSception--------------')
            if not os.path.exists(save_path / Path('Leave_one_subject_out_TS3/history')):
                os.makedirs(save_path / Path('Leave_one_subject_out_TS3/history'))
            data_file = os.path.join(save_path, 'Leave_one_subject_out_TS3', 'data.npy')
        else:
            if not os.path.exists(save_path / Path('Leave_one_subject_out_EXP/history')):
                os.makedirs(save_path / Path('Leave_one_subject_out_EXP/history'))
            data_file = os.path.join(save_path, 'Leave_one_subject_out_EXP', 'data.npy')
        data = []
        label = []
        for _file in self.arg['test_subject']:
            # for _file in np.arange(self.arg.get('num_sub')):
            _data, _label = self.load_data_per_subject_mat(_file)
                # _data: trial, seg, 1, chan, sample
            data.append(_data), label.append(_label)
        #shape_data = data.shape
        #shape_label = label.shape
        subject = len(data)
        channel = data[0].shape[3]
        frequency = data[0].shape[2]

        print(
            "Train:Leave_one_subject_out \n1)shape of data:" + str('NA') + " \n2)shape of label:" + str(
                'NA') +
            " \n3)trials:" + str('NA') + " \n4)frequency:" + str(frequency) + " \n5)channel:" + str(channel))
        ACC_subjects = []
        ACC_trial = []

        for j in range(subject):
        # for j in range(subject): # (subject) don't forget to add back  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            index = np.arange(subject)
            #index = np.array(self.arg['test_subject']) #np.arange(subject)
            index_train = np.delete(index, j)
            # Split the data into training set and test set
            # One subject is testing set
            # The rest are training set
            #index_train = np.delete(index, j)
            index_test = j
            data_train = [np.concatenate(data[i]) for i in index_train]   # a list of (seg, 1, chan, sample)
            data_train = np.concatenate(data_train) # (seg, 1, chan, sample)
            label_train = [np.concatenate(label[i]) for i in index_train]
            label_train = np.concatenate(label_train)
            data_test = data[index_test]  # (trial, seg, 1, chan, sample)
            label_test = label[index_test]  # (trial, seg)

            # train data->  segment x 1 x channel x data
            #data_train = np.concatenate(data_train)
            #label_train = label_train.flatten()
            print('train and label: '+str(data_train.shape)+str(label_train.shape))

            data_test = np.concatenate(data_test)  # (seg, 1, chan, sample)
            label_test = np.concatenate(label_test)
            print('test and label: '+str(data_test.shape)+str(label_test.shape))


            # normalize data
            data_train, data_test = self.normalize(data_train, data_test)
            print('after norm: ======= train: '+str(data_train.shape)+' test: '+str(data_test.shape))

            if self.arg.get('patient_es') > 0:
                # Split the training set into training set and validation set
                data_train, label_train, data_val, label_val = self.split_balance_class(data_train, label_train, True)
                print('---------------------> train data: '+str(data_train.shape)+' train label: '+ str(label_train.shape))
                #data_test, label_test = self.split_balance_class2(data_test, label_test, True)
                print('---------------------> test data: '+str(data_test.shape)+' test label: '+ str(label_test.shape))
            else:
                # patient_es == 0 means not using early-stopping, hence the validation set is not needed
                # we train n epochs and test anyway, hence making the data_val = data_train makes NO difference
                # to the final classification
                data_val = data_train
                label_val = label_train


            data_train,label_train = remove_empty(data_train,label_train,record_file, 'TR')
            data_val, label_val = remove_empty(data_val, label_val,record_file, 'VA')
            data_test,label_test = remove_empty(data_test,label_test,record_file, 'EV')
            
            
            # Prepare the data format for training the model
            data_train = torch.from_numpy(data_train).float()
            label_train = torch.from_numpy(label_train).long()

            data_val = torch.from_numpy(data_val).float()
            label_val = torch.from_numpy(label_val).long()

            data_test = torch.from_numpy(data_test).float()
            label_test = torch.from_numpy(label_test).long()

            # Check the dimention of the training, validation and test set
            print('Training:', data_train.size(), label_train.size())
            print('Validation:', data_val.size(), label_val.size())
            print('Test:', data_test.size(), label_test.size())

            # Instantiate model
            tm = TrainModel(self.arg)

            # Get the accuracy of the model
            ACC_subject = tm.get_result(train_data=data_train,
                                        train_label=label_train,
                                        test_data=data_test,
                                        test_label=label_test,
                                        val_data=data_val,
                                        val_label=label_val,
                                        subject=j,
                                        fold=0,
                                        cv_type="leave_one_subject_out",
                                        model_name=self.arg.get('model'))

            # ACC_day: [acc_day2,acc_day3,...]

            ACC_subjects.append(ACC_subject)

            # Log the results per day
            file = open('result_sub{}_model{}_{}.txt'.format(subject, self.arg.get('model'), date_time), 'a')
            file.write("Leave subject %d out acc: %f"%(j, ACC_subject) + '\n')
            file.close()

        ACC_subjects = np.array(ACC_subjects)
        mAcc = np.mean(ACC_subjects)
        std = np.std(ACC_subjects)
        print('------Results------')
        print("\n mACC: %.2f" % mAcc)
        print("std: %.2f" % std)
        print('-------------------')

        # Log the results per subject
        file = open('result_sub{}_model{}_{}.txt'.format(subject, self.arg.get('model'), date_time), 'a')
        file.write(' MeanACC:' + str(mAcc) + ' Std:' + str(std) + '\n')   
        file.close()
        calculate_statistics(tm.y_true, tm.y_pred, tm.y_hat, file.name)
        file.close()


       

    '''
    many other cross validation methods to be added...
    
    '''