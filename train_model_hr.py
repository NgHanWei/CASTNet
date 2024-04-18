# This is the script to train and evaluate the model
# Modified by Ding
# 3-Jun-2020
import torch
import time
import numpy as np
import h5py
import datetime
import os
import torch.nn as nn

from pathlib import Path
from eeg_dataset import *
from torch.utils.data import DataLoader
from networks import *
#from networks_fbcnet import *
# from deepCNN import *

# Acknowledgement:
# Thanks to this tutorial:
# [https://colab.research.google.com/github/dvgodoy/PyTorch101_ODSC_London2019/blob/master/PyTorch101_Colab.ipynb]
class TrainModel():
    def __init__(self, arg):
        self.data = None
        self.label = None
        self.result = None
        self.input_shape = arg.get('input_shape')  # should be (eeg_channel, time data point)
        self.model = arg.get('model')
        self.sampling_rate = arg.get('sampling_rate')

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Parameters: Training process
        self.random_seed = 42 if arg.get('random_seed') == None else arg.get('random_seed')
        self.learning_rate = 1e-3 if arg.get('learning_rate') == None else arg.get('learning_rate')
        self.num_epochs = 200 if arg.get('num_epochs') == None else arg.get('num_epochs')
        self.num_class = 2 if arg.get('num_class') == None else arg.get('num_class')
        self.batch_size = 128 if arg.get('batch_size') == None else arg.get('batch_size')
        self.patient = 4 if arg.get('patient_es') == None else arg.get('patient_es')

        # Parameters: Model
        self.dropout = 0.3 if arg.get('dropout') == None else arg.get('dropout')
        self.hiden_node = 128 if arg.get('hiden_node') == None else arg.get('hiden_node')
        self.T = 15 if arg.get('num_T') == None else arg.get('num_T')
        self.S = 15 if arg.get('num_S') == None else arg.get('num_S')
        self.Lambda = 1e-6 if arg.get('L1_lambda') == None else arg.get('L1_lambda')

        self.y_pred = None
        self.y_true = None
        self.y_hat = None

    def make_train_step(self, model, loss_fn, optimizer):
        def train_step(x, y):
            model.train()
            yhat = model(x)
            pred = yhat.max(1)[1]
            correct = (pred == y).sum()
            acc = correct.item() / len(pred)
            # L1 regularization
            loss_r = self.regulization(model, self.Lambda)
            # yhat is in one-hot representation;
            loss = loss_fn(yhat, y) + loss_r
            # loss = loss_fn(yhat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            return loss.item(), acc

        return train_step

    def regulization(self, model, Lambda):
        w = torch.cat([x.view(-1) for x in model.parameters()])
        err = Lambda * torch.sum(torch.abs(w))
        return err

    def get_result(self, train_data, train_label, test_data, test_label, val_data,
                            val_label, subject, fold, cv_type,model_name):
        # print('Available device:' + str(torch.cuda.get_device_name(torch.cuda.current_device())))
        torch.manual_seed(self.random_seed)
        torch.backends.cudnn.deterministic = True

        print('->- Eval Subject:{0:}'.format(subject))
        print('->- CV fold:{0:}/n'.format(str(fold+1)))
        # Train and validation loss
        losses = []
        accs = []

        Acc_val = []
        Loss_val = []
        val_losses = []
        val_acc = []

        test_losses = []
        test_acc = []
        Acc_test = []

        # hyper-parameter
        learning_rate = self.learning_rate
        num_epochs = self.num_epochs

        print(f"Input shape ->->->->->->->->->-> index 0: {self.input_shape[0]}, index 1: {self.input_shape[1]}")
        print(f"Sampling rate: {self.sampling_rate}")

        # build the model
        if self.model == 'TSception':
            model = TSception(num_classes=self.num_class, input_size=self.input_shape,
                              sampling_rate=self.sampling_rate, num_T=self.T, num_S=self.S,
                              hiden=self.hiden_node, dropout_rate=self.dropout)
        elif self.model == 'TSception_Revised':
            print('------------running model for revised TSception---------------')
            model = TSception_Revised(num_classes=self.num_class, input_size=self.input_shape,
                              sampling_rate=self.sampling_rate, num_T=self.T, num_S=self.S,
                              hiden=self.hiden_node, dropout_rate=self.dropout)
        elif self.model == 'EEGNet':
            print('------------running model for EEG Net---------------')
            model = EEGNet(channels=self.input_shape[0], samples=self.input_shape[1],
                           n_classes=self.num_class)
        elif self.model == 'DeepConvNet':
            model = deepConvNet(nChan=self.input_shape[0], nTime=self.input_shape[1],
                                nClass=self.num_class)#deepConvNet_kavi
        elif self.model == 'DeepCNN':
            model = SRDeep4Net(in_chans=self.input_shape[0], n_classes = self.num_class,
                               input_time_length=self.input_shape[1],final_conv_length='auto')
        elif self.model == 'FBCNet':
             model = FBCNet(nChan=self.input_shape[0],nTime=self.input_shape[1],nClass = self.num_class,nBands=self.input_shape[2])    
        # you can add your model here:
        elif self.model == "": # your model name
            pass

        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('->- Parameters:{}'.format(pytorch_total_params))

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        loss_fn = nn.CrossEntropyLoss()

        if torch.cuda.is_available():
            model = model.to(self.device)
            loss_fn = loss_fn.to(self.device)

        train_step = self.make_train_step(model, loss_fn, optimizer)

        # load the data 
        dataset_train = EEGDataset(train_data, train_label)
        dataset_test = EEGDataset(test_data, test_label)
        dataset_val = EEGDataset(val_data, val_label)

        # Dataloader for training process
        train_loader = DataLoader(dataset=dataset_train, batch_size=self.batch_size, shuffle=True, pin_memory=True)

        val_loader = DataLoader(dataset=dataset_val, batch_size=self.batch_size, pin_memory= True)

        test_loader = DataLoader(dataset=dataset_test, batch_size=self.batch_size, pin_memory=True)

        total_step = len(train_loader)

        ######## Training process ########
        Acc = []
        acc_max = 0
        patient = 0

        for epoch in range(num_epochs):
            loss_epoch = []
            acc_epoch = []
            for i, (x_batch, y_batch) in enumerate(train_loader):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                loss, acc = train_step(x_batch, y_batch)
                loss_epoch.append(loss)
                acc_epoch.append(acc)

            losses.append(sum(loss_epoch) / len(loss_epoch))
            accs.append(sum(acc_epoch) / len(acc_epoch))
            loss_epoch = []
            acc_epoch = []
            print('Epoch [{}/{}], Loss: {:.4f}, Acc: {:.4f}'
                  .format(epoch + 1, num_epochs, losses[-1], accs[-1]))

            ######## Validation process ########
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val = x_val.to(self.device)
                    y_val = y_val.to(self.device)

                    model.eval()

                    yhat = model(x_val)
                    pred = yhat.max(1)[1]
                    correct = (pred == y_val).sum()
                    acc = correct.item() / len(pred)
                    val_loss = loss_fn(yhat, y_val)
                    val_losses.append(val_loss.item())
                    val_acc.append(acc)

                Acc_val.append(sum(val_acc) / len(val_acc))
                Loss_val.append(sum(val_losses) / len(val_losses))
                print('Evaluation Loss:{:.4f}, Acc: {:.4f}'
                      .format(Loss_val[-1], Acc_val[-1]))
                val_losses = []
                val_acc = []

            ######## early stop ########
            if self.patient == 0:
                pass
            else:

                Acc_es = Acc_val[-1]

                if Acc_es > acc_max:
                    acc_max = Acc_es
                    patient = 0
                    print('----Model saved!----')
                    torch.save(model, model_name + '_max_model.pt')
                else:
                    patient += 1
                if patient > self.patient:
                    print('----Early stopping----')
                    break

        ######## test process ########
        if self.patient == 0:
            pass
        else:
            model = torch.load(model_name + '_max_model.pt')
            print(model)
        with torch.no_grad():
            for x_test, y_test in test_loader:
                x_test = x_test.to(self.device)
                y_test = y_test.to(self.device)

                model.eval()

                yhat = model(x_test)
                pred = yhat.max(1)[1]
                #print('yhat: '+str(yhat.max(1)[1]))
                correct = (pred == y_test).sum()
                acc = correct.item() / len(pred)
                test_loss = loss_fn(yhat, y_test)
                test_losses.append(test_loss.item())
                test_acc.append(acc)
                self.y_true = y_test.data.cpu().numpy() if self.y_true is None else np.concatenate(
                    [self.y_true, y_test.data.cpu().numpy()])
                self.y_hat = yhat.data.cpu().numpy()[:, 1] if self.y_hat is None else np.concatenate(
                    [self.y_hat, yhat.data.cpu().numpy()[:, 1]])
                self.y_pred = pred.data.cpu().numpy() if self.y_pred is None else np.concatenate(
                    [self.y_pred, pred.data.cpu().numpy()])

            print('Test Loss:{:.4f}, Acc: {:.4f}'
                  .format(sum(test_losses) / len(test_losses), sum(test_acc) / len(test_acc)))
            Acc_test = (sum(test_acc) / len(test_acc))
            test_losses = []
            test_acc = []

        # save the loss(acc) for plotting the loss(acc) curve
        save_path = Path(os.getcwd())
        if cv_type == "10-fold" or cv_type == "n-fold":
            filename_callback = save_path / Path('N_fold/history/'
                                                 + 'history_subject_' + str(subject) +
                                                 '_fold_' + str(fold) + '_history.hdf')
            save_history = h5py.File(filename_callback, 'w')
            save_history['acc'] = accs
            save_history['val_acc'] = Acc_val
            save_history['loss'] = losses
            save_history['val_loss'] = Loss_val
            save_history.close()
        
        elif cv_type == "leave_one_subject_out":
            filename_callback = save_path / Path('Leave_one_subject_out_EXP/history/'
                                                 + 'leave_subject_' + str(subject)
                                                 + '_history.hdf')
            save_history = h5py.File(filename_callback, 'w')
            save_history['acc'] = accs
            save_history['val_acc'] = Acc_val
            save_history['loss'] = losses
            save_history['val_loss'] = Loss_val
            save_history.close()

        return Acc_test
