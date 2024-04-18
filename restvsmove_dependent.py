from cross_validation_rvm import CrossValidation
import numpy as np
from sklearn.utils import shuffle
from tqdm import tqdm

from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from random import randint
import os
import matplotlib.pyplot as plt
from plotly.graph_objs import *
import plotly
import random
import torch
import mne

args = {
        "mode": 'train',
        # "data_dir": "/home/hanwei/HRehab/HR_DATA_REST_OC_1/" ,
        # "data_dir": "/home/hanwei/HRehab/CORRECTED_openclose/" ,
        ## XIMING
        "data_dir": "/home/hanwei/HRehab/FINAL/" ,
        "acc": None,
        "std": None,
        "results": None,
 
        "channel_list": 'muse',
        "num_class": 2,
        "sampling_rate": 250,
 
        "cross_validation": 'leave_one_subject_out',# n-fold
        "fold": 5,
        "num_sub": 12,
        "test_subject": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49],#1	2	3	4	5	6	7	8	9	10	11	12	13	14	15	16	17	18	19	20	21	22	23	24	25	26	28	29	30	31	33	34	35	36	37	38	39	40	41	42	43	44	45	46
        "segmentation": True,
        "window": 4,
        "shift": 0.2,
        "normalize": True,
 
        "random_seed": 42,
        "learning_rate": 1e-3,
        "num_epochs": 100,
        "batch_size": 40,
        "step_size": 10,
        "patient_es": 0,
        "dropout": 0.5,
 
        "model": 'DeepConvNet',
        "model_path": None,
        "data_path": None,
        "model_name": "testing_model"
    }

CV = CrossValidation(arg=args)

def plot_clustering(z_run, labels, engine ='plotly', download = False, folder_name ='clustering'):
    """
    Given latent variables for all timeseries, and output of k-means, run PCA and tSNE on latent vectors and color the points using cluster_labels.
    :param z_run: Latent vectors for all input tensors
    :param labels: Cluster labels for all input tensors
    :param engine: plotly/matplotlib
    :param download: If true, it will download plots in `folder_name`
    :param folder_name: Download folder to dump plots
    :return:
    """
    def plot_clustering_plotly(z_run, labels):

        labels = labels[:z_run.shape[0]]  # because of weird batch_size

        hex_colors = []
        for _ in np.unique(labels):
            hex_colors.append('#%06X' % randint(0, 0xFFFFFF))

        colors = [hex_colors[int(i)] for i in labels]

        z_run_pca = TruncatedSVD(n_components=3).fit_transform(z_run)
        z_run_tsne = TSNE(perplexity=8, min_grad_norm=1E-12, n_iter=3000).fit_transform(z_run)

        trace = Scatter(
            x=z_run_pca[:, 0],
            y=z_run_pca[:, 1],
            mode='markers',
            marker=dict(color=colors)
        )
        data = Data([trace])
        layout = Layout(
            title='PCA on z_run',
            showlegend=True
        )
        fig = Figure(data=data, layout=layout)
        plotly.offline.iplot(fig)

        trace = Scatter(
            x=z_run_tsne[:, 0],
            y=z_run_tsne[:, 1],
            mode='markers',
            marker=dict(color=colors)
        )
        data = Data([trace])
        layout = Layout(
            title='tSNE on z_run',
            showlegend=True
        )
        fig = Figure(data=data, layout=layout)
        plotly.offline.iplot(fig)

    def plot_clustering_matplotlib(z_run, labels, download, folder_name):

        labels = labels[:z_run.shape[0]] # because of weird batch_size

        hex_colors = []
        for _ in np.unique(labels):
            hex_colors.append('#%06X' % randint(0, 0xFFFFFF))

        if len(hex_colors) == 2:
            hex_colors = ['r','g']
        else:
            hex_colors = ['r','g','b']
        colors = [hex_colors[int(i)] for i in labels]

        z_run_pca = TruncatedSVD(n_components=3).fit_transform(z_run)
        z_run_tsne = TSNE(perplexity=80, min_grad_norm=1E-12, n_iter=3000).fit_transform(z_run)

        plt.scatter(z_run_pca[:, 0], z_run_pca[:, 1], c=colors, marker='*', linewidths=0)
        plt.title('PCA on z_run')
        plt.legend()
        if download:
            if os.path.exists(folder_name):
                pass
            else:
                os.mkdir(folder_name)
            plt.savefig(folder_name + "/pca.png")
        else:
            plt.show()

        plt.scatter(z_run_tsne[:, 0], z_run_tsne[:, 1], c=colors, marker='*', linewidths=0)
        plt.title('tSNE on z_run')
        plt.legend()
        if download:
            if os.path.exists(folder_name):
                pass
            else:
                os.mkdir(folder_name)
            plt.savefig(folder_name + "/tsne.png")
        else:
            plt.show()

    if (download == False) & (engine == 'plotly'):
        plot_clustering_plotly(z_run, labels)
    if (download) & (engine == 'plotly'):
        print("Can't download plotly plots")
    if engine == 'matplotlib':
        plot_clustering_matplotlib(z_run, labels, download, folder_name)

def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

## Subject Independent
def subject_independent_data(subj):
    all_data = []
    all_onehot_labels = []
    data = []
    onehot_labels = []
    for i in range (0,50):
        if i == subj:
            # target_data, target_labels = CV.load_data_per_subject_eeg(sub=subj)
            target_data, target_labels = CV.load_data_per_subject_mat_fbcnet(sub=subj)
            target_data = target_data[:,np.newaxis,:,:]
            target_onehot_labels = one_hot(target_labels[0],2)
            target_data,target_onehot_labels = shuffle(target_data,target_onehot_labels,random_state=0)
            # target_data = target_data[:len(target_data)-40]
            # target_onehot_labels = target_onehot_labels[:len(target_onehot_labels)-40]
            # print(target_data.shape)
            # print(target_onehot_labels.shape)
        else:
            # data,labels = CV.load_data_per_subject_eeg(sub=i)
            data, labels = CV.load_data_per_subject_mat_fbcnet(sub=subj)

            data = data[:,np.newaxis,:,:]
            onehot_labels = one_hot(labels[0],2)

            if len(all_data) == 0:
                all_data = data
            else:
                all_data = np.concatenate((all_data, data), axis=0)
            
            if len(all_onehot_labels) == 0:
                all_onehot_labels = onehot_labels
            else:
                all_onehot_labels = np.concatenate((all_onehot_labels, onehot_labels), axis=0)

            data,onehot_labels = shuffle(all_data,all_onehot_labels,random_state=0)

            # data_valid = data[len(data)-100:]
            # onehot_labels_valid = onehot_labels[len(onehot_labels)-100:]

            # data_train = data[:len(data)-100]
            # onehot_labels_train = onehot_labels[:len(onehot_labels)-100]

    # return all_data,all_onehot_labels

    return data,onehot_labels,target_data,target_onehot_labels

## Normalizing helper function
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

## Subject Dependent
def get_data_openclose(sub):
    # data,labels = CV.load_data_per_subject_mat_fbcnet(sub)

    # data_2,labels_2 = CV.load_data_per_subject_mat_fbcnet_2(sub)

    # cutoff = 0
    # for i in range(0,len(labels[0])):
    #     if labels[0][i] == labels[0][-1]:
    #         cutoff = i
    #         break
    # data_first = data[cutoff:]
    # label_first = labels[0][cutoff:]

    # cutoff = 0
    # for i in range(0,len(labels_2[0])):
    #     if labels_2[0][i] == labels_2[0][-1]:
    #         cutoff = i
    #         break
    # data_sec = data_2[cutoff:]
    # label_sec = labels_2[0][cutoff:]

    # labels_total = [np.concatenate((label_first,label_sec),axis=0)]
    # print(len(labels_total[0]))
    # data_total = np.concatenate((data_first,data_sec),axis=0)
    # print(data_total.shape)

    # data = data_total
    # onehot_labels = labels_total

    # XIMING / Raw Data
    data,labels = CV.load_data_per_subject_eeg(sub)

    data = data[:,np.newaxis,:,:]
    onehot_labels = one_hot(labels[0],2)

    # for i in range(0,len(data)):
    #     data[i] = NormalizeData(data[i])

    # from sklearn.preprocessing import StandardScaler
    # for i in range(0,len(data)):
    #     # data[i] = [-1 + 2 * (x - min(data[i])) / (max(data[i]) - min(data[i])) for x in data[i]]
    #     data[i] = StandardScaler().fit_transform(data[i][0])

    ### Shuffling
    # data,onehot_labels = shuffle(data,onehot_labels,random_state=0)

    data_valid = data[len(data)-30:]
    onehot_labels_valid = onehot_labels[len(onehot_labels)-30:]
    # data_valid = data
    # onehot_labels_valid = onehot_labels

    data_train = data[:len(data)-0]
    onehot_labels_train = onehot_labels[:len(onehot_labels)-0]
    
    return data_train,onehot_labels_train,data_valid,onehot_labels_valid


# data,onehot_labels = shuffle(all_data,all_onehot_labels,random_state=0)
# print(data,onehot_labels.shape)

# data_valid = data[len(all_data)-20:]
# onehot_labels_valid = onehot_labels[len(all_data)-20:]

# data = data[:len(all_data)-20]
# onehot_labels = onehot_labels[:len(all_data)-20]

import os
import time
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy, binary_crossentropy, MeanSquaredError
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics import cohen_kappa_score

import models
from preprocess import get_data
from vae import cut_data

def draw_learning_curves(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'val'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'val'], loc='upper left')
    plt.show()
    plt.close()

def draw_confusion_matrix(cf_matrix, sub, results_path):
    # Generate confusion matrix plot
    display_labels = ['Open','Close']
    disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, 
                                display_labels=display_labels)
    disp.plot()
    disp.ax_.set_xticklabels(display_labels, rotation=12)
    plt.title('Confusion Matrix of Subject: ' + sub )
    plt.savefig(results_path + '/subject_' + sub + '.png')
    plt.show()

def draw_performance_barChart(num_sub, metric, label):
    fig, ax = plt.subplots()
    x = list(range(1, num_sub+1))
    ax.bar(x, metric, 0.5, label=label)
    ax.set_ylabel(label)
    ax.set_xlabel("Subject")
    ax.set_xticks(x)
    ax.set_title('Model '+ label + ' per subject')
    ax.set_ylim([0,1])

def choose_data(labels,y_pred,model,X_valid,onehot_labels_valid,conf_threshold):
    new_labels = []
    new_data = []
    unchosen_data = []
    unchosen_labels = []
    for label_num in tqdm(range(0,len(labels))):
        if labels[label_num] == y_pred[label_num]:

            confidence_output = model.predict(np.array([X_valid[label_num]]), verbose = 0)

            if max(confidence_output[0]) > conf_threshold:

                if len(new_labels) == 0:
                    new_labels = np.array([onehot_labels_valid[label_num]])
                    new_data = np.array([X_valid[label_num]])
                else:
                    new_labels = np.concatenate((new_labels, np.array([onehot_labels_valid[label_num]])), axis=0)
                    new_data = np.concatenate((new_data,np.array([X_valid[label_num]])), axis=0)
            elif len(unchosen_data) < 400 and max(confidence_output[0]) > 0.80:
                if len(unchosen_data) == 0:
                    unchosen_labels = np.array([onehot_labels_valid[label_num]])
                    unchosen_data = np.array([X_valid[label_num]])
                else:
                    unchosen_labels = np.concatenate((unchosen_labels, np.array([onehot_labels_valid[label_num]])), axis=0)
                    unchosen_data = np.concatenate((unchosen_data,np.array([X_valid[label_num]])), axis=0)                            

    return new_data, new_labels, unchosen_data, unchosen_labels


def train(dataset_conf, train_conf, results_path):
    # Get the current 'IN' time to calculate the overall training time
    in_exp = time.time()
    # Create a file to store the path of the best model among several runs
    best_models = open(results_path + "/best models.txt", "w")
    # Create a file to store performance during training
    log_write = open(results_path + "/log.txt", "w")
    # Create a .npz file (zipped archive) to store the accuracy and kappa metrics 
    # for all runs (to calculate average accuracy/kappa over all runs)
    perf_allRuns = open(results_path + "/perf_allRuns.npz", 'wb')
    
    # Get dataset paramters
    n_sub = dataset_conf.get('n_sub')
    data_path = dataset_conf.get('data_path')
    isStandard = dataset_conf.get('isStandard')
    LOSO = dataset_conf.get('LOSO')
    # Get training hyperparamters
    batch_size = train_conf.get('batch_size')
    epochs = train_conf.get('epochs')
    patience = train_conf.get('patience')
    lr = train_conf.get('lr')
    LearnCurves = train_conf.get('LearnCurves') # Plot Learning Curves?
    n_train = train_conf.get('n_train')
    model_name = train_conf.get('model')

    # Initialize variables
    acc = np.zeros((n_sub, n_train))
    kappa = np.zeros((n_sub, n_train))

    ### Visualize Subject Data Topography Map ###
    # print("VISUALISING TOPOGRAPHY")
    # for sub in range(0,n_sub):
    #     full_data,topo_labels,_,_ = get_data_openclose(sub)
    #     full_data = np.squeeze(full_data)
    #     print(full_data.shape)

    #     cutoff = 0
    #     for i in range(0,len(topo_labels)):
    #         if topo_labels[i].argmax(axis=0) == topo_labels[-1].argmax(axis=0):
    #             cutoff = i
    #             break
    #     print(cutoff)
    #     full_data_first = full_data[:cutoff]
    #     full_data_second = full_data[cutoff:]


    #     ch_names = ['Fp1','Fz','F3','F7','FT9','FC5','FC1','C3','T7','CP5','CP1','Pz','P3','P7','O1','Oz','O2','P4','P8','CP6','CP2','Cz','C4','T8','FT10','FC6','FC2','F4','F8','Fp2','AF7','AF3','AFz','F1','F5','FT7','FC3','C1','C5','TP7','CP3','P1','P5','PO7','PO3','POz','PO4','PO8','P6','P2','CPz','CP4','TP8','C6','C2','FC4','FT8','F6','AF8','AF4','F2']

    #     n_channels = 61
    #     n_timepoints = 1000
    #     # data = np.random.rand(n_channels, n_timepoints)  # Example random data

    #     ##### Rest
    #     ### Using actual EEG data
    #     data = full_data_first

    #     # Create MNE info object
    #     info = mne.create_info(ch_names,
    #                         sfreq=1000,  # Assuming sampling frequency is 1000 Hz
    #                         ch_types='eeg')
        
    #     montage = mne.channels.make_standard_montage('standard_1020')
    #     info.set_montage(montage)

    #     # Create MNE RawArray object
    #     # raw = mne.io.RawArray(data, info)

    #     # Average the data across timepoints
    #     evoked = mne.EvokedArray(np.mean(data, axis=0, keepdims=False), info)

    #     # Plot topographical map
    #     evoked.plot_topomap(times=0,  # Assuming you want to plot the topography at timepoint 0
    #                         time_format='ms',  # Time format in milliseconds
    #                         scalings=dict(eeg=1e2))

    #     savefile = './ImagesRestClose/' + str(sub) + 'rest.png'
    #     # plt.show()
    #     plt.savefig(savefile)

    #     ##### Open/Close
    #     ### Using actual EEG data
    #     data = full_data_second

    #     # Create MNE info object
    #     info = mne.create_info(ch_names,
    #                         sfreq=1000,  # Assuming sampling frequency is 1000 Hz
    #                         ch_types='eeg')
        
    #     montage = mne.channels.make_standard_montage('standard_1020')
    #     info.set_montage(montage)

    #     # Create MNE RawArray object
    #     # raw = mne.io.RawArray(data, info)

    #     # Average the data across timepoints
    #     # evoked = mne.EvokedArray(np.mean(data, axis=0, keepdims=False), info)

    #     average_data = np.mean(data, axis=(0, 2))
    #     # Create MNE EvokedArray object for the average data
    #     evoked = mne.EvokedArray(average_data[:, np.newaxis], info)

    #     # Plot topographical map
    #     evoked.plot_topomap(times=0,  # Assuming you want to plot the topography at timepoint 0
    #                         time_format='ms',  # Time format in milliseconds
    #                         scalings=dict(eeg=1e2))

    #     savefile = './ImagesRestClose/' + str(sub) + 'close.png'
    #     # plt.show()
    #     plt.savefig(savefile)
    #     print(sub)


    # Iteration over subjects 
    for sub in range(0,n_sub): # (num_sub): for all subjects, (i-1,i): for the ith subject.
        # Get the current 'IN' time to calculate the subject training time
        in_sub = time.time()
        print('\nTraining on subject ', sub+1)
        log_write.write( '\nTraining on subject '+ str(sub+1) +'\n')
        # Initiating variables to save the best subject accuracy among multiple runs.
        BestSubjAcc = 0 
        bestTrainingHistory = [] 
        # Get training and test data
        # X_train, _, y_train_onehot, X_test, _, y_test_onehot = get_data(
        #     data_path, sub, LOSO, isStandard)

        X_valid = []
        onehot_labels_valid = []
        X_train = []
        y_train_onehot = []
        # X_valid,onehot_labels_valid,X_train,y_train_onehot = subject_independent_data(sub)
        # X_valid = X_valid[1::4]
        # onehot_labels_valid = onehot_labels_valid[1::4]

        # print(X_train.shape)
        # print(y_train_onehot.shape)
        # print(X_valid.shape)

        # print("AUGMENTING DATA")
        # X_train,y_train_onehot = cut_data.cut(dataset_conf=dataset_conf,sub=sub,LOSO=LOSO)

        # print(X_train.shape)
        # print(y_train_onehot.shape)
        
        # Iteration over multiple runs 
        for train in range(n_train): # How many repetitions of training for subject i.
            # Get the current 'IN' time to calculate the 'run' training time
            in_run = time.time()
            # Create folders and files to save trained models for all runs
            filepath = results_path + '/saved models/run-{}'.format(train+1)
            if not os.path.exists(filepath):
                os.makedirs(filepath)        
            filepath = filepath + '/subject-{}.h5'.format(sub+1)

            # Check GPU
            print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
            # tf.debugging.set_log_device_placement(True)
            # a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            # b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            # c = tf.matmul(a, b)

            # print(c)
            
            # Create the model
            model = getModel(model_name)
            model_2 = getModel(model_name)

            # print(model.summary())
            # Compile and train the model
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.001,
                decay_steps=20,
                decay_rate=0.98,
                staircase=True)
            model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=lr_schedule), metrics=['accuracy'])          
            lr_schedule_2 = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.00005,
                decay_steps=10,
                decay_rate=0.95,
                staircase=True)
            model_2.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=lr_schedule_2), metrics=['accuracy'])          
            callbacks = [
                ModelCheckpoint(filepath, monitor='accuracy', save_best_only=True, save_weights_only=True, mode='max'),
                # ReduceLROnPlateau(factor=0.90, patience=20, verbose=1, min_lr=0.0001),
                # EarlyStopping(monitor='val_accuracy', verbose=1, mode='max', patience=patience)
            ]
            print("FITTING MODEL")
            # print(model.summary())
            # history = model.fit(X_train[:len(X_train)-0], y_train_onehot[:len(X_train)-0], validation_data=(X_train[len(X_train)-0:], y_train_onehot[len(X_train)-0:]), 
            #                     epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=2)


            X_train,y_train_onehot,X_test,y_test = get_data_openclose(sub)
            print(X_train.shape)
            print(X_test.shape)
            # X_train,y_train_onehot,X_test,y_test = subject_independent_data(sub)
            history = model.fit(X_train, y_train_onehot, validation_data=(X_test, y_test), 
                                epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1)
            # draw_learning_curves(history)
        
            print("SELF-SUPERVISED LEARNING")
            # model.load_weights(filepath)
            # y_pred = model.predict(X_valid).argmax(axis=-1)
            # labels = onehot_labels_valid.argmax(axis=-1)

            # new_data,new_labels, unchosen_data, unchosen_data_labels = choose_data(labels,y_pred,model,X_valid,onehot_labels_valid,0.95)

            ## Visualization
            ## Get Intermediate Output
            # from keras import backend as K

            # get_3rd_layer_output = K.function(
            # [model.layers[0].input], # param 1 will be treated as layer[0].output
            # [model.get_layer('lambda_5').output]) # and this function will return output from 3rd layer

            # output = get_3rd_layer_output([X_train])[0]
            # output_labels = np.where(y_train_onehot==1)[1]
            # # plot_clustering(output, output_labels, engine='matplotlib', download = False)

            # # output = get_3rd_layer_output([X_valid])[0]
            # # output_labels = np.where(onehot_labels_valid==1)[1]
            # plot_clustering(output, output_labels, engine='matplotlib', download = False)

            # output_selected = get_3rd_layer_output([new_data])[0]
            # new_labels_v = np.full((len(output_selected)),2)
            # output_plot = np.concatenate((output,output_selected),axis=0)
            # output_plot_labels = np.concatenate((output_labels,new_labels_v),axis=0)

            # output_unchosen = get_3rd_layer_output([unchosen_data])[0]
            # unchosen_labels = np.full((len(output_unchosen)),2)
            # output_plot2 = np.concatenate((output,output_unchosen),axis=0)
            # output_plot2_labels = np.concatenate((output_labels,unchosen_labels),axis=0)

            # print(output.shape)
            # print(output_labels.shape)
            # plot_clustering(output_plot, output_plot_labels, engine='matplotlib', download = False)
            # plot_clustering(output_plot2, output_plot2_labels, engine='matplotlib', download = False)

            ## FINETUNING
            
            # print(len(unchosen_data))
            # _,_,X_test,y_test = get_data_openclose(sub)
            # history = model_2.fit(unchosen_data, unchosen_data_labels, validation_data=(X_train, y_train_onehot), 
            #                     epochs=100, batch_size=batch_size, callbacks=callbacks, verbose=2)                
            # model_2.load_weights(filepath)
            # y_pred = model_2.predict(X_train).argmax(axis=-1)
            # labels = y_train_onehot.argmax(axis=-1)

            # new_data,new_labels, unchosen_data, unchosen_data_labels = choose_data(labels,y_pred,model_2,X_train,y_train_onehot,0.9)
            # if len(new_data) == 0:
            #     new_data,new_labels, unchosen_data, unchosen_data_labels = choose_data(labels,y_pred,model_2,X_train,y_train_onehot,0.85)
            # if len(new_data) == 0:
            #     new_data,new_labels, unchosen_data, unchosen_data_labels = choose_data(labels,y_pred,model_2,X_train,y_train_onehot,0.7)
            # print(new_data)

            # new_train = np.concatenate((unchosen_data, X_train), axis=0)
            # new_labels = np.concatenate((unchosen_data_labels, y_train_onehot), axis=0)
            # history = model.fit(new_train, new_labels, validation_data=(X_train, y_train_onehot), 
            #         epochs=100, batch_size=batch_size, callbacks=callbacks, verbose=2)                


            ## Evaluation
            # _,_,X_test,y_test = get_data_openclose(sub)
            # history2 = model_2.fit(X_test, y_test, validation_data=(X_test, y_test), 
            #                     epochs=0, batch_size=batch_size, callbacks=callbacks, verbose=2)
            # model.load_weights(filepath)
            y_pred = model.predict(X_test).argmax(axis=-1)
            labels = y_test.argmax(axis=-1)

            acc[sub, train]  = accuracy_score(labels, y_pred)
            kappa[sub, train] = cohen_kappa_score(labels, y_pred)
              
            # Get the current 'OUT' time to calculate the 'run' training time
            out_run = time.time()
            # Print & write performance measures for each run
            info = 'Subject: {}   Train no. {}   Time: {:.1f} m   '.format(sub+1, train+1, ((out_run-in_run)/60))
            info = info + 'Test_acc: {:.4f}   Test_kappa: {:.4f}'.format(acc[sub, train], kappa[sub, train])
            print(info)
            log_write.write(info +'\n')
            # If current training run is better than previous runs, save the history.
            if(BestSubjAcc < acc[sub, train]):
                 BestSubjAcc = acc[sub, train]
                 bestTrainingHistory = history
        
        # Store the path of the best model among several runs
        best_run = np.argmax(acc[sub,:])
        filepath = '/saved models/run-{}/subject-{}.h5'.format(best_run+1, sub+1)+'\n'
        best_models.write(filepath)
        # Get the current 'OUT' time to calculate the subject training time
        out_sub = time.time()
        # Print & write the best subject performance among multiple runs
        info = '----------\n'
        info = info + 'Subject: {}   best_run: {}   Time: {:.1f} m   '.format(sub+1, best_run+1, ((out_sub-in_sub)/60))
        info = info + 'acc: {:.4f}   avg_acc: {:.4f} +- {:.4f}   '.format(acc[sub, best_run], np.average(acc[sub, :]), acc[sub,:].std() )
        info = info + 'kappa: {:.4f}   avg_kappa: {:.4f} +- {:.4f}'.format(kappa[sub, best_run], np.average(kappa[sub, :]), kappa[sub,:].std())
        info = info + '\n----------'
        print(info)
        log_write.write(info+'\n')
        # Plot Learning curves 
        if (LearnCurves == True):
            print('Plot Learning Curves ....... ')
            draw_learning_curves(bestTrainingHistory)
          
    # Get the current 'OUT' time to calculate the overall training time
    out_exp = time.time()
    info = '\nTime: {:.1f} h   '.format( (out_exp-in_exp)/(60*60) )
    print(info)
    log_write.write(info+'\n')
    
    # Store the accuracy and kappa metrics as arrays for all runs into a .npz 
    # file format, which is an uncompressed zipped archive, to calculate average
    # accuracy/kappa over all runs.
    # print(acc)
    total = 0
    for i in range(0,n_sub):
        total += np.average(acc[i,:])
    print(total/n_sub)
    np.savez(perf_allRuns, acc = acc, kappa = kappa)
    
    # Close open files 
    best_models.close()   
    log_write.close() 
    perf_allRuns.close() 


#%% Evaluation 
def test(model, dataset_conf, results_path, allRuns = True):
    # Open the  "Log" file to write the evaluation results 
    log_write = open(results_path + "/log.txt", "a")
    # Open the file that stores the path of the best models among several random runs.
    best_models = open(results_path + "/best models.txt", "r")   
    
    # Get dataset paramters
    n_classes = dataset_conf.get('n_classes')
    n_sub = dataset_conf.get('n_sub')
    data_path = dataset_conf.get('data_path')
    isStandard = dataset_conf.get('isStandard')
    LOSO = dataset_conf.get('LOSO')
    
    # Initialize variables
    acc_bestRun = np.zeros(n_sub)
    kappa_bestRun = np.zeros(n_sub)  
    cf_matrix = np.zeros([n_sub, n_classes, n_classes])

    # Calculate the average performance (average accuracy and K-score) for 
    # all runs (experiments) for each subject.
    if(allRuns): 
        # Load the test accuracy and kappa metrics as arrays for all runs from a .npz 
        # file format, which is an uncompressed zipped archive, to calculate average
        # accuracy/kappa over all runs.
        perf_allRuns = open(results_path + "/perf_allRuns.npz", 'rb')
        perf_arrays = np.load(perf_allRuns)
        acc_allRuns = perf_arrays['acc']
        # print(acc_allRuns)
        kappa_allRuns = perf_arrays['kappa']
    
    # Iteration over subjects 
    for sub in range(n_sub): # (num_sub): for all subjects, (i-1,i): for the ith subject.
        # Load data
        # _, _, _, X_test, _, y_test_onehot = get_data(data_path, sub, LOSO, isStandard)
        _, _,X_test, y_test_onehot = get_data_openclose(sub)
        # Load the best model out of multiple random runs (experiments).
        filepath = best_models.readline()
        print(results_path+filepath[:-1])
        print(acc_allRuns[sub])
        print(max(acc_allRuns[sub]))
        model.load_weights(results_path + filepath[:-1])
        # Predict MI task
        y_pred = model.predict(X_test).argmax(axis=-1)
        # Calculate accuracy and K-score
        labels = y_test_onehot.argmax(axis=-1)
        # acc_bestRun[sub] = accuracy_score(labels, y_pred)
        acc_bestRun[sub] = max(acc_allRuns[sub])
        kappa_bestRun[sub] = cohen_kappa_score(labels, y_pred)
        # Calculate and draw confusion matrix
        cf_matrix[sub, :, :] = confusion_matrix(labels, y_pred, normalize='pred')
        # draw_confusion_matrix(cf_matrix[sub, :, :], str(sub+1), results_path)
        
        # Print & write performance measures for each subject
        info = 'Subject: {}   best_run: {:2}  '.format(sub+1, (filepath[filepath.find('run-')+4:filepath.find('/sub')]) )
        info = info + 'acc: {:.4f}   kappa: {:.4f}   '.format(acc_bestRun[sub], kappa_bestRun[sub] )
        if(allRuns): 
            info = info + 'avg_acc: {:.4f} +- {:.4f}   avg_kappa: {:.4f} +- {:.4f}'.format(
                np.average(acc_allRuns[sub, :]), acc_allRuns[sub,:].std(),
                np.average(kappa_allRuns[sub, :]), kappa_allRuns[sub,:].std() )
        print(info)
        log_write.write('\n'+info)
      
    # Print & write the average performance measures for all subjects     
    info = '\nAverage of {} subjects - best runs:\nAccuracy = {:.4f}   Kappa = {:.4f}\n'.format(
        n_sub, np.average(acc_bestRun), np.average(kappa_bestRun)) 
    if(allRuns): 
        info = info + '\nAverage of {} subjects x {} runs (average of {} experiments):\nAccuracy = {:.4f}   Kappa = {:.4f}'.format(
            n_sub, acc_allRuns.shape[1], (n_sub * acc_allRuns.shape[1]),
            np.average(acc_allRuns), np.average(kappa_allRuns)) 
    print(info)
    log_write.write(info)
    
    # Draw a performance bar chart for all subjects 
    draw_performance_barChart(n_sub, acc_bestRun, 'Accuracy')
    draw_performance_barChart(n_sub, kappa_bestRun, 'K-score')
    # Draw confusion matrix for all subjects (average)
    draw_confusion_matrix(cf_matrix.mean(0), 'All', results_path)
    # Close open files     
    log_write.close() 
    
    
#%%
def getModel(model_name):
    # Select the model
    if(model_name == 'ATCNet'):
        # Train using the proposed model (ATCNet): https://doi.org/10.1109/TII.2022.3197419
        model = models.ATCNet( 
            # Dataset parameters
            n_classes = 2, 
            in_chans = 61, 
            in_samples = 1000, 
            # Sliding window (SW) parameter
            n_windows = 5, 
            # Attention (AT) block parameter
            attention = 'mha', # Options: None, 'mha','mhla', 'cbam', 'se'
            # Convolutional (CV) block parameters
            eegn_F1 = 4,
            eegn_D = 2, 
            eegn_kernelSize = 16,
            eegn_poolSize = 7,
            eegn_dropout = 0.3,
            # Temporal convolutional (TC) block parameters
            tcn_depth = 2, 
            tcn_kernelSize = 4,
            tcn_filters = 16,
            tcn_dropout = 0.3, 
            tcn_activation='elu'
            )     
    elif(model_name == 'TCNet_Fusion'):
        # Train using TCNet_Fusion: https://doi.org/10.1016/j.bspc.2021.102826
        model = models.TCNet_Fusion(n_classes = 4)      
    elif(model_name == 'EEGTCNet'):
        # Train using EEGTCNet: https://arxiv.org/abs/2006.00622
        model = models.EEGTCNet(n_classes = 4)          
    elif(model_name == 'EEGNet'):
        # Train using EEGNet: https://arxiv.org/abs/1611.08024
        model = models.EEGNet_classifier(n_classes = 4) 
    elif(model_name == 'EEGNeX'):
        # Train using EEGNeX: https://arxiv.org/abs/2207.12369
        model = models.EEGNeX_8_32(n_timesteps = 1125 , n_features = 22, n_outputs = 4)
    elif(model_name == 'DeepConvNet'):
        # Train using DeepConvNet: https://doi.org/10.1002/hbm.23730
        model = models.DeepConvNet(nb_classes = 4 , Chans = 22, Samples = 1125)
    elif(model_name == 'ShallowConvNet'):
        # Train using ShallowConvNet: https://doi.org/10.1002/hbm.23730
        model = models.ShallowConvNet(nb_classes = 4 , Chans = 22, Samples = 1125)
    else:
        raise Exception("'{}' model is not supported yet!".format(model_name))

    return model
    
    
#%%
def run():
    # Get dataset path
    data_path = os.path.expanduser('~') + '/BCI Competition IV/BCI Competition IV-2a/'
    data_path = "D:/atcnet/bci2a_mat/"
    
    # Create a folder to store the results of the experiment
    results_path = os.getcwd() + "/results"
    if not  os.path.exists(results_path):
      os.makedirs(results_path)   # Create a new directory if it does not exist 
      
    # Set dataset paramters 
    dataset_conf = { 'n_classes': 2, 'n_sub': 50, 'n_channels': 61, 'data_path': data_path,
                'isStandard': True, 'LOSO': True}
    # Set training hyperparamters
    train_conf = { 'batch_size': 32, 'epochs': 200, 'patience': 300, 'lr': 0.00001,
                  'LearnCurves': False, 'n_train': 1, 'model':'ATCNet'}
           
    # Train the model
    train(dataset_conf, train_conf, results_path)

    # Evaluate the model based on the weights saved in the '/results' folder
    model = getModel(train_conf.get('model'))
    test(model, dataset_conf, results_path)    
    
#%%
# if __name__ == "__dataselect__":
run()