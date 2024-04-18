from sklearn.metrics import confusion_matrix, cohen_kappa_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os

def calculate_statistics(y_true, y_pred, y_hat, file_name):
    fpr, tpr, thresholds = roc_curve(y_true, y_hat, pos_label=1)
    AUC = auc(fpr, tpr)
    print("AUC:", AUC)
    file = open(file_name, 'a')
    file.write("AUC:" + str(AUC) + '\n')
    '''
    conf_matrix = confusion_matrix(y_true, y_pred)
    sns.set()
    f, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, ax=ax)

    ax.set_title('confusion matrix')
    ax.set_xlabel('predict')
    ax.set_ylabel('true')
    plt.show()
    '''
    kappa_value = cohen_kappa_score(y_true, y_pred)
    print("kappa valus: %f" % kappa_value)
    file.write("kappa valus: %f" % kappa_value + '\n')

def set_gpu(x):
    torch.set_num_threads(1)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)



def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def remove_empty(data,label,file,strn):
    bad = np.where(label==-1)
    edata = np.delete(data, bad, 0)
    elabel = np.delete(label, bad, 0)
    print('removed', len(bad[0]) ,'segment(s) \n')
    
    file = open(file, 'a')
    file.write(strn + 'set - Dropped segments: ' + str(len(bad[0])) + '/' + str(len(label)) + '\n')
    file.close()
    
    return edata, elabel

def band_pass(raw, freq, fs, filtfilt, axis, plot_,trial): 
    import scipy
    import scipy.signal
    import numpy as np   
    
    # zero padding check
    iz = np.all(raw==0,axis=2)
    
    nx = raw.shape
    data = raw.transpose(0,2,1).reshape((nx[0]*nx[2],-1))
    
    low_cut_hz = freq[0]
    high_cut_hz = freq[1]
    step_cut_Hz = freq[2]
    Apass = 3
    Astop = 40
    
    nyq_freq = 0.5 * fs
    low = low_cut_hz / nyq_freq
    high = high_cut_hz / nyq_freq
    step = step_cut_Hz / nyq_freq
    
    if low_cut_hz-step_cut_Hz<=0:
        N, Wn = scipy.signal.cheb2ord(high, high+step, Apass, Astop)
        b, a = scipy.signal.cheby2(N, Astop, Wn)
    else:
        N, Wn = scipy.signal.cheb2ord([low, high], [low-step,high+step], Apass, Astop)
        if trial: N=10
        b, a = scipy.signal.cheby2(N, Astop, Wn, 'bp')

    # N, Wn = scipy.signal.cheb2ord([0.1, 0.6], [0.2, 0.5], 3, 60)
    # b, a = scipy.signal.cheby2(N, 60, Wn, 'stop')
    
    if plot_:
        w, h = scipy.signal.freqz(b, a)
        # plt.semilogx(w/np.pi, 20 * np.log10(abs(h)))
        plt.plot(w*fs/2/np.pi, 20 * np.log10(abs(h)))
        plt.title('Chebyshev Type II frequency response')
        plt.xlabel('Frequency [x fs/2 Hz]')
        plt.ylabel('Amplitude [dB]')
        plt.margins(0, .05)
        plt.grid(which='both', axis='both')
        
        plt.axvline(low, color='green') # cutoff frequency
        plt.axvline(high, color='green') # cutoff frequency
        plt.axhline(-Astop, color='green') # rs
        plt.show()    

    if filtfilt:
        data_bandpassed = scipy.signal.filtfilt(b, a, data, axis=axis)
    else:
        data_bandpassed = scipy.signal.lfilter(b, a, data, axis=axis)           
        
    if plot_:
        plt.figure()
        plt.plot(data[0])
        plt.plot(data_bandpassed[0])
        plt.show()        
    
    data_bandpassed = data_bandpassed.reshape(nx[0],nx[2],-1).transpose(0,2,1)
    
    #replace zeropadding
    data_bandpassed[iz] = 0
    
    return data_bandpassed  

# def bandpassfilter_cheby2_sos(data, bandFiltCutF=[0.3, 40], fs, filtAllowance=[0.2, 5], axis=2):
def bandpassfilter_cheby2_sos(data, bandFiltCutF, fs, filtAllowance, axis):

    '''
    Band-pass filter the EEG signal of one subject using cheby2 IIR filtering
    and implemented as a series of second-order filters with direct-form II transposed structure.
    Param:
        data: nparray, size [trials x channels x times], original EEG signal
        bandFiltCutF: list, len: 2, low and high cut off frequency (Hz),
                If any value is None then only one-side filtering is performed.
        fs: sampling frequency (Hz)
        filtAllowance: list, len: 2, transition bandwidth (Hz) of low-pass and high-pass f
        axis: the axis along which apply the filter.
    Returns:
        data_out: nparray, size [trials x channels x times], filtered EEG signal
    '''
    import numpy as np
    import scipy.signal as signal
    aStop = 40  # stopband attenuation
    aPass = 1  # passband attenuation
    nFreq = fs / 2  # Nyquist frequency
    data = np.squeeze(data)
    if (bandFiltCutF[0] == 0 or bandFiltCutF[0] is None) and (bandFiltCutF[1] == None or bandFiltCutF[1] >= fs / 2.0):
        # no filter
        print("Not doing any filtering. Invalid cut-off specifications")
        return data

    elif bandFiltCutF[0] == 0 or bandFiltCutF[0] is None:
        # low-pass filter
        print("Using lowpass filter since low cut hz is 0 or None")
        fPass = bandFiltCutF[1] / nFreq
        fStop = (bandFiltCutF[1] + filtAllowance[1]) / nFreq
        # find the order
        [N, ws] = signal.cheb2ord(fPass, fStop, aPass, aStop)
        sos = signal.cheby2(N, aStop, fStop, 'lowpass', output='sos')

    elif (bandFiltCutF[1] is None) or (bandFiltCutF[1] == fs / 2.0):
        # high-pass filter
        print("Using highpass filter since high cut hz is None or nyquist freq")
        fPass = bandFiltCutF[0] / nFreq
        fStop = (bandFiltCutF[0] - filtAllowance[0]) / nFreq
        # find the order
        [N, ws] = signal.cheb2ord(fPass, fStop, aPass, aStop)
        sos = signal.cheby2(N, aStop, fStop, 'highpass', output='sos')

    else:
        # band-pass filter
        # print("Using bandpass filter")
        fPass = (np.array(bandFiltCutF) / nFreq).tolist()
        fStop = [(bandFiltCutF[0] - filtAllowance[0]) / nFreq, (bandFiltCutF[1] + filtAllowance[1]) / nFreq]
        # find the order
        [N, ws] = signal.cheb2ord(fPass, fStop, aPass, aStop)
        sos = signal.cheby2(N, aStop, fStop, 'bandpass', output='sos')

    dataOut = signal.sosfilt(sos, data, axis=axis)
    # dataOut = signal.sosfiltfilt(sos, data, axis=axis)
    
    # plt.figure()
    # # plt.plot(data[0,0,:500])
    # plt.plot(dataOut[0,0,:500])
    # plt.show() 

    return dataOut[:,None,None,:,:]
"""
Very specific to result file that is saved after running main_ma
"""

import pandas as pd
import numpy as np


def file_save(tname):
    with open(tname) as f:
        lines = f.readlines()
       
    # if lines[5][-2:-1] == '0':
    #     ws = 'na'
    # else: 
    #     ind1 = str.find(lines[6],':')    
    #     ind2 = str.find(lines[7],':')
    #     ws = lines[6][ind1+1:-1] + '-' + lines[7][ind2+1:-1]
    #     # ws = ws.replace('.','')
    
    # ind = str.find(lines[11],':')
    # ind1 = str.find(lines[3],':')
    # fname = 'class-' + lines[3][ind1+1:-1] + '-chan-' + lines[9][-5:-1] + '-seg-' + ws + '-net-' + lines[11][ind+1:-1]
    
    data = []
    res=[]
    sub = []
    k=0
    for line in lines:
        if 'Subject:' + str(k) in line:         
            ind = str.find(line,'ACC:')
            res.append(float(line[ind+4:-1]))
            if len(res) == 5:             
                sub.append(str(k))
                data.append(np.vstack(res))
                res=[]
                k+=1
            
    df = pd.DataFrame(np.hstack(data), columns = sub)  
    
    resfolder = os.getcwd() + '/results/'
    
    if not os.path.exists(resfolder): 
        os.makedirs(resfolder)
        os.makedirs(resfolder + '/txt/')
    
    from datetime import datetime
    
    dstr = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    df.to_csv(resfolder + 'result-' + dstr + '.csv') 
    
    os.rename(tname, resfolder + '/txt/results-' + dstr + '.txt')