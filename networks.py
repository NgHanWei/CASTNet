# This is the script of idea TSception-II
# The variance layer and 1 x 1 conv is added
# Modified by Ding
# 2-Jun-2020
import torch
import torch.nn as nn
# torch.set_default_tensor_type(torch.cuda.FloatTensor)
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

######################################### TSception ####################################
class TSception(nn.Module):
    def __init__(self, num_classes, input_size, sampling_rate, num_T, num_S, hiden, dropout_rate):
        # input_size: EEG channel x datapoint
        super(TSception, self).__init__()
        self.inception_window = [0.5, 0.25, 0.125, 0.0625, 0.03125]
        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.Tception1 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, int(self.inception_window[0] * sampling_rate)), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 16)))
            #nn.Conv2d(num_T, num_T, kernel_size=(1,))
        self.Tception2 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, int(self.inception_window[1] * sampling_rate)), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 16)))
        self.Tception3 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, int(self.inception_window[2] * sampling_rate)), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 16)))

        self.Sception1 = nn.Sequential(
            nn.Conv2d(num_T, num_S, kernel_size=(int(input_size[0]), 1), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)))
        self.Sception2 = nn.Sequential(
            nn.Conv2d(num_T, num_S, kernel_size=(int(input_size[0] * 0.5), 1), stride=(int(input_size[0] * 0.5), 1),
                      padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)))

        self.BN_t = nn.BatchNorm2d(num_T)
        self.BN_s = nn.BatchNorm2d(num_S)
        size = self.get_size(input_size)

        self.fc1 = nn.Sequential(
            nn.Linear(size[1], hiden),
            nn.ReLU(),
            nn.Dropout(dropout_rate))
        self.fc2 = nn.Sequential(
            nn.Linear(hiden, num_classes),)
            #nn.LogSoftmax())

    def forward(self, x):
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)
        z = self.Sception1(out)
        out_final = z
        z = self.Sception2(out)
        out_final = torch.cat((out_final, z), dim=2)
        out = self.BN_s(out_final)
        out = out.view(out.size()[0], -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    def get_size(self, input_size):
        # here we use and array with the shape being
        # (1(mini-batch),1(convolutional channel),EEG channel,time data point)
        # to simulate the input data and get the output size
        data = torch.ones((1, 1, input_size[0], input_size[1]))

        y = self.Tception1(data)
        out = y
        y = self.Tception2(data)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(data)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)
        z = self.Sception1(out)
        out_final = z
        z = self.Sception2(out)
        out_final = torch.cat((out_final, z), dim=2)
        out = self.BN_s(out_final)
        out = out.view(out.size()[0], -1)
        return out.size()

################################### Revised TSception ###################################
class TSception_Revised(nn.Module):
    def __init__(self, num_classes, input_size, sampling_rate, num_T, num_S, hiden, dropout_rate):
        # input_size: EEG channel x datapoint
        super(TSception_Revised, self).__init__()
        self.inception_window = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]
        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.Tception1 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, int(self.inception_window[0] * sampling_rate)), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            # too much pooling for second conv layer (when inception window is 256)
            nn.Conv2d(num_T, num_T, kernel_size=(1,int(self.inception_window[1] * sampling_rate)), stride=1, padding=0),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)))
        self.Tception2 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, int(self.inception_window[1] * sampling_rate)), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Conv2d(num_T, num_T, kernel_size=(1,int(self.inception_window[2] * sampling_rate)), stride=1, padding=0),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)))
        self.Tception3 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, int(self.inception_window[2] * sampling_rate)), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Conv2d(num_T, num_T, kernel_size=(1,int(self.inception_window[3] * sampling_rate)), stride=1, padding=0),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)))

        self.Sception1 = nn.Sequential(
            nn.Conv2d(num_T, num_S, kernel_size=(int(input_size[0]), 1), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)))
        self.Sception2 = nn.Sequential(
            nn.Conv2d(num_T, num_S, kernel_size=(int(input_size[0] * 0.5), 1), stride=(int(input_size[0] * 0.5), 1),
                      padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)))

        self.BN_t = nn.BatchNorm2d(num_T)
        self.BN_s = nn.BatchNorm2d(num_S)
        size = self.get_size(input_size)

        self.fc1 = nn.Sequential(
            nn.Linear(size[1], hiden),
            nn.ReLU(),
            nn.Dropout(dropout_rate))
        self.fc2 = nn.Sequential(
            nn.Linear(hiden, num_classes),
            nn.LogSoftmax())

    def forward(self, x):
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)
        z = self.Sception1(out)
        out_final = z
        z = self.Sception2(out)
        out_final = torch.cat((out_final, z), dim=2)
        out = self.BN_s(out_final)
        out = out.view(out.size()[0], -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    def get_size(self, input_size):
        # here we use and array with the shape being
        # (1(mini-batch),1(convolutional channel),EEG channel,time data point)
        # to simulate the input data and get the output size
        data = torch.ones((1, 1, input_size[0], input_size[1])) #1,1,4,1024

        y = self.Tception1(data)
        out = y
        y = self.Tception2(data)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(data)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)
        z = self.Sception1(out)
        out_final = z
        z = self.Sception2(out)
        out_final = torch.cat((out_final, z), dim=2)
        out = self.BN_s(out_final)
        out = out.view(out.size()[0], -1)
        return out.size()
################################### Revised TSception ###################################
######################################### EEGNet ########################################

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)


class EEGNet(nn.Module):
    def __init__(self, n_classes=2, channels=4, samples=1024,
                 dropoutRate=0.5, kernelLength=128, kernelLength2=16, F1=8,
                 D=2, F2=16, ):
        super(EEGNet, self).__init__()
        self.F1 = F1
        self.F2 = F1 * D
        self.D = D
        self.samples = samples
        self.n_classes = n_classes
        self.channels = channels
        self.kernelLength = kernelLength
        self.kernelLength2 = kernelLength2
        self.dropoutRate = dropoutRate

        self.blocks = self.InitialBlocks(dropoutRate)
        self.blockOutputSize = self.CalculateOutSize(self.blocks, channels, samples)
        self.classifierBlock = self.ClassifierBlock(self.F2, n_classes, (1, self.blockOutputSize[1]))


    def InitialBlocks(self, dropoutRate, *args, **kwargs):
        block1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, self.kernelLength), stride=1, padding=(0, self.kernelLength // 2), bias=False),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.channels, 1), max_norm=1, stride=1, padding=(0, 0),
                                 groups=self.F1, bias=False),
            nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=4),
            nn.Dropout(p=dropoutRate))
        block2 = nn.Sequential(
            nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, self.kernelLength2), stride=1,
                      padding=(0, self.kernelLength2 // 2), bias=False, groups=self.F1 * self.D),
            nn.Conv2d(self.F1 * self.D, self.F2, 1, padding=(0, 0), groups=1, bias=False, stride=1),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(p=dropoutRate))
        return nn.Sequential(block1, block2)

    def ClassifierBlock(self, inputSize, n_classes, kernelSize, *args, **kwargs):
        return nn.Sequential(
            nn.Conv2d(inputSize, n_classes, kernelSize, *args, **kwargs),)
            #nn.Softmax(dim=1))

    def CalculateOutSize(self, model, channels, samples):
        '''
        Calculate the output based on input size.
        model is from nn.Module and inputSize is a array.
        '''
        data = torch.rand(1, 1, channels, samples)
        model.eval()
        out = model(data).shape
        return out[2:]

    def forward(self, x):
        x = self.blocks(x)
        x = self.classifierBlock(x)
        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)

        return x

###################################### deepConvNet ############################
class deepConvNet(nn.Module):
    def convBlock(self, inF, outF, dropoutP, kernalSize,  *args, **kwargs):
        return nn.Sequential(
            nn.Dropout(p=dropoutP),
            Conv2dWithConstraint(inF, outF, kernalSize, bias= False, max_norm = 2, *args, **kwargs),
            nn.BatchNorm2d(outF),
            nn.ELU(),
            nn.MaxPool2d((1,3), stride = (1,3))
            )

    def firstBlock(self, outF, dropoutP, kernalSize, nChan, *args, **kwargs):
        return nn.Sequential(
                Conv2dWithConstraint(1,outF, kernalSize, padding = 0, max_norm = 2, *args, **kwargs),
                Conv2dWithConstraint(25, 25, (nChan, 1), padding = 0, bias= False, max_norm = 2),
                nn.BatchNorm2d(outF),
                nn.ELU(),
                nn.MaxPool2d((1,3), stride = (1,3))
                )

    def lastBlock(self, inF, outF, kernalSize, *args, **kwargs):
        return nn.Sequential(
                Conv2dWithConstraint(inF, outF, kernalSize, max_norm = 0.5,*args, **kwargs),)
                #nn.LogSoftmax(dim = 1))

    def calculateOutSize(self, model, nChan, nTime):
        '''
        Calculate the output based on input size.
        model is from nn.Module and inputSize is a array.
        '''
        data = torch.rand(1,1,nChan, nTime)
        model.eval()
        out = model(data).shape
        return out[2:]

    def __init__(self, nChan, nTime, nClass = 2, dropoutP = 0.25, *args, **kwargs):
        super(deepConvNet, self).__init__()

        kernalSize = (1,10)
        nFilt_FirstLayer = 25
        nFiltLaterLayer = [25, 50, 100, 200]

        firstLayer = self.firstBlock(nFilt_FirstLayer, dropoutP, kernalSize, nChan)
        middleLayers = nn.Sequential(*[self.convBlock(inF, outF, dropoutP, kernalSize)
            for inF, outF in zip(nFiltLaterLayer, nFiltLaterLayer[1:])])

        self.allButLastLayers = nn.Sequential(firstLayer, middleLayers)

        self.fSize = self.calculateOutSize(self.allButLastLayers, nChan, nTime)
        self.lastLayer = self.lastBlock(nFiltLaterLayer[-1], nClass, (1, self.fSize[1]))

    def forward(self, x):

        x = self.allButLastLayers(x)
        x = self.lastLayer(x)
        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)

        return x
    

###################################### deepConvNet ############################
class deepConvNet_kavi(nn.Module):
    def convBlock(self, inF, outF, dropoutP, kernalSize,  *args, **kwargs):
        return nn.Sequential(
            nn.Dropout(p=dropoutP),
            Conv2dWithConstraint(inF, outF, kernalSize, bias= False, max_norm = 2, *args, **kwargs),
            nn.BatchNorm2d(outF),
            nn.ELU(),
            nn.MaxPool2d((1,3), stride = (1,3))
            )

    def firstBlock(self, outF, dropoutP, kernalSize, nChan, *args, **kwargs):
        return nn.Sequential(
                Conv2dWithConstraint(1,outF, kernalSize, padding = 0, max_norm = 2, *args, **kwargs),
                Conv2dWithConstraint(25, 25, (nChan, 1), padding = 0, bias= False, max_norm = 2),
                nn.BatchNorm2d(outF),
                nn.ELU(),
                nn.MaxPool2d((1,3), stride = (1,3))#change can
                )

    def lastBlock(self, inF, outF, kernalSize, *args, **kwargs):
        return nn.Sequential(
                Conv2dWithConstraint(inF, outF, kernalSize, max_norm = 0.5,*args, **kwargs),)
                #nn.LogSoftmax(dim = 1))

    def calculateOutSize(self, model, nChan, nTime):
        '''
        Calculate the output based on input size.
        model is from nn.Module and inputSize is a array.
        '''
        data = torch.rand(1,1,nChan, nTime)
        model.eval()
        out = model(data).shape
        return out[2:]

    def __init__(self, nChan, nTime, nClass = 2, dropoutP = 0.25, *args, **kwargs):
        super(deepConvNet_kavi, self).__init__()

        kernalSize = (1,10)
        nFilt_FirstLayer = 25
        nFiltLaterLayer = [25, 50, 100, 200]

        firstLayer = self.firstBlock(nFilt_FirstLayer, dropoutP, kernalSize, nChan)
        # middleLayers = nn.Sequential(*[self.convBlock(inF, outF, dropoutP, kernalSize)
        #     for inF, outF in zip(nFiltLaterLayer, nFiltLaterLayer[1:])])
        
        middleLayers0 = nn.Sequential(self.convBlock(25, 50, dropoutP, (1,10)))
        middleLayers1 = nn.Sequential(self.convBlock(50, 100, dropoutP, (1,10)))
        middleLayers2 = nn.Sequential(self.convBlock(100, 200, dropoutP, (1,10)))
        
        
        self.allButLastLayers = nn.Sequential(firstLayer, middleLayers0, middleLayers1, middleLayers2)

        self.fSize = self.calculateOutSize(self.allButLastLayers, nChan, nTime)
        self.lastLayer = self.lastBlock(nFiltLaterLayer[-1], nClass, (1, self.fSize[1]))

    def forward(self, x):

        x = self.allButLastLayers(x)
        x = self.lastLayer(x)
        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)

        return x