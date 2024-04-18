# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 10:59:39 2022

@author: NEETHU

"""

def eventlist_grazmi(event, fsamp):
    """
    Event list:   (UI --> VMRK --> MNE --> description)     
    100 --> S  4 --> 4 --> Experiment start/START_EXP_LABEL
    101 --> S  5 --> 5 --> ??/INTRUCTIONS_LABEL
    102 --> S  6 --> 6 --> Trial start/START_TRIAL_LABEL
    103 --> S  7 --> 7 --> Presentation of Left Cue/LEFT_PROMPT_LABEL
    105 --> S  9 --> 9 --> Presentation of Right Cue/RIGHT_PROMPT_LABEL
    104 --> S  8 --> 8 --> Start Left MI/LEFT_MI_LABEL
    106 --> S 10 --> 10 --> Start Right MI/RIGHT_MI_LABEL
    107 --> S 11 --> 11 --> Stop MI/RIGHT_MI_LABEL
    108 --> S 12 --> 12 --> Experiment stop/END_EXP_LABEL
    """
    codeL = 8
    codeR = 10
    
    ev = dict({'code': [], 'sampstart': [],'label': []})
    for i,t in enumerate(event[0]):     
        if t[2]!=1006: #exclude all the duplicate markers
            if (t[2] == codeL):
                ev['label'].append('left_mi')
                ev['code'].append(0)
            elif (t[2] == codeR):
                ev['label'].append('right_mi')
                ev['code'].append(1)
            else:
                ev['label'].append('non')
                ev['code'].append(-1)
            ev['sampstart'].append(t[0])    
    return ev

def eventlist_cirgmi(event, fsamp): #old version
    """
    Event list: (UI --> VMRK --> MNE --> description)     
    102 --> R  4 --> Start of trial
    103 --> R  6 --> Presentation of Left Cue 
    105 --> R 10 --> Presentation of Right Cue
    104 --> R  8 --> Left MI starts  
    106 --> R 12 --> Right MI starts
    107 --> R 14 --> MI ends    
    """
    codeL = 1004
    codeR = 1006
    
    ev = dict({'code': [], 'sampstart': [],'label': []})
    for i,t in enumerate(event[0]):        
        if (t[2] == codeL):
            ev['label'].append('left_mi')
            ev['code'].append(0)
        elif (t[2] == codeR):
            ev['label'].append('right_mi')
            ev['code'].append(1)
        else:
            ev['label'].append('non')
            ev['code'].append(-1)
        ev['sampstart'].append(t[0])    
    return ev

def eventlist_HRmi(event, fsamp):
    """
    Event list: (UI --> VMRK --> MNE --> description)     
    100 --> S  4 --> 4 --> Experiment start/START EXP_LABEL
    101 --> S  5 --> 5 --> FIXATION_LABEL 
    102 --> S  6 --> 6 --> PREP_CLOSE_LABEL
    103 --> S  7 --> 7 --> PREP_OPEN_LABEL
    104 --> S  8 --> 8 --> TRIAL_CLOSE_LABEL
    105 --> S  9 --> 9 --> TRIAL_OPEN_LABEL
    106 --> S 10 --> 10 --> REST_LABEL
    107 --> S 11 --> 11 --> END_EXP_LABEL
    """
    codeO = 8
    codeC = 9
    codeR = 10
    
    ev = dict({'code': [], 'sampstart': [],'label': []})
    for i,t in enumerate(event[0]):   
        if t[2]!=1006: #exclude all the duplicate markers
            # if (t[2] == codeO):
            #     ev['label'].append('hand_open')
            #     ev['code'].append(0)
            # elif (t[2] == codeC):
            #     ev['label'].append('hand_close')
            #     ev['code'].append(1)
            if (t[2] == codeR):
                ev['label'].append('hand_rest')
                ev['code'].append(1)
            else:
                ev['label'].append('non')
                ev['code'].append(-1)
            ev['sampstart'].append(t[0])    
    return ev

def eventlist_ROC(event, fsamp):
    """
    Event list: (UI --> VMRK --> MNE --> description)
   codeO = 107, beforeaudioO=106
   codeC = 104, beforeaudioC=103
   codeR=110,BESTILLAUDIO=109


    """
    codeO = 107
    codeC = 104
    codeR = 110
    import numpy as np
    rr = event[0]
    rr2 = np.delete(rr, rr[:, 2] == 1006, axis=0)  # deleting the 1006
    ev = dict({'code': [], 'sampstart': [], 'label': []})
    trialorder = 0
    for t in range(0, rr2.shape[0]):
        # print(rr2[t][2])
        # if  rr2[t][2]==5:
        if (rr2[t][2] == 109 and rr2[t + 1][2] == 110):  # bestill audio 109 and bestill task is 110
            ev['label'].append('rest')  #
            ev['code'].append(0)
            ev['sampstart'].append(rr2[t + 1][0])
            trialorder = trialorder + 1
        # if (rr2[t][2] == 106 and rr2[t + 1][2] == 107):  # prep open audio 106 and task open 107
        #     ev['label'].append('hand_open')
        #     ev['code'].append(1)
        #     ev['sampstart'].append(rr2[t + 1][0])
        #     trialorder = trialorder + 1
        if (rr2[t][2] == 103 and rr2[t + 1][2] == 104):  # prep close audio 103 and task close 104
            ev['label'].append('hand_close')
            ev['code'].append(2)
            ev['sampstart'].append(rr2[t + 1][0])
            trialorder = trialorder + 1

    return ev