# -*- coding: utf-8 -*-
"""
Created on Sat, Jan 23 23:16:50 2021

Utility functions for CNN and ResNet 

@author: JDE65 (Github)
j.dessain@navagne.com   ///  j.dessain@ieseg.fr
www.navagne.com
All rights reserved - Copyright Navagne (2020)
"""
# ====  PART 0. Installing libraries ============

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


## 0. Adjusted Cross-entropy loss function
"""
Define a crossentropy loss function that adjusts for the distance between classes
Loss function = cross entropy times exp(abs(y-yhat))
"""
class AdjCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(AdjCrossEntropyLoss, self).__init__()
                
    def forward(self, outputs, labels):
        num_examples = labels.shape[0]
        batch_size = outputs.size()[0]                  # batch_size 
        _, out_adj = torch.max(outputs, 1)              # Create a tensor with the predicted class
        out_adj = abs(out_adj - labels) / np.exp(1)     # Compute the distance between the true class and the predicted class and divide by exp(1)
        out_adj = torch.exp(out_adj)                    # Adjustment = exp(absolute distance / e)
        outputs = F.log_softmax(outputs, dim=1)         # compute the log of softmax values
        outputs = outputs[range(batch_size), labels]    # pick the values corresponding to the labels
        outputs = outputs * out_adj                     # loss is weighted by a parameter link to the distance between the true class and the predicted class
        return -torch.sum(outputs)/num_examples
    

