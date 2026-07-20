import numpy as np
import torch
import random
import os
import scipy

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_init(conv):
    """
    Initialize convolutional layer weights using Kaiming normal initialization and set biases to zero.

    Parameters
    ----------
    conv : torch.nn.Conv2d or torch.nn.Conv1d
        The convolutional layer to be initialized.
    """
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    """
    Initialize batch normalization layer weights to a specified scale and set biases to zero.

    Parameters
    ----------
    bn : torch.nn.BatchNorm2d or torch.nn.BatchNorm1d
        The batch normalization layer to be initialized.
    scale : float
        The scale value to initialize the weights of the batch normalization layer.
    """
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)