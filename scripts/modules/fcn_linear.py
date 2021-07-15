import os
import sys
import numpy as np
import random
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# class for fully connected linear neural network
class fully_connected_linear_network( nn.Module ):
    
    # define and intialize the structure of the neural network
    def __init__( self, input_size, output_size, learning_rate ):
        
        super( fully_connected_linear_network, self ).__init__()
        
        # define hyperparameters
        self.input_size    = input_size
        self.output_size   = output_size
        self.learning_rate = learning_rate
        
        # define layers
        
        self.layer = nn.Linear( self.input_size, self.output_size )
        self.optimizer = torch.optim.Adam( self.parameters(), lr=self.learning_rate )
        #self.optimizer = torch.optim.SGD( self.parameters(), lr=self.learning_rate )
        
    def forward( self, x ):
        output = self.layer( x )
        return output
