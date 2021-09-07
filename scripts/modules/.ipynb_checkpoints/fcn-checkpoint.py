import os
import sys
import numpy as np
import random
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# class for fully connected neural network
class fully_connected_network( nn.Module ):
    
    # define and intialize the structure of the neural network
    def __init__( self, input_size, output_size, hidden_size, n_hidden, learning_rate ):
        
        super( fully_connected_network, self ).__init__()
        
        # define hyperparameters
        self.input_size    = input_size
        self.output_size   = output_size
        self.hidden_size   = hidden_size
        self.n_hidden      = n_hidden
        self.learning_rate = learning_rate
        
        # define layers
        
        self.input_layer = nn.Linear( self.input_size, self.hidden_size )
        self.hidden_layers = nn.ModuleList()

        for i in range( self.n_hidden ):
            self.hidden_layers.append( nn.Linear( self.hidden_size, self.hidden_size ) )    

        self.output_layer = nn.Linear( self.hidden_size, self.output_size )
        self.optimizer = torch.optim.Adam( self.parameters(), lr=self.learning_rate )
        
    def forward( self, x ):
        x = F.relu( self.input_layer( x ) )
        for layer in self.hidden_layers:
            x = F.relu(layer( x ))
        output = self.output_layer( x )
        return output
