import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

def contrastive_loss( x_i, x_j, xdevice, temperature, alpha):
    batch_size = x_i.shape[0]
    z_i = F.normalize( x_i, dim=1 )
    z_j = F.normalize( x_j, dim=1 )
    z   = torch.cat( [z_i, z_j], dim=0 )
    similarity_matrix = F.cosine_similarity( z.unsqueeze(1), z.unsqueeze(0), dim=2 )
    sim_ij = torch.diag( similarity_matrix,  batch_size ) # batch_size above the main diagonal -- x_i x_i'
    sim_ji = torch.diag( similarity_matrix, -batch_size ) # below the main diagonal -- x_i' x_i
    positives = torch.cat( [sim_ij, sim_ji], dim=0 )
    nominator = torch.exp( positives / temperature )
    negatives_mask = ( ~torch.eye( 2*batch_size, 2*batch_size, dtype=bool ) ).float()
    negatives_mask = negatives_mask.to( xdevice )
    denominator = negatives_mask * torch.exp( similarity_matrix / temperature )
    loss_partial = -torch.log( nominator / (torch.sum( denominator, dim=1 )).pow(exponent=alpha) )
    loss = torch.sum( loss_partial )/( 2*batch_size )
    return loss



def contrastive_loss_num_den( x, y, xdevice, alpha ):
    # the numerator and denominator inside the log
    batch_size = x.shape[0]
    
    reps_x = x.clone()
    reps_y = y.clone()
    reps_x = F.normalize(reps_x, dim=1).to(xdevice)
    reps_y = F.normalize(reps_y, dim=1).to(xdevice)

    z   = torch.cat( [reps_x, reps_y], dim=0 )
    similarity_matrix = F.cosine_similarity( z.unsqueeze(1), z.unsqueeze(0), dim=2 )
    sim_ij = torch.diag( similarity_matrix,  batch_size ) # above the main diagonal
    sim_ji = torch.diag( similarity_matrix, -batch_size ) # below the main diagonal
    positives = torch.cat( [sim_ij, sim_ji], dim=0 )
    negatives_mask = ( ~torch.eye( 2*batch_size, 2*batch_size, dtype=bool ) ).float()
    negatives_mask = negatives_mask.to( xdevice )
    denominator = negatives_mask * torch.exp( similarity_matrix  )
    
    loss_partial_num = -1.0* ( positives )
    loss_numer = torch.sum( loss_partial_num )/( 2*batch_size )
    loss_partial_den = -torch.log( 1.0 / (torch.sum( denominator, dim=1 )).pow(exponent=alpha) )
    loss_denom = torch.sum( loss_partial_den )/( 2*batch_size )
    
    # returns what you want to DECREASE
    return loss_numer, loss_denom



"""
Loss proxies
"""

def align_loss(x, y, xdevice, alpha=2):
    #xdevice = x.get_device()
    reps_x = x.clone()
    reps_y = y.clone()
    reps_x = F.normalize(reps_x, dim=1).to(xdevice)
    reps_y = F.normalize(reps_y, dim=1).to(xdevice)
    loss_align = (reps_x-reps_y).norm(p=2, dim=1).pow(exponent=alpha).mean()
    return loss_align

def uniform_loss(x, xdevice, t=2):
    #xdevice = x.get_device()
    reps_x = x.clone()
    reps_x = F.normalize(reps_x, dim=1).to(xdevice)
    loss_uniform = torch.pdist(reps_x, p=2).pow(2).mul(-t).exp().mean().log()
    return loss_uniform
    