import os
import sys
import numpy as np
import random
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


# function to translate jets in eta and phi
def translate_jet_vectorized(batch, width=1.0):
    '''
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of eta-phi translated jets, same shape as input
    '''
    
    mask = (batch[:,0] > 0) # 1 for constituents with non-zero pT, 0 otherwise
                            # shape = (batchsize, n_constit)
    
    ptp_eta  = np.ptp(batch[:,1,:], axis=-1, keepdims=True) # ptp = 'peak to peak' = max - min
    ptp_phi  = np.ptp(batch[:,2,:], axis=-1, keepdims=True) # ptp = 'peak to peak' = max - min
    
    low_eta  = -width*ptp_eta
    high_eta = +width*ptp_eta
    
    low_phi  = np.maximum(-width*ptp_phi, -np.pi-np.amin(batch[:,2,:], axis=1).reshape(ptp_phi.shape))
    high_phi = np.minimum(+width*ptp_phi, +np.pi-np.amax(batch[:,2,:], axis=1).reshape(ptp_phi.shape))
    
    shift_eta = mask*np.random.uniform(low=low_eta, high=high_eta, size=(batch.shape[0], 1))
    shift_phi = mask*np.random.uniform(low=low_phi, high=high_phi, size=(batch.shape[0], 1))
    
    # (batchsize, 3, n_constit)
    shift = np.stack([np.zeros((batch.shape[0], batch.shape[2])), shift_eta, shift_phi], 1)

    shifted_batch = batch+shift
    return shifted_batch


# function to rotate jet anti-clockwise in the eta-phi plane
def rotate_jet_vectorized(batch):
    '''
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of eta-phi rotated jets, same shape as input
    '''
    rot_angle = np.random.rand(batch.shape[0])*2*np.pi
    c = np.cos(rot_angle)
    s = np.sin(rot_angle)
    o = np.ones_like(rot_angle)
    z = np.zeros_like(rot_angle)
    rot_matrix = np.array([[o, z, z], [z, c, -s], [z, s, c]]) # (3, 3, batchsize)
    return np.einsum('ijk,lji->ilk', batch, rot_matrix)

# function to normalise the pT's of the jet so that they sum to one
def normalise_pt_vectorized(batch):
    batch_norm = batch.copy()
    batch_norm[:,0,:] = np.nan_to_num(batch_norm[:,0,:]/np.sum(batch_norm[:,0,:], axis=1)[:, np.newaxis], posinf = 0.0, neginf = 0.0 )
    return batch_norm

# function to re-scale jet pTs by a single overall factor
def rescale_pt_vectorized(batch):
    batch_rscl = batch.copy()
    batch_rscl[:,0,:] = np.nan_to_num(batch_rscl[:,0,:]/600, posinf = 0.0, neginf = 0.0 )
    return batch_rscl

# function to crop jets
def crop_jet_vectorized(batch, nc):
    batch_crop = batch.copy()
    return batch_crop[:,:,0:nc]

# function to distort jet consituent positions as a function of pT
def distort_jet_vectorized(batch, strength=0.1, pT_clip_min=0.01):
    '''
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of eta-phi translated jets, translation drawn from normal with mean 0, std strength/pT, same shape as input
    pT is clipped to pT_clip_min before dividing
    the default strength value assumes that the constituents are not pT normalised
    '''
    pT = batch[:,0]   # (batchsize, n_constit)
    shift_eta = np.nan_to_num( strength * np.random.randn(batch.shape[0], batch.shape[2]) / np.maximum( np.abs(pT), pT_clip_min ), posinf = 0.0, neginf = 0.0 )
    shift_phi = np.nan_to_num( strength * np.random.randn(batch.shape[0], batch.shape[2]) / np.maximum( np.abs(pT), pT_clip_min ), posinf = 0.0, neginf = 0.0 )
    shift = np.stack([np.zeros((batch.shape[0], batch.shape[2])), shift_eta, shift_phi], 1)     # (batchsize, 3, n_constit)
    return batch + shift

# function to fill zero-padded elements of jets with soft constituents
def fill_jet_vectorized(batch, scales=(0.1, 0.3, 0.3)):
    '''
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of jets with all constituents filled with soft noise
            centered around zero with standard deviation given in scales,
            pT is normed after sampling the normal distribution to avoid negative transverse impulse
            nonzero constituents are NOT altered
    '''
    anti_mask = (batch[:,0] == 0)   # 1 for constituents with zero pT, 0 otherwise
                                    # shape = (batchsize, n_constit)
    soft_batch = np.zeros_like(batch)
    soft_batch[:, 0, :] = np.abs(np.random.normal(0, scales[0], size=(soft_batch.shape[0], soft_batch.shape[2])))*anti_mask
    soft_batch[:, 1, :] =        np.random.normal(0, scales[1], size=(soft_batch.shape[0], soft_batch.shape[2]))*anti_mask
    soft_batch[:, 2, :] =        np.random.normal(0, scales[2], size=(soft_batch.shape[0], soft_batch.shape[2]))*anti_mask

    return batch + soft_batch

# function to randomly drop constituents from the jets
def drop_constituents_vectorized(batch, ratio=0.05):
    '''
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of jets where any constituent has a chance of 'ratio' that its pT is dropped to zero.
    '''
    mask = np.stack([np.random.random((batch.shape[0], batch.shape[2]))>ratio, np.ones(shape=(batch.shape[0], batch.shape[2])), np.ones(shape=(batch.shape[0], batch.shape[2]))], 1)
    return batch * mask


