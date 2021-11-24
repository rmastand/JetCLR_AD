import os
import sys
import numpy as np
import random
import time
from tqdm import tqdm

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from modules.utils import LRScheduler, EarlyStopping

class NeuralNet(nn.Module):
    def __init__(self, input_shape):
        super(NeuralNet, self).__init__()

        # Designed to ensure that adjacent pixels are either all 0s or all active
        # with an input probability
        self.dropout1 = nn.Dropout2d(0.25)

        # First fully connected layer
        self.fc1 = nn.Linear(input_shape, 64) # first size is output of flatten
        # Second fully connected layer that outputs our labels
        self.fc2 = nn.Linear(64, 1)

        
    # x represents our data
    def forward(self, x):

        x = torch.flatten(x, 1)
        # Pass data through fc1
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
    
        # Apply softmax to x
        #output = F.log_softmax(x, dim=1)
        output = torch.sigmoid(x) # for BCE 
        

        return output
    
    
def train_and_eval_nn(device, my_nn, num_epochs_nn, criterion, optimizer, early_stopping, lr_scheduler,
                      batch_size_nn, data_nn_train, labels_nn_train, 
                      data_nn_val, labels_nn_val,
                      data_nn_test, labels_nn_test, verbose = True, update = 10):
    
    """
    The data should already be processed!!!
    """

    def send_through_net(data, labels):
        # data, labels input are np arrays
        inputs = torch.from_numpy(data).float().to(device)
        labels = torch.from_numpy(labels).long().to(device)
        outputs = my_nn(inputs)
        loss = criterion(outputs, labels.reshape(-1,1).float()).to(device)
        return loss
    
    """
    Train the neural net
    """
    epochs = []
    losses = []

    epochs_val = []
    losses_val = []
    
    
    
    if verbose: print("Starting training...")
    
    my_nn.to(device) # make sure were using the gpu!
        
    for epoch in tqdm(range(num_epochs_nn)):  # loop over the dataset multiple times
        
        if epoch % update == 0:
            #if verbose: print("On epoch", epoch)
            
            """
            Overtraining check
            """
            with torch.no_grad():
                epochs_val.append(epoch)
                loss = send_through_net(data_nn_val, labels_nn_val)
                losses_val.append(loss.detach().cpu().numpy())
                
                # early stopping
                if early_stopping != "":
                    early_stopping(loss)
                    if early_stopping.early_stop:
                        break
                # LR rescheduling
                elif lr_scheduler != "":
                    lr_scheduler(loss)
                
        epochs.append(epoch)
        # make batches
        indices_list = torch.split(torch.randperm( data_nn_train.shape[0] ), batch_size_nn )
        local_losses = []
        
        for k, index in enumerate( indices_list ):
            batch_data = data_nn_train[index,:]
            batch_labels = labels_nn_train[index]

            # zero the parameter gradients
            optimizer.zero_grad()
            loss = send_through_net(batch_data, batch_labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            local_losses.append(loss.detach().cpu().numpy())
          
        losses.append(np.mean(local_losses))
        
    
    
    if verbose: print("Finished Training")
    
    """
    Evaluate the neural net
    """
 
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        inputs = torch.from_numpy(data_nn_test).float().to(device)
        outputs = my_nn(inputs).detach().cpu().numpy()
        predicted = np.round(outputs).reshape(labels_nn_test.size)
        # calculate auc 
        auc = roc_auc_score(labels_nn_test, outputs)
        
        fpr, tpr, _ = roc_curve(labels_nn_test, outputs)

        total = labels_nn_test.size
        correct = (predicted == labels_nn_test).sum()    
    
    performance_stats = {"epochs":epochs, "losses": losses, "val_epochs":epochs_val, "val_losses":losses_val, 
              "tpr":tpr, "fpr":fpr, "acc":correct / total , "auc":auc}
    
    return performance_stats


def create_and_run_nn(device, input_shape, num_epochs, batch_size, update_epochs, lr, 
                      data_train_nn, labels_train, 
                      data_val_nn, labels_lct_train,
                      data_test_nn, labels_lct_test, 
                      verbose, early_stop = True, LRschedule = False):
    
    """
    early_stop: stops training once validation loss stops improving 5 epochs in a row
    LRscheduleL halves LR once validation loss stops improving 5 epochs in a row; stops training once LR gets below a certain threshold
    """
    
    local_nn = NeuralNet(input_shape = input_shape)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(local_nn.parameters(), lr=lr)
    
    # early stopping / LR scheduler mechanism
    if early_stop:
        early_stopping = EarlyStopping()
        lr_scheduler = ""
    elif LRschedule:
        early_stopping = ""
        lr_scheduler = LRScheduler(optimizer)
    else:
        early_stopping = ""
        lr_scheduler = ""
    
    performance_stats = train_and_eval_nn(device, local_nn, num_epochs, criterion, optimizer, early_stopping, lr_scheduler,
                                          batch_size, data_train_nn, labels_train, 
                      data_val_nn, labels_lct_train,
                      data_test_nn, labels_lct_test, verbose = verbose, update = update_epochs)
    
    return performance_stats


