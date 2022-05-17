import numpy as np
from sklearn import metrics


def calc_auc_and_macSIC(fpr, tpr):
    
    auc = metrics.auc(fpr, tpr)

    SIC = tpr/np.sqrt(fpr)
    finite_SIC = SIC[np.isfinite(SIC)]
    maxSIC = np.max(finite_SIC)
    
    return auc, maxSIC


def calc_FPR_at_TPR(fpr, tpr, fixed_tpr = 0.5):
    
    dist_from_fixed_tpr = np.abs(tpr - fixed_tpr)
    min_dist_ind = np.where(dist_from_fixed_tpr == np.min(dist_from_fixed_tpr))[0][0]
    
    return fpr[min_dist_ind]
    
   
    
    