import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

def roc_threshold(label, prediction):
    fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
    fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
    c_auc = roc_auc_score(label, prediction)
    return c_auc, threshold_optimal

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def eval_metric(oprob, label):

    auc, threshold = roc_threshold(label, oprob)

    prob = oprob > threshold

    label = label > threshold

    TP = np.sum(prob.ravel() & label.ravel(), axis=0)

    TN = np.sum((~prob.ravel()) & (~label.ravel()),axis=0)
    FP = np.sum(prob.ravel() & (~label.ravel()), axis=0)
    FN = np.sum((~prob.ravel()) & label.ravel(), axis=0)

    accuracy = np.mean(( TP + TN ) / ( TP + TN + FP + FN + 1e-12))
    precision = np.mean(TP / (TP + FP + 1e-12))
    recall = np.mean(TP / (TP + FN + 1e-12))
    specificity = np.mean ( TN / (TN + FP + 1e-12))
    F1 = 2*(precision * recall) / (precision + recall+1e-12)

    return accuracy, precision, recall, specificity, F1, auc
