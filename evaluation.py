import numpy as np
import glob
import tqdm
from PIL import Image
import cv2 as cv
import os
from sklearn.metrics import confusion_matrix,cohen_kappa_score
from skimage import io
from skimage import measure
from scipy import ndimage
from sklearn.metrics import f1_score, accuracy_score
from skimage.measure import regionprops


def mean_iou(input, target, classes):
    """  compute the value of mean iou
    :param input:  2d array, int, prediction
    :param target: 2d array, int, ground truth
    :param classes: list
    :return:
        miou: float, the value of miou
    """
    miou = 0
    for i in classes:
        intersection = np.logical_and(target == i, input == i)
        # print(intersection.any())
        union = np.logical_or(target == i, input == i)
        temp = np.sum(intersection) / np.sum(union)
        miou += temp
    return  miou/len(classes)


def iou(input, target, classes=1):
    """  compute the value of iou
    :param input:  2d array, int, prediction
    :param target: 2d array, int, ground truth
    :param classes: int, the number of class
    :return:
        iou: float, the value of iou
    """
    intersection = np.logical_and(target == classes, input == classes)
    # print(intersection.any())
    union = np.logical_or(target == classes, input == classes)
    iou = np.sum(intersection) / np.sum(union)
    return iou


def compute_f1(prediction, target):
    """
    :param prediction: 2d array, int,
            estimated targets as returned by a classifier
    :param target: 2d array, int,
            ground truth
    :return:
        f1: float
    """
    prediction.tolist(), target.tolist()
    img, target = np.array(prediction).flatten(), np.array(target).flatten()
    f1 = f1_score(y_true=target, y_pred=img)
    return  f1


def compute_kappa(prediction, target):
    """
    :param prediction: 2d array, int,
            estimated targets as returned by a classifier
    :param target: 2d array, int,
            ground truth
    :return:
        kappa: float
    """
    prediction.tolist(), target.tolist()
    img, target = np.array(prediction).flatten(), np.array(target).flatten()
    kappa = cohen_kappa_score(target, img)
    return kappa


def compute_acc(gt, pred):
    matrix = confusion_matrix(y_true=np.array(gt).flatten(), y_pred=np.array(pred).flatten())
    acc = np.diag(matrix).sum() / matrix.sum()
    return acc

def compute_class_acc(gt, pred, classes):
    # aa = 0
    class_acc = []
    matrix = confusion_matrix(y_true=np.array(gt).flatten(), y_pred=np.array(pred).flatten())
    for i, cla in enumerate(classes):
        accuracy_score(y_true=np.array(gt).flatten(), y_pred=np.array(pred).flatten())
        
        # print(matrix.sum(axis=1).shape)
        # acc = matrix[i][i] / matrix.sum(axis=1)[i]
        # aa += acc
        intersection = np.logical_and(gt == cla, pred == cla)
        acc = np.sum(intersection) / np.sum(gt == cla)
        class_acc.append(acc)
    return class_acc

def class_accuracy(gt, pred, classes):
    cm = confusion_matrix(y_true=np.array(gt).flatten(), y_pred=np.array(pred).flatten())
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    ca = cm.diagonal()
    return ca

def compute_avg_acc(gt, pred, classes):
    aa = 0
    for i, cla in enumerate(classes):
        intersection = np.logical_and(gt == cla, pred == cla)
        # print(intersection.any())
        # union = np.logical_or(gt == i, pred == i)
        # print(np.sum(gt == cla))
        temp = np.sum(intersection) / np.sum(gt == cla)
        # print(temp)
        aa += temp
    return aa/len(classes)
    
    
def compute_recall(gt, pred):
    #  返回所有类别的召回率recall
    matrix = confusion_matrix(y_true=np.array(gt).flatten(), y_pred=np.array(pred).flatten())
    recall = np.diag(matrix) / matrix.sum(axis = 0)
    return recall


def superpixel_2darray(y, segments):
    regions = regionprops(segments)
    pred_img = np.zeros((segments.shape[0], segments.shape[1]))
    # mask = np.repeat(mask[..., np.newaxis], 3, 2)
    for i, props in enumerate(regions):
        coor = props['Coordinates']
        c_row = coor[:, 0:1].reshape(1, -1)[0]
        c_col = coor[:, 1:2].reshape(1, -1)[0]
        pred_img[c_row, c_col] = y[i]
    return pred_img