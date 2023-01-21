from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
from skimage.segmentation import slic, mark_boundaries, find_boundaries
from skimage.measure import regionprops
from tqdm import tqdm
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from collections import Counter
from train_potsdam import test_model
from evaluation import *
from sklearn.metrics import classification_report

def floodnet_label():
    img = io.imread(r'E:\donwload\FloodNet\test-org-img\7577.jpg')
    label = io.imread(r"E:\donwload\FloodNet\test-label-img\7577_lab.png")
    segments = np.load('../slic2_cache/floodnet/7577/zeroslic_35_scaler.npy')
    y_raw = np.load('../slic2_cache/floodnet/7577/zeroy_35_scaler.npy')
    y_count = Counter(y_raw)
    for (k, v) in y_count.items():
        print(k, v, v/len(y_raw))
    # exit(-2)
    label_c = set(label.reshape(1, -1)[0])
    print(label_c)
    label_count = [0] * 10
    out = np.zeros_like(img)
    color_list = [[0, 0, 0],  # 背景
                  [199, 91, 88],  # 暗红色，淹没的建筑
                  [255, 0, 0],  # 红色，建筑
                  [128, 128, 0],  # 棕色，淹没的道路
                  [128, 128, 128],  # 灰色，道路
                  [0, 255, 255],  # 青色，水
                  [0, 0, 255],  # 蓝色，树木
                  [255, 0, 255],  # 粉色，车辆
                  [255, 255, 0],  # 黄色，水池
                  [0, 255, 0]  # 绿色，草地
                  ]
    color_list = np.array(color_list)
    
    for i in trange(label.shape[0]):
        for j in range(label.shape[1]):
            p = label[i][j]
            # print(p)
            out[i][j] = color_list[p]
            label_count[p] += 1
    label_count = np.asarray(label_count)
    print(label_count)
    print(label_count/np.sum(label_count))
    plt.imshow(out)
    plt.show()
    # plt.imsave('slic2_cache/floodnet/7374/label.png', out)


def potsdam_label():
    # [0  4944599 15182061  2679388  5447007   313148  7433797        0
    #  0        0]
    # [0.         0.13734997 0.42172392 0.07442744 0.15130575 0.00869856
    #  0.20649436 0.         0.         0.]

    # [0  1890467 11428326  8780245  5128149   434615  8338198        0
    #  0        0]
    # [0.         0.05251297 0.3174535  0.24389569 0.14244858 0.01207264
    #  0.23161661 0.         0.         0.]
    img = io.imread(r'../data/img/top_potsdam_3_10_RGB.png')
    img = img[:, :, 0:3]
    label = io.imread(r"../data/label/potsdam_3_10_trans.png")
    # segments = np.load('../slic2_cache/floodnet/7577/zeroslic_35_scaler.npy')
    y_raw = np.load('../slic2_cache/potsdam/3_10/y_20.npy')
    y_count = Counter(y_raw)
    for (k, v) in y_count.items():
        print(k, v, v / len(y_raw))
    # exit(-2)
    label_c = set(label.reshape(1, -1)[0])
    print(label_c)
    label_count = [0] * 10
    out = np.zeros_like(img)
    class1 = np.array([255, 0, 0])  # 红色 其他（含沙地、不分类区域等）
    class2 = np.array([0, 255, 255])  # 青色 草地
    class3 = np.array([0, 255, 0])  # 绿色 树木
    class4 = np.array([0, 0, 255])  # 蓝色 建筑
    class5 = np.array([255, 255, 0])  # 黄色 车辆
    class6 = np.array([255, 255, 255])  # 白色 道路
    color_list = [[255, 0, 0],
                  [0, 255, 255],
                  [0, 255, 0],
                  [0, 0, 255],
                  [255, 255, 0],
                  [255, 255, 255]
                  ]
    color_list = np.array(color_list)
    
    for i in trange(label.shape[0]):
        for j in range(label.shape[1]):
            p = label[i][j]
            # print(p)
            out[i][j] = color_list[p-1]
            label_count[p] += 1
    label_count = np.asarray(label_count)
    print(label_count)
    print(label_count / np.sum(label_count))
    plt.imshow(out)
    plt.show()


def color2label(label):
    from tqdm import trange
    
    class1 = np.array([255, 0, 0])  # 红色 其他（含沙地、不分类区域等）
    class2 = np.array([0, 255, 255])  # 青色 草地
    class3 = np.array([0, 255, 0])  # 绿色 树木
    class4 = np.array([0, 0, 255])  # 蓝色 建筑
    class5 = np.array([255, 255, 0])  # 黄色 车辆
    class6 = np.array([255, 255, 255])  # 白色 道路
    
    label_copy = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)
    # mask = np.full((label.shape[0], label.shape[1]), True, dtype=np.bool)
    for i in trange(label.shape[0]):
        for j in range(label.shape[1]):
            tmp = label[i][j]
            # tmp = np.array([tmp[2], tmp[1], tmp[0]])
            
            # if (tmp != IGNORE).all():
            #     print(label[i][j])
            if (tmp == class1).all():
                label_copy[i][j] = 1
            elif (tmp == class2).all():
                label_copy[i][j] = 2
            elif (tmp == class3).all():
                label_copy[i][j] = 3
            elif (tmp == class4).all():
                label_copy[i][j] = 4
            elif (tmp == class5).all():
                label_copy[i][j] = 5
            elif (tmp == class6).all():
                label_copy[i][j] = 6
            # elif (tmp == IGNORE).all():
            #     mask[i][j] = False
            else:
                print(i, j)
                print(label[i][j])
                raise Exception

    return label_copy


def test_model(base_array, test_array, label, y_raw):
    classes = list(set(y_raw))
    # print(classes)
    
    # base_miou = mean_iou(base_array, label, classes)
    base_oa = compute_acc(gt=label, pred=base_array)
    base_aa = compute_avg_acc(gt=label, pred=base_array, classes=classes)
    base_kappa = compute_kappa(prediction=base_array, target=label)
    # base_class_acc = compute_class_acc(gt=label, pred=base_array, classes=classes)
    base_class_acc = class_accuracy(gt=label, pred=base_array, classes=classes)
    

    # test_miou = mean_iou(test_array, label, classes)
    test_oa = compute_acc(gt=label, pred=test_array)
    test_aa = compute_avg_acc(gt=label, pred=test_array, classes=classes)
    test_kappa = compute_kappa(prediction=test_array, target=label)
    # test_class_acc = compute_class_acc(gt=label, pred=test_array, classes=classes)
    test_class_acc = class_accuracy(gt=label, pred=test_array, classes=classes)
    # print('miou base/test:', base_miou, test_miou)
    report = classification_report(y_true=label.flatten(), y_pred=base_array.flatten(), digits=4)
    print(report)
    report = classification_report(y_true=label.flatten(), y_pred=test_array.flatten(), digits=4)
    print(report)
    print('overall_acc base/test:', base_oa, test_oa)
    print('avg_acc base/test:', base_aa, test_aa)
    print('kappa base/test:', base_kappa, test_kappa)
    print('acc_class base/test', base_class_acc, test_class_acc)
    print('base')
    print(base_class_acc)
    # for i, class_name in enumerate(classes):
    #     print(base_class_acc[i])
    print('test')
    print(test_class_acc)
    # for i, class_name in enumerate(classes):
    #     print(test_class_acc[i])
    # return [base_kappa], [test_kappa]
    return [base_oa, base_aa, base_kappa], [test_oa, test_aa, test_kappa]

def potsdam_array2color(array, name):
    img = io.imread(r'../data/img/top_potsdam_2_10_RGB.png')
    img = img[:, :, 0:3]
    # label = io.imread(r"../data/label/potsdam_2_10_trans.png")
    out = np.zeros_like(img)
    color_list = [[255, 0, 0],
                  [0, 255, 255],
                  [0, 255, 0],
                  [0, 0, 255],
                  [255, 255, 0],
                  [255, 255, 255]
                  ]
    color_list = np.array(color_list)
    for i in trange(array.shape[0]):
        for j in range(array.shape[1]):
            p = int(array[i][j])
            # print(p)
            out[i][j] = color_list[p-1]

    plt.imshow(out)
    plt.show()
    plt.imsave('../paper/data/potsdam/' + str(name) +'.png', out)
    

def floodnet_array2color(array, name):
    img = io.imread(r'E:\donwload\FloodNet\train-org-img\6651.jpg')
    # label = io.imread(r"E:\donwload\FloodNet\test-label-img\7577_lab.png")
    img = img[:, :, 0:3]

    out = np.zeros_like(img)
    color_list = [[0, 0, 0],  # 背景
                  [199, 91, 88],  # 暗红色，淹没的建筑
                  [255, 0, 0],  # 红色，建筑
                  [128, 128, 0],  # 棕色，淹没的道路
                  [128, 128, 128],  # 灰色，道路
                  [0, 255, 255],  # 青色，水
                  [0, 0, 255],  # 蓝色，树木
                  [255, 0, 255],  # 粉色，车辆
                  [255, 255, 0],  # 黄色，水池
                  [0, 255, 0]  # 绿色，草地
                  ]
    color_list = np.array(color_list)
    color_list = np.array(color_list)
    for i in trange(array.shape[0]):
        for j in range(array.shape[1]):
            p = int(array[i][j])
            # print(p)
            out[i][j] = color_list[p]

    plt.imshow(out)
    plt.show()
    plt.imsave('../paper/data/floodnet/' + str(name) + '.png', out)
  
if __name__ == "__main__":
    # base_img = io.imread('../paper/data/potsdam/2_10BT7-12.png')[:, :, 0:3]
    # test_img = io.imread('../paper/data/potsdam/2_10BTOURS7-12.png')[:, :, 0:3]
    # label = io.imread(r"../data/label/potsdam_2_10_trans.png")
    # y_raw = np.load('../slic2_cache/potsdam/2_10/y_20' + '.npy')
    # print(base_img.shape)
    # base_array = color2label(base_img)
    # test_array = color2label(test_img)
    # test_model(base_array, test_array, label)
    base_array = np.load('../slic2_cache/floodnet/6651/base_array_svm_random1.npy')
    test_array = np.load('../slic2_cache/floodnet/6651/test_array_svm_random1.npy')
    # print(base_array.shape)
    # plt.imshow(base_array)
    # plt.imshow(test_array)
    # plt.show()
    # plt.imsave('../paper/data/floodnet/7577_svm_base0.png', base_array)
    # plt.imsave('../paper/data/floodnet/7577_svm_snssl0.png', test_array)
    floodnet_array2color(base_array, '6651_svm_base1')
    floodnet_array2color(test_array, '6651_svm_snssl1')
    
    # potsdam_array2color(test_array, '3_10_MCLUOURS7-12-2')