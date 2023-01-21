from skimage.segmentation import slic, mark_boundaries, find_boundaries
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange
from tqdm.contrib import tenumerate, tzip
from skimage.measure import regionprops
from skimage.color import rgb2gray, rgb2xyz, xyz2lab
from skimage.feature import greycomatrix, greycoprops, local_binary_pattern
import skimage
import cv2
from sklearn.preprocessing import MinMaxScaler
import copy
import time
import os
import math
from scipy import ndimage as ndi
from skimage.morphology import dilation, square
import _thread
# def Superpixels_RGB(img, segments):
#     img = np.asarray(img)
#     # print(img[:, :, 0].shape)
#     segments = np.asarray(segments)
#     # print(segments.max())
#     superpixels = [[]] * segments.max()
#     # print(superpixels)
#     img_R = img[:, :, 0]
#     img_G = img[:, :, 1]
#     img_B = img[:, :, 2]
#     segments_R = list()
#     segments_G = list()
#     segments_B = list()
#     seg_num = segments.max()
#     for i in trange(seg_num):
#         pixel_index = np.where(segments == i)
#         R = img_R[pixel_index]
#         G = img_G[pixel_index]
#         B = img_B[pixel_index]
#         segments_R.append(R)
#         segments_G.append(G)
#         segments_B.append(B)
#     # 返回 所有的超像素的R[单个超像素的R]
#     return segments_R, segments_G, segments_B

# def Superpixels2Rectangle(img, segments):  # 返回外接矩形的坐标
#     # rec = [[0, 0, 0, 0]] * (segments.max() + 1 )
#     rec = np.zeros((segments.max() + 1, 4), dtype=np.int32)
#     rec[:, 2:] = max(img.shape) + 1
#     # rec.shape = [Xmin, Ymin, Xmax, Ymax] * seg_num
#     # print(rec.shape)
#     for i in range(segments.shape[0]):
#         for j in range(segments.shape[1]):
#             seg_idx = segments[i][j]
#             if rec[seg_idx][0] < i:
#                 rec[seg_idx][0] = i
#             if rec[seg_idx][2] > i:
#                 rec[seg_idx][2] = i
#
#             if rec[seg_idx][1] < j:
#                 rec[seg_idx][1] = j
#             if rec[seg_idx][3] < j:
#                 rec[seg_idx][3] = j
#
#     return rec

def PerSuperpixel(img, segments):
    for (i, segVal) in enumerate(np.unique(segments)):
        mask = np.zeros(img.shape[:2], dtype="uint8")
        mask[segments == segVal] = 255
    

def Adjacent_Superpixel(segments, rec):
    adjacent = []
    centers = []
    
    for i in range(rec.shape[0]):
        x = (rec[i][0] + rec[i][2]) // 2
        y = (rec[i][1] + rec[i][3]) // 2
        centers.append([x, y])


def Trans_Rectangle(img, segments, rec):
    pass
    seg_num = segments.max() + 1
    Superpixels_Rec = []
    for i in range(seg_num):
        Xmin, Ymin, Xmax, Ymax = rec[i]
        # 提取一个外接矩形
        temp = img[Xmin:Xmax+1, Ymin:Ymax+1]
        # 外接矩形中，属于超像素的像素值不变，其余都为NAN，便于GLCM计算
        sp_nan = np.argwhere(temp == i, temp, np.NAN)
        Superpixels_Rec.append(sp_nan)
    
    return Superpixels_Rec


def GLCM_Feature(gray_array,
                 feature_name=['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'],
                 angle=[0, np.pi/2]):


    glcm = greycomatrix(gray_array, [1], angle, 256, symmetric=True,
                        normed=True)  # , np.pi / 4, np.pi / 2, np.pi * 3 / 4
    
    # print(glcm.shape)
    texture_feature = []
    if 'entropy' in feature_name:
        feature_name.remove('entropy')
        P = copy.deepcopy(glcm)
        # normalize each GLCM
        P = P.astype(np.float64)
        glcm_sums = np.apply_over_axes(np.sum, P, axes=(0, 1))
        glcm_sums[glcm_sums == 0] = 1
        P /= glcm_sums
        entropy = np.apply_over_axes(np.sum, (P * -np.log(P)), axes=(0, 1))[0, 0]
        # print(entropy.shape)
        texture_feature.append(entropy[0])
    
    texture_feature = []
    for prop in feature_name:
        f = skimage.feature.greycoprops(glcm, prop)
        # print(f.shape)
        # temp=np.array(temp).reshape(-1)
        # values_temp.append(temp)
        texture_feature.append(f[0])
        # print(prop, f)
        # print('len:', len(f))
        # print('')
    
    
    return np.array(texture_feature)

def Superpixels_RGB_Rec(img, segments):
    img = np.asarray(img)

    segments = np.asarray(segments)
    segments_R = [[] for k in range(segments.max() + 1)]
    segments_G = segments_R
    segments_B = segments_R

    img_R = img[:, :, 0]
    img_G = img[:, :, 1]
    img_B = img[:, :, 2]

    # 超像素的最小外接矩形，后续计算GLCM
    # rec = np.zeros((segments.max() + 1, 4), dtype=np.int32)
    # rec[Xmin, Ymin, Xmax, Ymax]
    # rec[:, :2] = max(img.shape) + 1
    # print(rec)
    for i in trange(segments.shape[0]):
        for j in range(segments.shape[1]):
            seg_idx = segments[i][j]
            
            # 分离RGB
            segments_R[seg_idx].append(img_R[i][j])
            segments_G[seg_idx].append(img_G[i][j])
            segments_B[seg_idx].append(img_B[i][j])

            # 分离外接矩形
            # if rec[seg_idx][0] > i:
            #     rec[seg_idx][0] = i
            # if rec[seg_idx][1] > j:
            #     rec[seg_idx][1] = j
            # if rec[seg_idx][2] < i:
            #     rec[seg_idx][2] = i
            # if rec[seg_idx][3] < j:
            #     rec[seg_idx][3] = j
    # print(np.argwhere(rec > 5000))
    return segments_R, segments_G, segments_B
    

def Standard_Deviation(Superpixels_R, Superpixels_G, Superpixels_B):
    if not (len(Superpixels_R) and len(Superpixels_G) and len(Superpixels_B)):
        raise ValueError("长度不一致")
    else:
        segments_num = len(Superpixels_R)
    # Superpixels_R, Superpixels_G, Superpixels_B = Superpixels_RGB(img, segments)
    Superpixels_std = list()
    # print(Superpixels_R)
    for i in range(segments_num):
        std_R = np.std(Superpixels_R[i])
        std_G = np.std(Superpixels_G[i])
        std_B = np.std(Superpixels_B[i])
        Superpixels_std.append([std_R, std_G, std_B])
    return Superpixels_std

def Brightness(Superpixels_R, Superpixels_G, Superpixels_B):
    Superpixels_brightness = list()
    if not (len(Superpixels_R) and len(Superpixels_G) and len(Superpixels_B)):
        raise ValueError("长度不一致")
    else:
        segments_num = len(Superpixels_R)
    for i in range(segments_num):
        avg_R = np.average(Superpixels_R[i])
        avg_G = np.average(Superpixels_G[i])
        avg_B = np.average(Superpixels_B[i])
    
        Superpixels_brightness.append((avg_R+avg_G+avg_B)/3)
    
    return Superpixels_brightness

def adjoin_supperpixels_id(segments):
    print(segments.min(), segments.max())
    regions = regionprops(segments)
    adjoin_sp = []
    row, col = segments.shape[0], segments.shape[1]
    for i, props in enumerate(tqdm(regions)):
        # 超像素的外接圆，圆心坐标为相对值
        (CircleX, CircleY), radius = cv2.minEnclosingCircle(props['Coordinates'])
        # print("CircleX, CircleY",CircleX, CircleY)
        cx, cy = int(CircleX), int(CircleY)
        # print("cx,cy", cx, cy)
        if cx > row or cy > col:
            mask = np.zeros_like(segments)
            print(props['Coordinates'])
            mask[segments == i + 1] = 1
            plt.imshow(mask)
            plt.show()
    
        # 超像素外接圆半径
        # r = int(((cx-bbox[0])**2 + (cy-bbox[1])**2)**0.5)

        # 在若干倍外接圆半径范围内寻找邻接超像素,也可以为固定值
        rr = int(radius * 1.5)
        # 0至2pi, 8个方向, list初始化
        adjoin = [[]] * 8

        # 间隔pi/8记录一次
        for k in range(len(adjoin)):
            if k == 0:
                tmpy = cy - rr
                # print(cx, (tmpy > 0) and tmpy or 0)
                # print(segments[cx][(tmpy > 0) and tmpy or 0])
                adjoin[k] = segments[cx][(tmpy > 0) and tmpy or 0]
            if k == 1:
                tmpx = int(cx - rr // 1.414)
                tmpy = int(cy - rr // 1.414)
                adjoin[k] = segments[(tmpx > 0) and tmpx or 0][(tmpy > 0) and tmpy or 0]
            if k == 2:
                tmpx = cx - rr
                adjoin[k] = segments[(tmpx > 0) and tmpx or 0][cy]
            if k == 3:
                tmpx = int(cx - rr // 1.414)
                tmpy = int(cy + rr // 1.414)
                # print((tmpx > 0) and tmpx or 0)
                # print(col -1 )
                # print((tmpy >= col - 1) and col - 1 or tmpy)
                adjoin[k] = segments[(tmpx > 0) and tmpx or 0][(tmpy > col - 1) and col - 1 or tmpy]
            if k == 4:
                tmpy = cy + rr
                adjoin[k] = segments[cx][(tmpy > col - 1) and col - 1 or tmpy]
            if k == 5:
                tmpx = int(cx + rr // 1.414)
                tmpy = int(cy + rr // 1.414)
                adjoin[k] = segments[(tmpx > row - 1) and row - 1 or tmpx][(tmpy > col - 1) and col - 1 or tmpy]
            if k == 6:
                tmpx = cx + rr
                adjoin[k] = segments[(tmpx > row - 1) and row - 1 or tmpx][cy]
            if k == 7:
                tmpx = int(cx + rr // 1.414)
                tmpy = int(cy - rr // 1.414)
                adjoin[k] = segments[(tmpx > row - 1) and row - 1 or tmpx][(tmpy > 0) and tmpy or 0]

        adjoin = np.array(adjoin)
        # print(adjoin)
        # adjoin = np.where(adjoin == props.label, props.label, adjoin)
        adjoin_sp.append(adjoin)
    return adjoin_sp
    
    

def slic_process(img, mask=None, n_segments=100, compactness=10, show_img=True, use_irregular_GLCM = True, start_label = 1):
    # img = io.imread(img_path)
    # print(img)
    
    feature_name = ['homogeneity', 'correlation']
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Luv_img = cv2.cvtColor(img, cv2.COLOR_BGR2Luv)
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    r = 3  # 邻域半径
    p = 8 * r  # 邻域采样点数量
    uniformLBP_img = local_binary_pattern(gray_img, p, r, method="uniform")
    

    # segments = slic(img, n_segments=n_segments, compactness=compactness, convert2lab=True, start_label=start_label, mask=mask, slic_zero=True)

    slic0 = cv2.ximgproc.createSuperpixelSLIC(img, region_size=20, algorithm=cv2.ximgproc.SLICO)
    
    slic0.iterate(10)  # 迭代次数，越大效果越好
    slic0.enforceLabelConnectivity()
    segments = slic0.getLabels()  # 获取超像素标签
    
    # slic_ = cv2.ximgproc.createSuperpixelSLIC(image=img, region_size=20)
    # slic_.iterate(10)
    if show_img == True:
        out = mark_boundaries(img, segments)
        # boundary = find_boundaries(segments)
        # outlines = dilation(boundary, square(1))
        # print(segments)
        plt.imshow(out)
        plt.show()
    row, col = segments.shape[0], segments.shape[1]
    print(row, col)
    regions = regionprops(segments, gray_img)
    # 旋转不变均匀LBP的超像素块
    regions_LBP = regionprops(segments, uniformLBP_img)

    print(segments.min(), segments.max())

    adjoin_sp = []
    Superpixels_texture = []
    Superpixels_avg = []
    Superpixels_std = []
    Superpixels_brightness = []
    Superpixels_lab = []
    Superpixels_luv = []

    for i, props in enumerate(tqdm(regions)):
        # count_ += 1
        # 超像素外接矩形的相对质心
        # cx, cy = int(props.local_centroid[0]), int(props.local_centroid[1])
        
        # 超像素的外接矩形
        bbox = props.bbox

        # 超像素的外接圆，圆心坐标为相对值
        (CircleX, CircleY), radius = cv2.minEnclosingCircle(props['Coordinates'])
        # print("CircleX, CircleY",CircleX, CircleY)
        cx, cy = int(CircleX), int(CircleY)
        # print("cx,cy", cx, cy)
        if cx > row or cy > col:
            mask = np.zeros_like(segments)
            # print(props['Coordinates'])
            mask[segments == i+1] = 1
            plt.imshow(mask)
            plt.show()
        
        
        # 超像素外接圆半径
        # r = int(((cx-bbox[0])**2 + (cy-bbox[1])**2)**0.5)
        
        # 在若干倍外接圆半径范围内寻找邻接超像素,也可以为固定值
        rr = int(radius * 1.5)
        # 0至2pi, 8个方向, list初始化
        adjoin = [[]] * 8
        
        # 间隔pi/8记录一次
        for k in range(len(adjoin)):
            if k == 0:
                tmpy = cy - rr
                # print(cx, (tmpy > 0) and tmpy or 0)
                # print(segments[cx][(tmpy > 0) and tmpy or 0])
                adjoin[k] = segments[cx][(tmpy > 0) and tmpy or 0]
            if k == 1:
                tmpx = int(cx - rr // 1.414)
                tmpy = int(cy - rr // 1.414)
                adjoin[k] = segments[(tmpx > 0) and tmpx or 0][(tmpy > 0) and tmpy or 0]
            if k == 2:
                tmpx = cx - rr
                adjoin[k] = segments[(tmpx > 0) and tmpx or 0][cy]
            if k == 3:
                tmpx = int(cx - rr // 1.414)
                tmpy = int(cy + rr // 1.414)
                # print((tmpx > 0) and tmpx or 0)
                # print(col -1 )
                # print((tmpy >= col - 1) and col - 1 or tmpy)
                adjoin[k] = segments[(tmpx > 0) and tmpx or 0][(tmpy > col - 1) and col - 1 or tmpy]
            if k == 4:
                tmpy = cy + rr
                adjoin[k] = segments[cx][(tmpy > col - 1) and col - 1 or tmpy]
            if k == 5:
                tmpx = int(cx + rr // 1.414)
                tmpy = int(cy + rr // 1.414)
                adjoin[k] = segments[(tmpx > row - 1) and row - 1 or tmpx][(tmpy > col - 1) and col - 1 or tmpy]
            if k == 6:
                tmpx = cx + rr
                adjoin[k] = segments[(tmpx > row - 1) and row - 1 or tmpx][cy]
            if k == 7:
                tmpx = int(cx + rr // 1.414)
                tmpy = int(cy - rr // 1.414)
                adjoin[k] = segments[(tmpx > row - 1) and row - 1 or tmpx][(tmpy > 0) and tmpy or 0]
                
        
        adjoin = np.array(adjoin)
        adjoin = np.where(adjoin == props.label, 0, adjoin)
        # print(adjoin)
        adjoin_sp.append(adjoin)

        
        
        # 超像素内的所有点坐标
        coor = props['Coordinates']
        # print(coor.shape)
        # print(props.intensity_image.shape)
        # exit(-2)
        # coor = props.coor
        c_row = coor[:, 0:1].reshape(1, -1)[0]
        c_col = coor[:, 1:2].reshape(1, -1)[0]
 
        # 按segments的id顺序，计算每个超像素的特征信息
        sp_bgr = img[c_row, c_col]
        R = sp_bgr[:, 2]
        G = sp_bgr[:, 1]
        B = sp_bgr[:, 0]
        bright_R = np.average(R)
        bright_G = np.average(G)
        bright_B = np.average(B)
        Superpixels_brightness.append([bright_R, bright_G, bright_B])

        sp_lab = lab_img[c_row, c_col]
        avg_l = np.average(sp_lab[:, 0])
        avg_a = np.average(sp_lab[:, 1])
        avg_b = np.average(sp_lab[:, 2])
        Superpixels_lab.append([avg_l, avg_a, avg_b])
        # print(lab.shape)
        
        sp_luv = lab_img[c_row, c_col]
        avg_l = np.average(sp_luv[:, 0])
        avg_u = np.average(sp_luv[:, 1])
        avg_v = np.average(sp_luv[:, 2])
        Superpixels_luv.append([avg_l, avg_u, avg_v])

        # avg_R = np.average(R)
        # avg_G = np.average(G)
        # avg_B = np.average(B)
        Superpixels_avg.append(np.average(sp_bgr))
        
        std_R = np.std(R)
        std_G = np.std(G)
        std_B = np.std(B)
        Superpixels_std.append([std_R, std_G, std_B])
        
        

        
        if use_irregular_GLCM == True:
            # 超像素的外接矩形，不属于目标超像素的像素点为0
            gray_array = props.intensity_image
        else:
            gray_array = props.image
        texture_feature = GLCM_Feature(gray_array, feature_name)
        # texture_feature = GLCM_Feature(gray_array)
        Superpixels_texture.append(texture_feature)
        
        
        '''glcm = greycomatrix(gray_array, [1], [0, np.pi / 2], 256, symmetric=True,
                            normed=True)  # , np.pi / 4, np.pi / 2, np.pi * 3 / 4
        texture_feature = []
        for prop in feature_name:
            f = greycoprops(glcm, prop)
            # temp=np.array(temp).reshape(-1)
            # values_temp.append(temp)
            texture_feature.append(f[0])
            # print(prop, f)
            # print('len:', len(f))
            # print('')
        Superpixels_texture.append(texture_feature)'''
    
    
    # return np.array(Superpixels_std), np.array(Superpixels_brightness), np.array(Superpixels_texture), np.array(Superpixels_avg),\
    #        adjoin_sp, segments, np.array(Superpixels_lab)
    
    return np.array(Superpixels_std), np.array(Superpixels_brightness), np.array(Superpixels_texture), np.array(Superpixels_avg),\
           adjoin_sp, segments, np.array(Superpixels_lab)


def slic_process2(img, show_img=False, use_irregular_GLCM=True, region_size=20):
    feature_name = ['homogeneity', 'correlation']
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    r = 3  # 邻域半径
    p = 8 * r  # 邻域采样点数量
    uniformLBP_img = local_binary_pattern(gray_img, p, r, method="uniform")
   
    slic0 = cv2.ximgproc.createSuperpixelSLIC(img, region_size=region_size, algorithm=cv2.ximgproc.SLICO)
    
    slic0.iterate(10)  # 迭代次数
    slic0.enforceLabelConnectivity()
    segments = slic0.getLabels()  # 获取超像素标签
    segments = segments+1

   
    if show_img == True:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = mark_boundaries(img, segments)
        # boundary = find_boundaries(segments)
        # outlines = dilation(boundary, square(1))
        # print(segments)
        plt.imshow(out)
        plt.show()
    row, col = segments.shape[0], segments.shape[1]
    print(row, col)
    regions = regionprops(segments, gray_img)
    # 旋转不变均匀LBP的超像素块
    regions_LBP = regionprops(segments, uniformLBP_img)
    
    print(segments.min(), segments.max())
    
    # adjoin_sp = []
    Superpixels_glcm = []
    Superpixels_lab = []
    Superpixels_lbp = []
    Superpixels_rgb = []
    Superpixels_context = []
    avg_lab = []
    avg_rgb = []
    
    print('颜色纹理信息：')
    for i, (props, props_LBP) in enumerate(tzip(regions, regions_LBP)):
        # 超像素的外接圆，圆心坐标为相对值
        (CircleX, CircleY), radius = cv2.minEnclosingCircle(props['Coordinates'])
        # print("CircleX, CircleY",CircleX, CircleY)
        cx, cy = int(CircleX), int(CircleY)
        # print(segments[cx, cy])
        # if i==0 or i==49663:
        #     print(segments[cx][cy])
        # print("cx,cy", cx, cy)
        
        if cx > row or cy > col:
            mask = np.zeros_like(segments)
            # print(props['Coordinates'])
            mask[segments == i + 1] = 1
            plt.imshow(mask)
            plt.show()
        
        # 超像素外接圆半径
        # r = int(((cx-bbox[0])**2 + (cy-bbox[1])**2)**0.5)
        '''
        # 在若干倍外接圆半径范围内寻找邻接超像素,也可以为固定值
        rr = int(radius * 1.5)
        # 0至2pi, 8个方向, list初始化
        adjoin = [list() for t in range(8)]
        
        # 间隔pi/8记录一次
        for k in range(len(adjoin)):
            if k == 0:
                tmpy = cy - rr
                # print(cx, (tmpy > 0) and tmpy or 0)
                # print(segments[cx][(tmpy > 0) and tmpy or 0])
                adjoin[k] = segments[cx][(tmpy > 0) and tmpy or 0]
            if k == 1:
                tmpx = int(cx - rr // 1.414)
                tmpy = int(cy - rr // 1.414)
                adjoin[k] = segments[(tmpx > 0) and tmpx or 0][(tmpy > 0) and tmpy or 0]
            if k == 2:
                tmpx = cx - rr
                adjoin[k] = segments[(tmpx > 0) and tmpx or 0][cy]
            if k == 3:
                tmpx = int(cx - rr // 1.414)
                tmpy = int(cy + rr // 1.414)
                # print((tmpx > 0) and tmpx or 0)
                # print(col -1 )
                # print((tmpy >= col - 1) and col - 1 or tmpy)
                adjoin[k] = segments[(tmpx > 0) and tmpx or 0][(tmpy > col - 1) and col - 1 or tmpy]
            if k == 4:
                tmpy = cy + rr
                adjoin[k] = segments[cx][(tmpy > col - 1) and col - 1 or tmpy]
            if k == 5:
                tmpx = int(cx + rr // 1.414)
                tmpy = int(cy + rr // 1.414)
                adjoin[k] = segments[(tmpx > row - 1) and row - 1 or tmpx][(tmpy > col - 1) and col - 1 or tmpy]
            if k == 6:
                tmpx = cx + rr
                adjoin[k] = segments[(tmpx > row - 1) and row - 1 or tmpx][cy]
            if k == 7:
                tmpx = int(cx + rr // 1.414)
                tmpy = int(cy - rr // 1.414)
                adjoin[k] = segments[(tmpx > row - 1) and row - 1 or tmpx][(tmpy > 0) and tmpy or 0]
        
        adjoin = np.array(adjoin)
        # adjoin = np.where(adjoin == props.label, props.label, adjoin)
        adjoin_sp.append(adjoin)'''
        
        
        # 超像素内的所有点坐标
        coor = props['Coordinates']
        c_row = coor[:, 0:1].reshape(1, -1)[0]
        c_col = coor[:, 1:2].reshape(1, -1)[0]
        
        # 按segments的id顺序，计算每个超像素的特征信息
        
        # LAB直方图
        regions_pixels = lab_img[c_row, c_col]
        l_pixels = regions_pixels[:, 0]
        a_pixels = regions_pixels[:, 1]
        b_pixels = regions_pixels[:, 2]
        l_hist, _ = np.histogram(l_pixels, range=(0, 256), bins=16)
        l_hist = single_feature_scaler(l_hist)
        a_hist, _ = np.histogram(a_pixels, range=(0, 256), bins=16)
        a_hist = single_feature_scaler(a_hist)
        b_hist, _ = np.histogram(b_pixels, range=(0, 256), bins=16)
        b_hist = single_feature_scaler(b_hist)
        Superpixels_lab.append(np.concatenate((l_hist, a_hist, b_hist), axis=0))
        
        # RGB直方图
        regions_pixels = img[c_row, c_col]
        b_pixels = regions_pixels[:, 0]
        g_pixels = regions_pixels[:, 1]
        r_pixels = regions_pixels[:, 2]
        b_hist, _ = np.histogram(b_pixels, range=(0, 256), bins=16)
        g_hist, _ = np.histogram(g_pixels, range=(0, 256), bins=16)
        r_hist, _ = np.histogram(r_pixels, range=(0, 256), bins=16)
        b_hist = single_feature_scaler(b_hist)
        g_hist = single_feature_scaler(g_hist)
        r_hist = single_feature_scaler(r_hist)
        Superpixels_rgb.append(np.concatenate((b_hist, g_hist, r_hist), axis=0))
        
        # LBP直方图
        coor = props_LBP['Coordinates']
        c_row = coor[:, 0:1].reshape(1, -1)[0]
        c_col = coor[:, 1:2].reshape(1, -1)[0]
        regions_LBP = lab_img[c_row, c_col]
        lbp_hist, lbp_bins = np.histogram(regions_LBP)
        lbp_hist = single_feature_scaler(lbp_hist)
        Superpixels_lbp.append(lbp_hist)
        
        #平均lab
        sp_lab = lab_img[c_row, c_col]
        avg_l = np.average(sp_lab[:, 0])
        avg_a = np.average(sp_lab[:, 1])
        avg_b = np.average(sp_lab[:, 2])
        avg_lab.append([avg_l, avg_a, avg_b])

        # 平均lab
        sp_bgr = img[c_row, c_col]
        avg_b = np.average(sp_lab[:, 0])
        avg_g = np.average(sp_lab[:, 1])
        avg_r = np.average(sp_lab[:, 2])
        avg_rgb.append([avg_r, avg_g, avg_b])
        
        if use_irregular_GLCM == True:
            # 超像素的外接矩形，不属于目标超像素的像素点为0
            gray_array = props.intensity_image
        else:
            gray_array = props.image
        texture_feature = GLCM_Feature(gray_array, feature_name)
        texture_feature = texture_feature.reshape(1, -1)[0]
        texture_feature = single_feature_scaler(texture_feature)
        Superpixels_glcm.append(texture_feature)


    Superpixels_lab = np.array(Superpixels_lab)
    Superpixels_lbp = np.array(Superpixels_lbp)
    Superpixels_glcm = np.array(Superpixels_glcm)
    Superpixels_rgb = np.array(Superpixels_rgb)
    avg_lab = np.array(avg_lab)
    avg_rgb = np.array(avg_rgb)
    print(Superpixels_glcm.shape)
    print(Superpixels_rgb.shape)
    print('上下文信息：')
    adjoin_sp = find_neighbor(segments)
    # adjoin_sp = np.array(adjoin_sp)
    for i, ad in enumerate(tqdm(adjoin_sp)):
        # print(ad)
        ad = np.array(list(ad))
        # print(ad)
        ad_lab = Superpixels_lab[ad-1]
        # print(ad_lab)
        # ad_lbp = Superpixels_lbp[ad-1]
        # ad_rgb = Superpixels_rgb[ad-1]
        ad_glcm = Superpixels_glcm[ad-1]
        # ad_rgb = avg_rgb[ad-1]
        
        all_rgb = np.average(ad_lab, axis=0)
        all_glcm = np.average(ad_glcm, axis=0)
        # all_rgb = ad_rgb.reshape(1, -1)[0]
        # print(all_rgb)
        # all_glcm = ad_glcm.reshape(1, -1)[0]
        # print(all_rgb.shape)
        # exit(-2)
        
        
        Superpixels_context.append(np.concatenate((all_rgb, all_glcm), axis=0))
    
    X_raw = np.concatenate((np.array(Superpixels_lab), np.array(Superpixels_lbp),
                           np.array(Superpixels_glcm), np.array(Superpixels_context)), axis=1)
    
    
    scaler = MinMaxScaler()
    X_raw = scaler.fit_transform(X_raw)
    print(X_raw.shape)
    
    
    
    return X_raw, segments, adjoin_sp



def find_neighbor(segments):
    direction_x = np.array([-1, 0, 1, 0])
    direction_y = np.array([0, -1, 0, 1])
    adjoin_sp = [set() for i in range(segments.max())]
    # print(adjoin_sp)
    # adjoin_sp -= 1
    # print(adjoin_sp)
    max_x = segments.shape[0]
    max_y = segments.shape[1]
    for i in trange(max_x):
        for j in range(max_y):
            label = segments[i][j]
            x = i + direction_x
            y = j + direction_y
            center = [i, j]
            # print(x, y)
            
            for m in range(x.shape[0]):
                if x[m]>0 and y[m]>0 and x[m]<max_x and y[m]<max_y:
                    ix = x[m]
                    iy = y[m]
                    nlabel = segments[ix][iy]
                    # n = [ix, iy]
                    if nlabel != label:
                        adjoin_sp[label-1].add(nlabel-1)
    # for i in trange(len(adjoin_sp)):
    #     adjoin_sp[i] = set(adjoin_sp[i])
    
    return adjoin_sp
    
    

def RGBhistogram(clt):
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist


def LBP_feature(gray):
    from skimage.feature import local_binary_pattern
    
    # LBP算法中范围半径的取值
    r = 2  # 邻域半径
    p = 8 * r  # 邻域采样点数量
    
    # 基本 LBP，灰度不变
    basicLBP = local_binary_pattern(gray, 8, 1)
    # 圆形 LBP，扩展灰度和旋转不变
    circleLBP = local_binary_pattern(gray, p, r, method="ror")
    # 旋转不变 LBP
    invarLBP = local_binary_pattern(gray, p, r, method="var")
    # 等价 LBP，灰度和旋转不变
    uniformLBP = local_binary_pattern(gray, p, r, method="uniform")
    
def single_feature_scaler(feature):
    scaler = MinMaxScaler()
    feature = scaler.fit_transform(feature.reshape(-1, 1))
    feature = feature.reshape(1, -1)[0]
    return feature