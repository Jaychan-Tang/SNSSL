import numpy as np
import skimage.io as io
import cv2
from unfolding.similarity import similarity_feature, compute_similarity
from slic import find_neighbor
from tqdm.contrib import tenumerate

def similarity_txt(segments, img_rgb, neighbor, txt_path):
    # segments = np.load(r'../slic2_cache/floodnet/6651/zeroslic_50_scaler.npy')
    # img = io.imread(r'E:\donwload\FloodNet\train-org-img\6651.jpg')
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    Superpixels_avg_lab, Superpixels_glcm = similarity_feature(img_bgr, segments)
    # neighbor_path = r'D:\code\ActiveLearning\modAL-master\slic2_cache\floodnet\6651\6651neighbor.txt'
    # neighbor = read_neighbor(neighbor_path)
    # neighbor = find_neighbor(segments)
    # txt_path = r'D:\code\ActiveLearning\modAL-master\slic2_cache\floodnet\6651\6651similarity.txt'
    txt = open(txt_path, 'a+')
    print('create graph')
    for i, n_sp in tenumerate(neighbor):
        sp_id = i
        for n in n_sp:
            avg_lab1 = Superpixels_avg_lab[i]
            correlation1 = Superpixels_glcm[i][0]
            contrast1 = Superpixels_glcm[i][1]
            asm1 = Superpixels_glcm[i][2]
    
            avg_lab2 = Superpixels_avg_lab[n]
            correlation2 = Superpixels_glcm[n][0]
            contrast2 = Superpixels_glcm[n][1]
            asm2 = Superpixels_glcm[n][2]
            
            
            
            delta = compute_similarity(avg_lab1, correlation1, contrast1, asm1, avg_lab2, correlation2, contrast2, asm2)
            line = str(i) + ' ' + str(n) + ' ' + str(delta) + '\n'
            txt.write(line)
    txt.close()

def neighbor_txt(txt_path, segments):
    # segments = np.load(r'../slic2_cache/floodnet/6651/zeroslic_50_scaler.npy')
    neighbor = find_neighbor(segments)
    # txt_path = r'D:\code\ActiveLearning\modAL-master\slic2_cache\floodnet\6651\6651neighbor.txt'
    txt = open(txt_path, 'w+')

    for superpixels in neighbor:
        line = str()
        for i, s in enumerate(superpixels):
            if i == 0:
                line += str(s)
            else:
                line += ' ' + str(s)
        line += '\n'
        # print(line)
        txt.write(line)
    txt.close()

def save_neighbor(txt_path, neighbor):
    txt = open(txt_path, 'w+')
    
    for superpixels in neighbor:
        line = str()
        for i, s in enumerate(superpixels):
            if i == 0:
                line += str(s)
            else:
                line += ' ' + str(s)
        line += '\n'
        # print(line)
        txt.write(line)
    txt.close()

def read_neighbor(txt_path):
    # txt_path = 'slic2_cache/floodnet/6651/6651neighbor.txt'
    txt = open(txt_path, 'r')
    neighbor = []
    for line in txt.readlines():
        sp = line.rstrip('\n')
        # print(sp)
        # sp = sp.rstrip(' ')
        sp = sp.split(' ')
        # print(sp)
        neighbor.append([int(s) for s in sp])
    # print(neighbor)
    txt.close()
    return neighbor
    
def read_similarity(txt_path):
    # txt_path = r'D:\code\ActiveLearning\modAL-master\slic2_cache\floodnet\6651\6651similarity.txt'
    txt = open(txt_path, 'r+')
    similarity = []
    for line in txt.readlines():
        sp = line.rstrip('\n')
        # print(sp)
        # sp = sp.rstrip(' ')
        sp = sp.split(' ')
        similarity.append([int(sp[0]), int(sp[1]), float(sp[2])])
    return similarity

def sim_feature_save(img_rgb, segments, path):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    Superpixels_avg_lab, Superpixels_glcm = similarity_feature(img_bgr, segments)
    sim_feature = np.concatenate((Superpixels_avg_lab, Superpixels_glcm), axis=1)
    print(sim_feature.shape)
    np.save(path, sim_feature)
    
    
if __name__ == "__main__":
    # neighbor_txt(r'D:\code\ActiveLearning\modAL-master\slic2_cache\floodnet\6651\6651neighbor_50.txt')
    img = io.imread(r'E:\donwload\FloodNet\train-org-img\6651.jpg')
    label = io.imread(r"E:\donwload\FloodNet\train-label-img\6651_lab.png")
    segments = np.load('../slic2_cache/floodnet/6651/zeroslic_40_scaler.npy')
    similarity_path = r'D:\code\ActiveLearning\modAL-master\slic2_cache\floodnet\6651\simfeature_40.npy'
    sim_feature_save(img, segments, similarity_path)
    