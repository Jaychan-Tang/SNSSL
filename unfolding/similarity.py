from skimage.measure import regionprops
import cv2
import numpy as np
from slic import GLCM_Feature
from unfolding.ciede2000 import CIEDE2000
from tqdm.contrib import tenumerate
import networkx as nx
# from unfolding.create_txt import read_similarity

def similarity_feature(img_bgr, segments):
    print('similarity_feature')
    gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lab_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    regions = regionprops(segments, gray_img)
    Superpixels_avg_lab = []
    Superpixels_glcm = []
    
    for i, props in tenumerate(regions):
        # 超像素内的所有点坐标
        coor = props['Coordinates']
        c_row = coor[:, 0:1].reshape(1, -1)[0]
        c_col = coor[:, 1:2].reshape(1, -1)[0]

        sp_lab = lab_img[c_row, c_col]
        avg_l = np.average(sp_lab[:, 0])
        avg_a = np.average(sp_lab[:, 1])
        avg_b = np.average(sp_lab[:, 2])
        Superpixels_avg_lab.append([avg_l, avg_a, avg_b])

        gray_array = props.intensity_image
        feature_name = ['correlation', 'contrast', 'ASM']
        texture_feature = GLCM_Feature(gray_array, feature_name, [0, np.pi/4, np.pi/2, np.pi*3/4])
        # print([texture_feature[0], texture_feature[1]])
        texture_feature = texture_feature.reshape(1, -1)[0]
        Superpixels_glcm.append([texture_feature[0], texture_feature[1], texture_feature[2]])

    return Superpixels_avg_lab, Superpixels_glcm


def compute_similarity(avg_lab1, correlation1, contrast1, asm1, avg_lab2, correlation2, contrast2, asm2):
    dE_00 = CIEDE2000(avg_lab1, avg_lab2)**2
    dCorrelation = np.sum((correlation1 - correlation2) ** 2)
    dContrast = np.sum((contrast1 - contrast2) ** 2)
    dASM = np.sum((asm1 - asm2) ** 2)
    delta = np.sqrt(dE_00 + dCorrelation * 10 + dContrast * 10 + dASM * 10)
    # print(dE_00**0.5, dCorrelation**0.5, dContrast**0.5, dASM**0.5)
    return delta


def find_edge_weight(graph, idx1, idx2):
    if idx1 == idx2:
        return 0
    for g in graph:
        idx = [g[0], g[1]]
        if idx1 in idx and idx2 in idx:
            return g[3]
    return 99999

def find_similarity_superpixels(segments, neighbor, idx, graph):
    stack = []
    merge_idx = []
    seen = set()
    seen_dis = set()
    stack.append(idx)
    seen.add(idx)
    seen_dis.add(0)
    
    while (len(stack) > 0):
        vertex = stack.pop(0)
        note = neighbor[vertex]
        # notes = neighbor[ver]-1
        # v_proba = graph[ver]
        for n in note:
            
            if n not in seen:
                stack.append(n)
                seen.add(n)
                dis = find_edge_weight(graph, vertex, n)
                if vertex == idx:
                    base_dis = 0
                else:
                    pass
                
                

    
    return merge_idx


def BFS_nstage(neighbor, idx, nstage):
    queue = []
    seen = set()
    queue.append(idx)  # 将任一个节点放入
    seen.add(idx)  # 同上

    # 第n层的超像素, 用list保存
    nstage_queue = [[] for i in range(nstage+1)]
    # print(nstage_queue)
    # 放入第0层节点
    nstage_queue[0].append(idx)
    # 用于记录当前层数
    n = 0
    # 图，边表
    graph = []
    
    while (len(nstage_queue[n]) > 0 and n < nstage):  # 当n层队列里还有东西,且小于设定层数时
        vertex = nstage_queue[n].pop(0)  # 第n层取出队头元素
        notes = neighbor[vertex]  # 查看neighbor里面的key,对应的邻接点

        for i in notes:  # 遍历邻接点
            # graph.append([vertex, i])
            if i not in seen:  # 如果该邻接点还没出现过
                nstage_queue[n+1].append(i)  # 存入下一层的queue
                seen.add(i)  # 存入集合
                # print(seen)
        
        # 如果这层已经没有元素，则进入下一层
        if len(nstage_queue[n]) == 0:
            n += 1
            # print(n)
    return seen


def similar_superpixels(center_id, nstage_superpixels_id, similarity_weight, similarity_t=10):
    # sim_path = r'D:\code\ActiveLearning\modAL-master\slic2_cache\floodnet\6651\6651similarity.txt'
    # similarity = read_similarity(sim_path)
    G = nx.Graph()
    selected_id = []
    for i, sim in enumerate(similarity_weight):
        if sim[0] in nstage_superpixels_id and sim[1] in nstage_superpixels_id:
            G.add_edge(sim[0], sim[1], weight=sim[2])
    
    for ns_id in nstage_superpixels_id:
        min_len = nx.bellman_ford_path_length(G, source=center_id, target=ns_id)
        if min_len < similarity_t:
            selected_id.append(ns_id)
    
    return selected_id
    
def similar_superpixels_global(img, segments, center_id, simfeature, similarity_t=10):
    # img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # avg_lab, glcm = similarity_feature(img_bgr, segments)
    print(simfeature.shape)
    selected_id = []
    center_lab = simfeature[center_id][0:3]
    center_glcm = simfeature[center_id][3:]
    for i, feature in enumerate(simfeature):
        if i != center_id:
            sim = compute_similarity(center_lab, center_glcm[0], center_glcm[1], center_glcm[2],
                                     feature[0:3], feature[3], feature[4], feature[5])
            if sim < similarity_t:
                selected_id.append(i)
    return selected_id
    



def similar_sp_around_query(neighbor, query_id, similarity_weight, nstage=4, similarity_t=9):
    nstage_superpixels_id = BFS_nstage(neighbor, query_id, nstage)
    nstage_superpixels_id.add(query_id)
    selected_id = similar_superpixels(query_id, nstage_superpixels_id, similarity_weight, similarity_t)
    return selected_id

if __name__ == "__main__":
    pass