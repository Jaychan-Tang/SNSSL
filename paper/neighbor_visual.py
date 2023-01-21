import skimage.io as io
import numpy as np
from unfolding.create_txt import read_neighbor, read_similarity
from skimage.segmentation import slic, mark_boundaries, find_boundaries
from skimage.measure import regionprops
from tqdm import tqdm
import matplotlib.pyplot as plt
from unfolding.similarity import similar_sp_around_query, similar_superpixels_global


def draw_mask(img, segments, center, neighbor, name):
    mask = np.zeros((segments.shape[0], segments.shape[1]))
    mask = np.repeat(mask[..., np.newaxis], 3, 2)
    out = mark_boundaries(img, segments, mode='thick', color=[0.8, 0.8, 0])
    regions = regionprops(segments)
    color_center = np.array([32, 208, 255]) / 255
    color_neighbor = np.array([255, 85, 49]) / 255
    # [255, 85, 49] 亮红
    # [32, 208, 255] 亮青蓝
    # 127 255 0 黄绿色
    # [160, 32, 240] 紫色
    # [255, 97, 0] 橙色
    # [255, 215, 0] 金黄色
    for i, props in enumerate(tqdm(regions)):
        coordinates = props['Coordinates']
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        if i == center:
            out[x, y] = 0.25 * color_center + 0.75 * out[x, y]
        elif i in neighbor:
            out[x, y] = 0.25 * color_neighbor + 0.75 * out[x, y]
        else:
            pass
    
    # out = out * 0.7 + mask * 0.3
    plt.imshow(out)
    plt.show()
    plt.imsave(r"D:\code\ActiveLearning\modAL-master\paper\picture/" + str(name) + ".png", out)
    print(out.shape)
    return out

# raw = io.imread(r'D:\code\ActiveLearning\modAL-master\paper\picture\tree_t7_pos.png')
# cut = raw[1650:2100, 900:1350, 0:3]
# print(cut.shape)
# plt.imsave(r'D:\code\ActiveLearning\modAL-master\paper\picture\tree_t7_pos_cut.png', cut)
# exit(-2)
# region_size = 25
img = io.imread(r'E:\donwload\FloodNet\test-org-img\7577.jpg')
label = io.imread(r"E:\donwload\FloodNet\test-label-img\7577_lab.png")
segments = np.load('../slic2_cache/floodnet/7577/zeroslic_35_scaler.npy')
print(segments.shape)
similarity_path = r'D:\code\ActiveLearning\modAL-master\slic2_cache\floodnet\7577\similarity_35.txt'
# simfeature_path = r'D:\code\ActiveLearning\modAL-master\slic2_cache\floodnet\7577\simfeature_35.npy'
similarity_weight = read_similarity(similarity_path)
# print(segments[640][2975])
p = segments[2480][3375]
sp_id = p - 1
print(sp_id)


neighbor_path = r'D:\code\ActiveLearning\modAL-master\slic2_cache\floodnet\7577\neighbor_35.txt'
neighbor = read_neighbor(neighbor_path)
# sp_id = 2792
n_ids = neighbor[sp_id]
selected_id = similar_sp_around_query(neighbor, sp_id, similarity_weight, nstage=6, similarity_t=10)
selected_id.remove(sp_id)

# simfeature = np.load(simfeature_path)
# selected_id = similar_superpixels_global(img, segments, sp_id, simfeature, similarity_t=2)



img = draw_mask(img, segments, sp_id, selected_id, '7577-1')
