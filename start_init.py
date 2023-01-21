def all_random_init(n_initial, X_raw, seed):
    import numpy as np
    np.random.seed(seed)
    train_idx = np.random.choice(range(X_raw.shape[0]), size=n_initial, replace=False)
    return np.array(train_idx)

def class_random_init(y, sample_per_class, seed, class_num=7):
    import numpy as np
    np.random.seed(seed)
    # init_per_class = 5
    # n_initial = init_per_class * 7
    initial_sample_id = []
    for i in range(class_num):
        select_idx = np.argwhere(y == i + 1)
        if select_idx.shape[0] >= sample_per_class:
            train_idx = np.random.choice(select_idx.reshape(-1, ), size=sample_per_class, replace=False)
            initial_sample_id.extend(train_idx)
    return np.array(initial_sample_id)

def class_select_init(x, y, sample_per_class=[8,2,4,8,2,3,3]):
    # sample_per_class=[9,1,5,8,2,1,4]
    import numpy as np
    
    # from tqdm import tqdm
    # from sklearn.metrics.pairwise import euclidean_distances
    initial_sample_id = []
    for i in range(7):
        select_idx = np.argwhere(y == i + 1).reshape(-1,)
        # print(np.unique(y[select_idx]))
        select_x = x[select_idx]
        # print(x[select_idx].shape)
        center_x = np.average(select_x, axis=0)
        # print(center_x.shape)
        dis = np.array([np.linalg.norm(sx - center_x) for sx in select_x])
        # print("center_x", center_x)
        # 距离最远的n个样本
        dis_min_idx = np.argpartition(dis, sample_per_class[i])[:sample_per_class[i]]
        # dis_max_idx = np.argpartition(dis, -sample_per_class[i])[-sample_per_class[i]:]
        # dis_min_idx = np.argsort(dis)[:sample_per_class[i]]
        # print("dis_min", dis[dis_min_idx])
        # print(y[dis_min_idx])
        initial_sample_id.extend(select_idx[dis_min_idx])
        
    # print(Counter)
    return np.array(initial_sample_id)


# 弃用
def partition_cluster_init(x, y, seed=0):
    from sklearn.cluster import DBSCAN
    from sklearn.cluster import KMeans
    from sklearn.metrics import jaccard_score
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    
    ss = StandardScaler()
    # x = ss.fit_transform(x)
    
    std = x[:, :3]
    std = ss.fit_transform(std)
    
    avg_RGB = x[:, 3:6]
    avg_RGB = ss.fit_transform(avg_RGB)
    
    text = x[:, 6:10]
    adjoin_avg = x[:, 10:]
    
    part1 = np.concatenate((std, avg_RGB), axis=1)
    db = DBSCAN(eps=0.3, min_samples=20).fit(part1)
    clustering_labels = db.labels_
    n_clusters_ = len(set(clustering_labels)) - (1 if -1 in clustering_labels else 0)
    print("DBSCAN: n_clusters_", n_clusters_)
    js = jaccard_score(y_true=y, y_pred=clustering_labels, average="micro")
    print("DBSCAN: jaccard_score", js)
    # print(std.shape)

    k_means = KMeans(n_clusters=7, random_state=seed).fit(part1)
    clustering_label = k_means.labels_
    js = jaccard_score(y_true=y, y_pred=clustering_label, average="micro")
    print("k-means: jaccard_score", js)

def X_StandardScaler(x):
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    ss1 = StandardScaler()
    std = x[:, :3]
    std = ss1.fit_transform(std)
    
    ss2 = StandardScaler()
    avg_RGB = x[:, 3:6]
    avg_RGB = ss2.fit_transform(avg_RGB)
    
    ss3 = StandardScaler()
    text = x[:, 6:10]
    text = ss3.fit_transform(text)
    
    ss4 = StandardScaler()
    adjoin_avg = x[:, 10:]
    adjoin_avg = ss4.fit_transform(adjoin_avg)
    
    x = np.concatenate((std, avg_RGB, text, adjoin_avg), axis=1)
    return x

def clustering_init(x, n_clusters=30, seed=0, sec=False):
    from sklearn.cluster import DBSCAN, AgglomerativeClustering, SpectralClustering, KMeans, Birch
    from sklearn.preprocessing import scale, MinMaxScaler, normalize
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    import numpy as np
    np.random.seed(seed)
    x_norm = normalize(x, axis=0, norm='max')
    # x = X_StandardScaler(x)
    
    
    initial_sample_id = []
    # clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward", compute_distances=True).fit(x)
    # clustering = SpectralClustering(n_clusters=300, n_jobs=-1, eigen_solver='amg').fit(x)
    clustering = Birch(n_clusters=n_clusters).fit(x_norm)
    # clustering = KMeans(n_clusters=n_clusters, max_iter=20000, random_state=1).fit(x)
    clustering_label = clustering.labels_
    # print(set(clustering_label))
    for i in range(n_clusters):
        ind = np.argwhere(clustering_label == i).reshape(-1, )
        # print(ind.shape)
        select_x = x[ind]
        center_x = np.average(select_x, axis=0)
        dis = np.array([np.linalg.norm(sx - center_x) for sx in select_x])
        # print("center_x", center_x)
        # 距离最近的n个样本
        if len(dis) > 1:
            dis_min_idx = np.argpartition(dis, 1)[:1]
            
            if sec == True:
                if sec_clustering(x=x, sample_id=ind[dis_min_idx], clusters_samples_id=ind):
                    initial_sample_id.extend(ind[dis_min_idx])

            else:
                initial_sample_id.extend(ind[dis_min_idx])
            
        elif len(dis) == 1:
            initial_sample_id.extend(ind)
        # train_idx = np.random.choice(range(select_x.shape[0]), size=1, replace=False)
        
        # initial_sample_id.extend(train_idx)
        # center_x = np.average(select_x, axis=0)
    # plt.plot(clustering.distances_)
    # plt.show()
    return np.array(initial_sample_id), clustering_label

def refine(train_idx, x_raw, y_raw):
    from sklearn.preprocessing import scale, MinMaxScaler, normalize
    import numpy as np
    x = x_raw[train_idx]
    x_norm = normalize(x, axis=0, norm='max')
    # for i in range(x.shape[1]-1):
    #     x[:, i:i+1] = MinMaxScaler(x[:, i:i+1])
        
        
    rgb =x_norm[:, :3]
    rgb_std = x_norm[:, 3:6]
    texture = x_norm[:, 6:10]
    context = x_norm[:, 10:34]
    lab = x_norm[:, 34:]

    feature_list = [rgb, rgb_std, texture, context, lab]
    
    
    
    
    


def sec_clustering(x, sample_id, clusters_samples_id, n_clusters=2,  seed=0):
    from sklearn.cluster import DBSCAN, AgglomerativeClustering, SpectralClustering, KMeans, Birch
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    import numpy as np
    np.random.seed(seed)
    clusters_samples = x[clusters_samples_id]
    # print(sample_id)
    select_sample = x[sample_id][0]
    # x = X_StandardScaler(x)
    
    initial_sample_id = []
    # clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward", compute_distances=True).fit(clusters_samples)
    # clustering = DBSCAN(eps=20.0).fit(clusters_samples)
    # clustering = SpectralClustering(n_clusters=300, n_jobs=-1, eigen_solver='amg').fit(x)
    clustering = Birch(n_clusters=2).fit(clusters_samples)
    # clustering = KMeans(n_clusters=n_clusters, max_iter=20000, random_state=1).fit(clusters_samples)
    clustering_label = clustering.labels_
    # print(set(clustering_label))
    cnt = []
    for i in range(n_clusters):
        ind = np.argwhere(clustering_label == i).reshape(-1, )
        cnt.append(ind.shape[0])
    big_cluster_id = cnt.index(max(cnt))
    
    
    ind = np.argwhere(clustering_label == big_cluster_id).reshape(-1, )
    big_cluster_samples = clusters_samples[ind]
    # print(big_cluster_samples.shape)
    # print(select_sample.shape)
    for i in range(big_cluster_samples.shape[0]):
        # print(select_sample)
        if np.array_equal(big_cluster_samples[i].T, select_sample):
            return True

    return False
    



# 类内随机初始化
def inclass_select_init(x, y, sample_per_class=[9,1,5,8,2,1,4], seed=0):
    # [9,1,5,8,2,1,4]
    # [8,2,4,8,2,3,3]
    import numpy as np
    np.random.seed(seed)
    # from tqdm import tqdm
    # from sklearn.metrics.pairwise import euclidean_distances
    initial_sample_id = []
    for i in range(7):
        select_idx = np.argwhere(y == i + 1).reshape(-1, )
        # print(np.unique(y[select_idx]))
        select_x = x[select_idx]
        # print(x[select_idx].shape)
        # center_x = np.average(select_x, axis=0)
        train_idx = np.random.choice(range(select_x.shape[0]), size=sample_per_class[i], replace=False)
        # x_idx = select_x[train_idx]
        # print(i, x_idx)
        initial_sample_id.extend(train_idx)
        print(i, train_idx)
    return np.array(initial_sample_id)


def check_clustering(initial_sample_id, clustering_label, n_clusters, y):
    # assert len(initial_sample_id) == n_clusters, "样本数量与聚类数不等"
    import numpy as np
    sample_labels = y[initial_sample_id]
    out = []
    for i in range(n_clusters):
        ind = np.argwhere(clustering_label == i).reshape(-1, )
        labels = y[ind]
        cnt = 0
        for label in labels:
            if label == sample_labels[i]:
                cnt += 1
        rate = cnt/ind.shape[0]
        out.append([sample_labels[i], rate])
    return out
    
    
def check_mix(segments, label, train_idx):
    from skimage.measure import regionprops
    import numpy as np
    regions = regionprops(segments, label)
    s_class = []
    mix_count = 0
    # class_count = [0] * label.max()
    for i, props in enumerate(regions):
        if i in train_idx:
        
            # 超像素的外接矩形，不属于目标超像素的像素点为0
            label_array = props.intensity_image
            label_array = label_array.reshape(1, -1)[0]
            label_count = np.bincount(label_array, minlength=8)
            
            # 不属于该超像素范围内的像素点为0，所以不计入统计
            # label_count[0] = 0
            label_count = label_count[1:]
            # print(label_count)
            # print(np.argmax(label_count))
            
            # 占比最大的是目标类
            object_class = np.argmax(label_count)
            sort = np.argsort(label_count)
            # print(label_count)
            mix_superpixel = np.argwhere(label_count >= np.sum(label_count)*0.2).reshape(-1,)
            # print(mix_superpixel)
            if len(mix_superpixel) > 1:
                mix_count += 1
            # class_count[object_class - 1] += 1
            s_class.append(object_class)
    return mix_count / len(train_idx)


def Coassociation_matrix(n_samples, clusters_list: list = None):
    import numpy as np
    import itertools
    from tqdm import tqdm
    matrix = np.zeros([n_samples, n_samples], dtype=np.float16)
    
    for c in clusters_list:
        clustering_label = c.labels_
        n_clusters = clustering_label.max() + 1
        for i in tqdm(range(n_clusters)):
            ind = np.argwhere(clustering_label == i).reshape(-1, )
            ind_x, ind_y = np.meshgrid(ind, ind)
            # print(ind_x.shape, ind_y.shape)
            matrix[ind_x, ind_y] += 1
            # for j in tqdm(itertools.product(ind, ind)):
            #     matrix[j[0], j[1]] += 1
            
            # mesh = np.meshgrid(ind)
            # cart = np.array(mesh).T.reshape(-1, len(ind))
            # ind2d = np.meshgrid(ind, ind)

    matrix = matrix/len(clusters_list)
    matrix = np.where(matrix > 0.6, 1.0, 0.0)
    return matrix
    
def Matrix2Cluster(matrix):
    import numpy as np
    from tqdm import trange
    clusters = []
    flag = [0] * matrix.shape[0]
    finallabel = [-1] * matrix.shape[0]
    cur_label = 1
    uf = UnionFind(matrix.shape[0])
    for i in trange(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if j >= i:
                continue
            if matrix[i][j] == 1:
                uf.union(i, j)
            # if matrix[i][j] == 1:
            #     if flag[i] == 0 and flag[j] == 0:
            #         finallabel[i] = cur_label
            #         finallabel[j] = cur_label
            #         cur_label += 1
            #         flag[i] = 1
            #         flag[j] = 1
            #     elif flag[i] == 0 and flag[j] == 1:
            #         finallabel[i] = finallabel[j]
            #         flag[i] = 1
            #     elif flag[i] == 1 and flag[j] == 0:
            #         finallabel[j] = finallabel[i]
            #         flag[j] = 1
            #     else:
            #         if finallabel[i] != finallabel[j]:
            #             relabel = finallabel[j]
            #             for m in range(len(finallabel)):
            #                 if finallabel[m] == relabel:
            #                     finallabel[m] = finallabel[i]
                
            
    clusters_label = np.array(uf.label)
    print(clusters_label.shape)
    # exit(-3)
    # clusters_label = np.array((matrix.shape[0], ), dtype=np.int)
    label_ind = 0
    for cluster in clusters:
        clusters_label[cluster] = label_ind
        label_ind += 1
    print(clusters_label)
    return clusters_label
                
                

            
            
    

def clustering_ensemble(x, n_clusters=30):
    from sklearn.cluster import DBSCAN, AgglomerativeClustering, SpectralClustering, KMeans, Birch
    import numpy as np
    c1 = KMeans(n_clusters=n_clusters, max_iter=20000).fit(x)
    c2 = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward", compute_distances=False).fit(x)
    # c3 = SpectralClustering(n_clusters=30, n_init=10, gamma=1.0, affinity='precomputed', n_neighbors=10, eigen_tol=0.0, assign_labels='kmeans', degree=3, coef0=1).fit(x)
    c4 = Birch(n_clusters=30).fit(x)
    clusters_list = [c1, c2, c4]
    # clusters_list = [c1]
    print("cluster list fit done")
    matrix = Coassociation_matrix(n_samples=x.shape[0], clusters_list=clusters_list)
    # clustering_label = Matrix2Cluster(matrix)
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward", connectivity=matrix, compute_distances=True).fit(x)
    clustering_label = clustering.labels_
    print("clustering_label done")
    # return clustering_label
    initial_sample_id = []
    for i in range(n_clusters):
        ind = np.argwhere(clustering_label == i).reshape(-1, )
        print(ind.shape)
        select_x = x[ind]
        center_x = np.average(select_x, axis=0)
        dis = np.array([np.linalg.norm(sx - center_x) for sx in select_x])

        # 距离最近的n个样本
        if len(dis) > 1:
            dis_min_idx = np.argpartition(dis, 1)[:1]
            initial_sample_id.extend(ind[dis_min_idx])
        elif len(dis) == 1:
            initial_sample_id.extend(ind)

    return np.array(initial_sample_id), clustering_label


class UnionFind(object):
    count = 0
    parent = []
    
    def __init__(self, n):
        self.uf = [-1 for i in range(n + 1)]  # 列表0位置空出
        self.label = [-1] * n
        self.sets_count = n  # 判断并查集里共有几个集合, 初始化默认互相独立

    def find(self, p):
        if self.uf[p] < 0:
            return p
        self.uf[p] = self.find(self.uf[p])
        return self.uf[p]

    def union(self, p, q):
        proot = self.find(p)
        qroot = self.find(q)
        self.label[p] = proot
        self.label[q] = qroot
        if proot == qroot:
            return
        elif self.uf[proot] > self.uf[qroot]:  # 负数比较, 左边规模更小
            self.uf[qroot] += self.uf[proot]
            self.uf[proot] = qroot
        else:
            self.uf[proot] += self.uf[qroot]  # 规模相加
            self.uf[qroot] = proot
        self.sets_count -= 1  # 连通后集合总数减一
    
    def is_connected(self, p, q):
        return self.find(p) == self.find(q)     # 即判断两个结点是否是属于同一个祖先

