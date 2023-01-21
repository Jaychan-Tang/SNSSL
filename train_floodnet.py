import numpy as np
import time

# Set our RNG seed for reproducibility.
RANDOM_STATE_SEED = 123
np.random.seed(RANDOM_STATE_SEED)

from modAL.expected_error import *
from modAL.density import similarize_distance, information_density
from scipy.spatial.distance import euclidean
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score, average_precision_score, precision_score
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage import io
from slic import *
from modAL.custom_query_strategy import *
import warnings
from start_init import *
from unfolding.similarity import similar_sp_around_query
from unfolding.create_txt import *
from excel import write2excel
from evaluation import *
Xcache_path = "slic_cache/"


def trans_label(label):
    # label = io.imread("data/label/1_0.tif")
    # print(label.shape)
    # label = np.array(label)
    trans = np.zeros((label.shape[0], label.shape[1]))
    # label2 = np.squeeze(label)
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i][j] != 0:
                trans[i][j] = 1
    return trans


def superpixel_class(label, segments):

    regions = regionprops(segments, label)
    s_class = []
    class_count = [0] * label.max()
    mix_cnt = 0
    for i, props in tenumerate(regions):
        # 超像素的外接矩形，不属于目标超像素的像素点为0
        label_array = props.intensity_image
        label_array = label_array.reshape(1, -1)[0]
        label_count = np.bincount(label_array, minlength=label.max())
        
        # 不属于该超像素范围内的像素点为0，所以不计入统计
        label_count[0] = 0
        # print(label_count)
        # print(np.argmax(label_count))
        
        # 占比最大的是目标类
        object_class = np.argmax(label_count)
        
        
        class_count[object_class - 1] += 1
        s_class.append(object_class)
        rate = label_count[object_class] / label_count.sum()
        if rate < 0.7:
            mix_cnt += 1
    
    print('超像素数量', class_count)
    print('混合超像素比例', mix_cnt / sum(class_count))
    return np.array(s_class), np.array(class_count)


def Superpixel_Classification2Semantic_Segmentation(segments, y_raw, predictions):
    regions = regionprops(segments)
    correct_index = predictions == y_raw
    predict_mask = np.zeros((segments.shape[0], segments.shape[1]))
    for i, props in enumerate(tqdm(regions)):
        coordinates = props['Coordinates']
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        # print(coordinates)
        # print(correct_index[i])
        # for c in coordinates:
        predict_mask[x, y] = predictions[i]
    
    return predict_mask


# 多分类的面积比,输入超像素中的所有像素的下标[[x0,x1,...],[y0,y1,...]]和标签，返回超像素真实类别和真实类别占比
def superposition_rate(pixels_index, label):
    # superpixel中的某像素值为1，则对应位置的label数量+1；否则跳过
    # 可以先提取该超像素中的所有label值，再用Counter
    # print(label.tolist())
    l_count = np.array([0] * int(label.max()))
    # print(l_count)
    
    for pi in pixels_index:
        pixel_label = label[pi[0]][pi[1]]
        l_count[pixel_label - 1] += 1
    # x, y = np.where(superpixel == 1)
    # # print(x, y)
    #
    #
    # for i in range(len(x)):
    #    xt, yt = x[i], y[i]
    #    if superpixel[xt, yt] == 1:
    #        l_count[label[xt, yt]-1] += 1
    label_class = np.argmax(l_count)
    # array_and = np.logical_and(superpixel, label)
    
    rate = l_count.max() / pixels_index.shape[0]
    return rate, label_class


def init_sample_visualization(img, segments, center, sample_id, name, save=False):
    mask = np.zeros((segments.shape[0], segments.shape[1]))
    mask = np.repeat(mask[..., np.newaxis], 3, 2)
    
    out = mark_boundaries(img, segments)
    # index_ = np.ones(len(y_raw), dtype=np.int)
    
    # print(correct_index.shape)
    # print(correct_index)
    regions = regionprops(segments)
    segments_copy = segments.copy()
    color_correct = np.array([0, 1, 1])
    color_wrong = np.array([1, 0, 0])
    for i, props in enumerate(tqdm(regions)):
        coordinates = props['Coordinates']
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        # print(coordinates)
        # print(correct_index[i])
        if i in sample_id:
            # for c in coordinates:
            mask[x, y] = color_wrong
        elif i == center:
            mask[x, y] = color_correct
        else:
            mask[x, y] = out[x, y]
    
    out = out * 0.8 + mask * 0.2
    plt.imshow(out)
    if save == True:
        plt.imsave("slic2_cache/floodnet/6651/test/" + str(name) + ".png", out)
    plt.show()
    
    return mask


def draw_mask(img, segments, y_raw, predictions, score, prob):
    mask = np.zeros((segments.shape[0], segments.shape[1]))
    mask = np.repeat(mask[..., np.newaxis], 3, 2)
    # print(mask.shape)
    out = mark_boundaries(img, segments)
    # index_ = np.ones(len(y_raw), dtype=np.int)
    correct_index = predictions == y_raw
    # print(correct_index.shape)
    # print(correct_index)
    regions = regionprops(segments)
    color_correct = np.array([0, 0.5, 1])
    color_wrong = np.array([1, 0, 0])
    for i, props in enumerate(tqdm(regions)):
        coordinates = props['Coordinates']
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        # print(coordinates)
        # print(correct_index[i])
        if correct_index[i] == True:
            # for c in coordinates:
            mask[x, y] = mask[x, y]
            # print(mask.shape)
        # elif prob[i].max() >= 0.8:
        #     mask[x, y] = color_wrong
        else:
            mask[x, y] = color_wrong
    
    out = out * 0.8 + mask * 0.2
    plt.imshow(out)
    plt.imsave("slic2_cache/floodnet/6651/" + str(score) +".png", out)
    plt.show()
    return mask


def Feature_Extract(img, n_segments, mask=None):
    Superpixels_std, Superpixels_brightness, Superpixels_texture, Superpixels_avg, Adjoin_Superpixels, segments, Superpixels_lab = slic_process(
        img, n_segments=n_segments, compactness=30, show_img=False, mask=mask)
    Superpixels_context = []
    
    for ad in tqdm(Adjoin_Superpixels):
        ad1 = Superpixels_lab[ad - 1]
        # ad2 = Superpixels_brightness[ad -1]
        # ad_s = np.concatenate((ad1, ad2))
        # Superpixels_context.append(ad1.reshape(1, -1)[0])
        Superpixels_context.append(ad1.reshape(1, -1)[0])
    
    Superpixels_context = np.array(Superpixels_context)
    print("All Feature Extract Done")
    Superpixels_texture = Superpixels_texture.reshape(Superpixels_texture.shape[0], -1)
    
    Superpixels_brightness = Superpixels_brightness.reshape(Superpixels_brightness.shape[0], -1)
    X_raw = np.concatenate(
        (Superpixels_brightness, Superpixels_std, Superpixels_texture, Superpixels_context, Superpixels_lab), axis=1)
    # print(Superpixels_brightness.shape)
    # print(Superpixels_std.shape)
    # print(Superpixels_texture.shape)
    # print(Superpixels_context.shape)
    # print(Superpixels_lab)
    return X_raw, segments


def color2label(label, use_cache=False):
    if use_cache == True:
        translabel = np.load('slic_cache/Vaihingen/1/translabel.npy')
        # mask = np.load('slic_cache/DroneDeploy/1/mask.npy')
        return translabel
    from tqdm import trange
    
    # Impervious surfaces(RGB: 255, 255, 255)
    # Building(RGB: 0, 0, 255)
    # Low vegetation(RGB: 0, 255, 255)
    # Tree(RGB: 0, 255, 0)
    # Car(RGB: 255, 255, 0)
    # Clutter / background(RGB: 255, 0, 0)
    
    surfaces = np.array([255, 255, 255])  # 白色 背景
    Building = np.array([0, 0, 255])  # 蓝色 建筑
    vegetation = np.array([0, 255, 255])  # 青色 草地
    Tree = np.array([0, 255, 0])  # 绿色 树木
    Car = np.array([255, 255, 0])  # 黄色 车辆
    # CAR = np.array([200, 130, 0])
    Clutter = np.array([255, 0, 0])  # 水体 红色
    
    label_copy = np.zeros((label.shape[0], label.shape[1]), dtype=np.int)
    # mask = np.full((label.shape[0], label.shape[1]), True, dtype=np.bool)
    for i in trange(label.shape[0]):
        for j in range(label.shape[1]):
            tmp = label[i][j]
            # tmp = np.array([tmp[2], tmp[1], tmp[0]])
            
            # if (tmp != IGNORE).all():
            #     print(label[i][j])
            if (tmp == surfaces).all():
                label_copy[i][j] = 1
            elif (tmp == Building).all():
                label_copy[i][j] = 2
            elif (tmp == vegetation).all():
                label_copy[i][j] = 3
            elif (tmp == Tree).all():
                label_copy[i][j] = 4
            elif (tmp == Car).all():
                label_copy[i][j] = 5
            elif (tmp == Clutter).all():
                label_copy[i][j] = 6
            # elif (tmp == IGNORE).all():
            #     mask[i][j] = False
            else:
                print(i, j)
                print(label[i][j])
                # print(IGNORE)
                raise Exception
    
    np.save('slic_cache/Vaihingen/1/translabel.npy', arr=label_copy)
    # np.save('slic_cache/Vaihingen/1/mask.npy', arr=mask)
    return label_copy


def draw_prediction(img, segments, predictions, name):
    mask = np.zeros((segments.shape[0], segments.shape[1]), dtype=np.uint8)
    mask = np.repeat(mask[..., np.newaxis], 3, 2)
    regions = regionprops(segments)
    color_list = [[255, 255, 255],
                  [0, 0, 255],
                  [0, 255, 255],
                  [0, 255, 0],
                  [255, 255, 0],
                  [255, 0, 0]
                  ]
    color_list = np.array(color_list, dtype=np.uint8)
    for i, props in enumerate(tqdm(regions)):
        coordinates = props['Coordinates']
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        label = predictions[i]
        # color = color_list[label-1]
        # color = np.array([color[2], color[1], color[0]])
        mask[x, y] = color_list[label - 1]
    plt.imshow(mask)
    plt.imsave("slic_cache/Vaihingen/1/AL_MS" + str(name) + ".png", mask)


def confidence_count(pred_prob, y_pred, y_test):
    high_confidence_idx = []
    low_confidence_idx = []
    high_correct = []
    high_wrong = []
    low_correct = []
    low_wrong = []
    for i in range(y_pred.shape[0]):
        if pred_prob[i].max() > 0.8:
            high_confidence_idx.append(i)
            if y_pred[i] == y_test[i]:
                high_correct.append(i)
            else:
                high_wrong.append(i)
        else:
            low_confidence_idx.append(i)
            if y_pred[i] == y_test[i]:
                low_correct.append(i)
            else:
                low_wrong.append(i)
    
    print('高置信度：正确/总数', len(high_correct), len(high_confidence_idx), len(high_correct) / len(high_confidence_idx))
    print('低置信度：正确/总数', len(low_correct), len(low_confidence_idx), len(low_correct) / len(low_confidence_idx))
    return low_confidence_idx


def find_merge(segments, neighbor, idx, proba):
    # idx = list(idx)
    queue = []
    merge_idx = []
    seen = set()
    queue.append(idx)
    seen.add(idx)
    
    while (len(queue)>0):
        ver = queue.pop(0)
        notes = neighbor[ver]
        # notes = neighbor[ver]-1
        v_proba = proba[ver]
        for n in notes:
            if n not in seen:
                seen.add(n)
                n_proba = proba[n]
                dis = np.sqrt(np.sum((v_proba - n_proba) ** 2))
                if dis < 0.12:
                    queue.append(n)
                    merge_idx.append(n)
    
    return merge_idx



def find_edge(idx, neighbor, segments):
    img_edge = io.imread('slic2_cache/floodnet/6651/edges.jpg')
    # 保存超像素id和邻域超像素id
    near_sp = list(neighbor[idx])
    near_sp.append(idx)
    regions = regionprops(segments)
    total = 0
    for i, props in enumerate(regions):
        if i in near_sp:
            coordinates = props['Coordinates']
            x = coordinates[:, 0]
            y = coordinates[:, 1]
            pixels = img_edge[x, y]
            sp_total = np.sum(pixels)
            total += sp_total
            # print(sp_total)
    if total > 0:
        return 1
    else:
        return 0
    # init_sample_visualization(img_edge, segments, near_sp, 'edge', False)


def test_model(learner, X_raw, y_raw, oracle_pred_dict, segments, label):
    classes = list(set(y_raw))
    # print(classes)
    predictions = learner.predict(X_raw)
    base_array = superpixel_2darray(predictions, segments)
    # base_miou = mean_iou(base_array, label, classes)
    base_oa = compute_acc(gt=label, pred=base_array)
    base_aa = compute_avg_acc(gt=label, pred=base_array, classes=classes)
    base_kappa = compute_kappa(prediction=base_array, target=label)

    
    for id, olabel in oracle_pred_dict.items():
        predictions[id] = olabel
    
    test_array = superpixel_2darray(predictions, segments)
    # test_miou = mean_iou(test_array, label, classes)
    test_oa = compute_acc(gt=label, pred=test_array)
    test_aa = compute_avg_acc(gt=label, pred=test_array, classes=classes)
    test_kappa = compute_kappa(prediction=test_array, target=label)
    
    # print('miou base/test:', base_miou, test_miou)
    print('overall_acc base/test:', base_oa, test_oa)
    print('avg_acc base/test:', base_aa, test_aa)
    print('kappa base/test:', base_kappa, test_kappa)
    # return [base_kappa], [test_kappa]
    return [base_oa, base_aa, base_kappa], [test_oa, test_aa, test_kappa]


def main(nstage, similarity_t, seed):
    img = io.imread(r'data/img/7577.jpg')
    label = io.imread(r"data/label/7577_lab.png")
    use_Xcache = False

    if use_Xcache:
        X_raw = np.load('cache/floodnet/7577/zerofeature_35_scaler.npy')
        segments = np.load('cache/floodnet/7577/zeroslic_35_scaler.npy')
        y_raw = np.load('cache/floodnet/7577/zeroy_35_scaler.npy')

        neighbor_path = r'cache\floodnet\7577\neighbor_35.txt'
        similarity_path = r'cache\floodnet\7577\similarity_35.txt'

        neighbor = read_neighbor(neighbor_path)
        # similarity_txt(segments, img, neighbor, similarity_path)
        similarity_weight = read_similarity(similarity_path)
        # X_test = np.load('slic_cache/Vaihingen/13/feature25000_30_rgblab.npy')
        # segments_test = np.load('slic_cache/Vaihingen/13/slic25000_30_rgblab.npy')
        print("use cache")
        # print("use X cache")
    else:
        print("slic and feature calculating...")
        X_raw, segments, neighbor = slic_process2(img, region_size=35, show_img=True)
        np.save('cache/floodnet/7577/zerofeature_35_scaler.npy', arr=X_raw)
        np.save('cache/floodnet/7577/zeroslic_35_scaler.npy', arr=segments)
        y_raw, _ = superpixel_class(label, segments)
        np.save('cache/floodnet/7577/zeroy_35_scaler.npy', arr=y_raw)
        neighbor_path = r'cache\floodnet\7577\neighbor_35.txt'
        similarity_path = r'floodnet\7577\similarity_35.txt'
        neighbor_txt(neighbor_path, segments)
        neighbor = read_neighbor(neighbor_path)
        similarity_txt(segments, img, neighbor, similarity_path)
        similarity_weight = read_similarity(similarity_path)

    print("y_raw", y_raw.shape)
    n_labeled_examples = segments.max()
    print('total samples:', n_labeled_examples)

    copy_ids = np.arange(0, X_raw.shape[0])
    chose_ids = []
    oracle_pred_dict = dict()
    
    # 初始化分类器
    knn = KNeighborsClassifier(n_neighbors=7)
    svc = SVC()
    xgboost_classifier = xgb.XGBClassifier()
    GNB = GaussianNB()
    xgb.set_config(verbosity=0)
    
    train_idx = all_random_init(n_initial=30, X_raw=X_raw, seed=seed)
    # train_idx = class_random_init(y_raw, 5, seed=seed, class_num=10)
    chose_ids.extend(train_idx.tolist())
    copy_ids = np.delete(copy_ids, train_idx)
    copy_ids = copy_ids.tolist()
    X_pool = np.delete(X_raw, train_idx, axis=0)
    y_pool = np.delete(y_raw, train_idx, axis=0)
    # euclidean_density = information_density(X_raw, similarize_distance(euclidean))
    # euclidean_density = np.delete(euclidean_density, train_idx, axis=0)
    # nstage = 6
    # similarity_t = 10
    query_strategy = margin_sampling
    # svc.decision_function()
    qname = ['BT', 'BT+ours']
    sheet_name = 'XGB' + '-seed' + str(seed)
    for tid in train_idx:
        query_label = y_raw[tid]
        similar_sp = similar_sp_around_query(neighbor, tid, similarity_weight, nstage=nstage, similarity_t=similarity_t)
        similar_sp.remove(tid)
    
        if len(similar_sp) > 0:
            correct_cnt = 0
            for id in similar_sp:
                y = y_raw[id]
                if y == query_label:
                    correct_cnt += 1
                if id not in oracle_pred_dict:
                    oracle_pred_dict[id] = query_label

    np.save('cache/floodnet/7577/model_save/xgb_random/' + 'seed' + str(seed) + '-train_idx' + '.npy', arr=train_idx)

    cnt = Counter(y_raw[train_idx])
    print(cnt)
    
    print(train_idx.shape, train_idx)
    
    X_train = X_raw[train_idx]
    y_train = y_raw[train_idx]
    print(Counter(y_train))
    y_train = np.array(y_train)
    X_train = np.array(X_train)

    print(y_pool.shape)
    

    learner = ActiveLearner(
        estimator=xgb.XGBClassifier(),
        X_training=X_train, y_training=y_train, query_strategy=query_strategy)
    # learner = ActiveLearner(
    #     estimator=svc,
    #     X_training=X_train, y_training=y_train, query_strategy=query_strategy)
    learner.save('cache/floodnet/7577/model_save/xgb_random/', 'seed' + str(seed) + '-model-' + str(0))

    # 开始训练
    n_queries = 500
    max_acc = 0
    query_num = 0
    model_saved = learner
    acc_list = []
    acc_excel = []
    models = []

    # base, ours = test_model(learner, X_raw, y_raw, oracle_pred_dict, segments, label)
    # acc_excel.append([base, ours])
    
    onehot = False
    if onehot == True:
        onehot_encoder = OneHotEncoder(sparse=False)
        y_pool = y_pool.reshape(-1, 1)
        y_pool = onehot_encoder.fit_transform(y_pool)
        print(y_pool)

    
    for idx in trange(n_queries):
        query_idx, query_instance = learner.query(X_pool=X_pool)
        # query_idx, query_instance = learner.query_svc(X_pool=X_pool)
        
        qid = query_idx[0]
        chose_ids.append(copy_ids.pop(qid))
        rid = chose_ids[-1]
        print(rid)
        # euclidean_density = np.delete(euclidean_density, query_idx, axis=0)

        # if not (X_pool[query_idx[0]] == X_raw[chose_ids[-1]]).all():
        #     print('False')

        query_label = y_raw[rid]
        similar_sp = similar_sp_around_query(neighbor, rid, similarity_weight, nstage=nstage, similarity_t=similarity_t)
        similar_sp.remove(rid)

        if len(similar_sp) > 0:
            correct_cnt = 0
            for id in similar_sp:
                y = y_raw[id]
                if y == query_label:
                    correct_cnt += 1
                if id not in oracle_pred_dict:
                    oracle_pred_dict[id] = query_label
            # init_sample_visualization(img, segments, rid, similar_sp, str(idx), False)
        
        
        selected_x = X_pool[query_idx].reshape(1, -1)
        selected_y = y_pool[query_idx].reshape(1, )
        learner.teach(
            X=selected_x,
            y=selected_y,
        )
        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx)
        
        model_accuracy = learner.score(X_pool, y_pool)
        
        acc_list.append(model_accuracy)
        if model_accuracy > max_acc:
            model_saved = learner
            query_num = idx + 1
            max_acc = model_accuracy
        # if idx % 10 == 0:
        #     print('Accuracy after query {n}: {acc:0.4f}'.format(n=idx + 1, acc=model_accuracy), end='    ')
        #     predictions = learner.predict(X_raw)
        #     bin = np.bincount((predictions == y_raw))
        #     print('correct:', bin[1], 'wrong:', bin[0])
        if (idx+1) % 20 == 0:
            print(idx+1)
            learner.save('cache/floodnet/7577/model_save/xgb_random/', 'seed' + str(seed) + '-model-' + str(idx + 1))

            # learner.save('slic2_cache/floodnet/7577/model/xgb/', str(idx+1))
            base, ours = test_model(learner, X_raw, y_raw, oracle_pred_dict, segments, label)
            acc_excel.append([base, ours])

    # init_sample_visualization(img, segments, train_idx)
    np.save('cache/floodnet/7577/model_save/xgb_random/' + 'seed' + str(seed) + '-chosen_ids-' + str(500) + '.npy',
            arr=np.array(chose_ids))
    excel_path = 'paper/data/floodnet/7577' + str(seed) + '.xlsx'
    write2excel(excel_path, acc_excel, qname, sheet_name=sheet_name, per_num=10)
    



if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=Warning)
    nstage = [1, 2, 3, 4, 5, 6, 7, 8]
    similarity_t = [2, 4, 6, 8, 10, 12, 14, 16]
    seeds = [0,1,2,3,4]
    # for n in nstage:
    #     for t in similarity_t:
    #         if n == 6:
    #             print(n, t)
    #             main(n, t, seed=1)
    # for n in nstage:
    #     for t in similarity_t:
    #         if t == 10 and n != 6:
    #             print(n, t)
    #             main(n, t, seed=1)
    #
    # for seed in seeds:
    #     main(4, 8, seed)
    main(4, 8, seed=0)