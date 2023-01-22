import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
import time
import copy
# Set our RNG seed for reproducibility.
RANDOM_STATE_SEED = 123
np.random.seed(RANDOM_STATE_SEED)
from modAL.expected_error import *
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score, average_precision_score, precision_score
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC, SVC
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from skimage import io
from slic import *
from modAL.custom_query_strategy import *
# from deepforest import CascadeForestClassifier
# from deepforest import ExtraTreesClassifier
# from gcForest_pylablanche.GCForest import gcForest
import warnings
from start_init import *
from unfolding.similarity import similar_sp_around_query
from unfolding.create_txt import *
from excel import write2excel
from evaluation import *
import joblib


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
    print('superpixel_class')
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
            out[x, y] = out[x, y] * 0.8 + color_wrong * 0.2
            # mask[x, y] = color_wrong
        elif i == center:
            # mask[x, y] = color_correct
            out[x, y] = out[x, y] * 0.8 + color_correct * 0.2
        else:
            # mask[x, y] = out[x, y]
            pass
    
    # out = out * 0.8 + mask * 0.2
    plt.imshow(out)
    if save == True:
        plt.imsave("slic2_cache/floodnet/6651/test/" + str(name) + ".png", out)
    plt.show()
    
    return out



def draw_mask(img, segments, y_raw, predictions, score):
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
    plt.imsave("slic2_cache/potsdam/2_10/" + str(score) + ".png", out)
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


def color2label(label):
    
    from tqdm import trange

    class1 = np.array([255, 0, 0])  # 红色 其他（含沙地、不分类区域等）
    class2 = np.array([0, 255, 255]) #青色 草地
    class3 = np.array([0, 255, 0])   #绿色 树木
    class4 = np.array([0, 0, 255])  #蓝色 建筑
    class5 = np.array([255, 255, 0]) #黄色 车辆
    class6 = np.array([255, 255, 255])  #白色 道路
    
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
    io.imsave('data/label/2_10_trans.png', label_copy)
    # np.save('slic_cache/Vaihingen/1/translabel.npy', arr=label_copy)
    # np.save('slic_cache/Vaihingen/1/mask.npy', arr=mask)
    return label_copy


def draw_prediction(segments, predictions, oracle_pred_dict, name):
    mask = np.zeros((segments.shape[0], segments.shape[1]), dtype=np.uint8)
    mask = np.repeat(mask[..., np.newaxis], 3, 2)
    regions = regionprops(segments)
    color_list = [[255, 0, 0],
                  [0, 255, 255],
                  [0, 255, 0],
                  [0, 0, 255],
                  [255, 255, 0],
                  [255, 255, 255]
                  ]
    color_list = np.array(color_list, dtype=np.uint8)
    for i, props in enumerate(regions):
        coordinates = props['Coordinates']
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        label = predictions[i]
        # color = color_list[label-1]
        # color = np.array([color[2], color[1], color[0]])
        mask[x, y] = color_list[label-1]
    plt.imshow(mask)
    plt.show()
    # plt.imsave("paper/data/potsdam/" + str(name) + ".png", mask)


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
    
    while (len(queue) > 0):
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


def test_model(learner, X_raw, y_raw, oracle_pred_dict, segments, label, name):
    classes = list(set(y_raw))
    # print(classes)
    predictions = learner.predict(X_raw)
    base_array = superpixel_2darray(predictions, segments)
    # base_miou = mean_iou(base_array, label, classes)
    base_oa = compute_acc(gt=label, pred=base_array)
    base_aa = compute_avg_acc(gt=label, pred=base_array, classes=classes)
    base_kappa = compute_kappa(prediction=base_array, target=label)
    # base_class_acc = compute_class_acc(gt=label, pred=base_array, classes=classes)
    
    for id, olabel in oracle_pred_dict.items():
        predictions[id] = olabel
    
    test_array = superpixel_2darray(predictions, segments)
    # test_miou = mean_iou(test_array, label, classes)
    test_oa = compute_acc(gt=label, pred=test_array)
    test_aa = compute_avg_acc(gt=label, pred=test_array, classes=classes)
    test_kappa = compute_kappa(prediction=test_array, target=label)
    # test_class_acc = compute_class_acc(gt=label, pred=test_array, classes=classes)
    
    # np.save('slic2_cache/potsdam/2_10/base_array_'+str(name)+'.npy', arr=base_array)
    # np.save('slic2_cache/potsdam/2_10/test_array_'+str(name)+'.npy', arr=test_array)
    
    # print('miou base/test:', base_miou, test_miou)
    report = classification_report(y_true=label.flatten(), y_pred=base_array.flatten(), digits=4)
    print(report)
    report = classification_report(y_true=label.flatten(), y_pred=test_array.flatten(), digits=4)
    print(report)
    print('overall_acc base/test:', base_oa, test_oa)
    print('avg_acc base/test:', base_aa, test_aa)
    print('kappa base/test:', base_kappa, test_kappa)
    # print('acc_class base/test', base_class_acc, test_class_acc)
    
    # return [base_kappa], [test_kappa]
    return [base_oa, base_aa, base_kappa], [test_oa, test_aa, test_kappa]


def main(nstage, similarity_t, seed=1):
    img = io.imread(r'data/img/top_potsdam_2_10_RGB.png')
    label = io.imread(r"data/label/potsdam_2_10_trans.png")
    
    # print(Counter(label.tolist()))
    print("read done")
    
    warnings.filterwarnings(action='ignore')
    use_Xcache = True
    region_size = 20
    if use_Xcache:
        X_raw = np.load('slic2_cache/potsdam/2_10/feature_' + str(region_size) + '.npy')
        segments = np.load('slic2_cache/potsdam/2_10/slic_' + str(region_size) + '.npy')
        y_raw = np.load('slic2_cache/potsdam/2_10/y_' + str(region_size) + '.npy')
        
        neighbor_path = r'D:\code\ActiveLearning\modAL-master\slic2_cache\potsdam\2_10\neighbor_' + str(region_size) + '.txt'
        similarity_path = r'D:\code\ActiveLearning\modAL-master\slic2_cache\potsdam\2_10\similarity_' + str(region_size) + '.txt'
        
        neighbor = read_neighbor(neighbor_path)
        # similarity_txt(segments, img, neighbor, similarity_path)
        similarity_weight = read_similarity(similarity_path)
        print("use cache")
    else:
        X_raw, segments, neighbor = slic_process2(img, region_size=region_size)
        np.save('slic2_cache/potsdam/2_10/feature_' + str(region_size) + '.npy', arr=X_raw)
        np.save('slic2_cache/potsdam/2_10/slic_' + str(region_size) + '.npy', arr=segments)
        # X_raw = np.load('slic2_cache/potsdam/2_10/feature_' + str(region_size) + '.npy')
        # segments = np.load('slic2_cache/potsdam/2_10/slic_' + str(region_size) + '.npy')
        y_raw, _ = superpixel_class(label, segments)
        np.save('slic2_cache/potsdam/2_10/y_' + str(region_size) + '.npy', arr=y_raw)
        neighbor_path = r'D:\code\ActiveLearning\modAL-master\slic2_cache\potsdam\2_10\neighbor_' + str(region_size) + '.txt'
        similarity_path = r'D:\code\ActiveLearning\modAL-master\slic2_cache\potsdam\2_10\similarity_' + str(region_size) + '.txt'
        neighbor_txt(neighbor_path, segments)
        save_neighbor(neighbor_path, neighbor)
        similarity_txt(segments, img, neighbor, similarity_path)
        similarity_weight = read_similarity(similarity_path)
    
    print("y_raw", y_raw.shape)
    # print("count", class_count / y_raw.shape[0])
    
    n_labeled_examples = segments.max()
    print('total samples:', n_labeled_examples)

    copy_ids = np.arange(0, X_raw.shape[0])
    chose_ids = []
    oracle_pred_dict = dict()
    
    # Isolate the non-training examples we'll be querying.
    # seed=1
    # train_idx = all_random_init(n_initial=30, X_raw=X_raw, seed=seed)
    train_idx = class_random_init(y_raw, 5, seed=seed, class_num=10)
    chose_ids.extend(train_idx.tolist())
    copy_ids = np.delete(copy_ids, train_idx)
    copy_ids = copy_ids.tolist()
    X_pool = np.delete(X_raw, train_idx, axis=0)
    y_pool = np.delete(y_raw, train_idx, axis=0)
   
    
    # nstage = 6
    # similarity_t = 10
    query_strategy = MCLU
    qname = ['MCLU', 'MCLU+ours']
    sheet_name = 'SVC' + 'S' + str(nstage)+'-T' + str(similarity_t)

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
    
    np.save('slic2_cache/potsdam/2_10/model_save/svm/' + 'seed' + str(seed) + '-train_idx' + '.npy', arr=train_idx)
    
    # 初始化分类器
    # knn = KNeighborsClassifier(n_neighbors=5)
    svc = SVC()
    # svc.decision_function()
    xgboost_classifier = xgb.XGBClassifier()
    xgb.set_config(verbosity=0)
    
    # adjoin_sp = adjoin_supperpixels_id(segments)

    cnt = Counter(y_raw[train_idx])
    print(cnt)
    X_train = X_raw[train_idx]
    y_train = y_raw[train_idx]

    y_train = np.array(y_train)
    X_train = np.array(X_train)
    
    print(y_pool.shape)

    learner = ActiveLearner(
        estimator=svc,
        X_training=X_train, y_training=y_train, query_strategy=query_strategy)
    learner.save('slic2_cache/potsdam/2_10/model_save/svm/', 'seed' + str(seed) + '-model-' + str(0))
    # 开始训练
    n_queries = 1000
    onehot = False
    acc_excel = []
    base, ours = test_model(learner, X_raw, y_raw, oracle_pred_dict, segments, label)
    acc_excel.append([base, ours])
    
    if onehot == True:
        onehot_encoder = OneHotEncoder(sparse=False)
        y_pool = y_pool.reshape(-1, 1)
        y_pool = onehot_encoder.fit_transform(y_pool)
        print(y_pool)


    for idx in trange(n_queries):
        # query_idx, query_instance = learner.query(X_pool=X_pool)
        query_idx, query_instance = learner.query_svc(X_pool=X_pool)
        
        qid = query_idx[0]
        chose_ids.append(copy_ids.pop(qid))
        rid = chose_ids[-1]
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
            # init_sample_visualization(img, segments, rid, similar_sp, 'tst', False)

        
        selected_x = X_pool[query_idx].reshape(1, -1)
        selected_y = y_pool[query_idx].reshape(1, )
        learner.teach(
            X=selected_x,
            y=selected_y,
        )
        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx)
        
        # if (idx+1) % 10 == 0:
        #     print('Accuracy after query {n}: {acc:0.4f}'.format(n=idx + 1, acc=model_accuracy), end='    ')
        #     predictions = learner.predict(X_pool)980127
        
        #     bin = np.bincount((predictions == y_pool))
        #     print('correct:', bin[1], 'wrong:', bin[0])
        if (idx + 1) % 20 == 0:
            # print(idx + 1)
            learner.save('slic2_cache/potsdam/2_10/model_save/svm/', 'seed' + str(seed) + '-model-' + str(idx+1))
            
            # pred = learner.predict(X_raw)
            # draw_prediction(segments, pred, oracle_pred_dict, '2_10MCLU7-12')
            # for id, olabel in oracle_pred_dict.items():
            #     pred[id] = olabel
            # print(classification_report(y_raw, pred))
            # draw_prediction(segments, pred, oracle_pred_dict, '2_10MCLUOURS7-12')
            base, ours = test_model(learner, X_raw, y_raw, oracle_pred_dict, segments, label, 'MCLU7-12-'+str(seed))
            acc_excel.append([base, ours])
    np.save('slic2_cache/potsdam/2_10/model_save/svm/' + 'seed' + str(seed) + '-chosen_ids-' + str(0) + '.npy',
            arr=np.array(chose_ids))
    # init_sample_visualization(img, segments, train_idx)
    # excel_path = 'paper/data/potsdam/2_10-slic'+str(region_size)+'-n' + str(nstage) + '-sim' + str(similarity_t) + '--.xlsx'
    # excel_path = 'paper/data/potsdam/2_10set' + '--.xlsx'
    # write2excel(excel_path, acc_excel, qname, sheet_name=sheet_name, per_num=20)
    



if __name__ == "__main__":
    nstage = [1, 2, 3, 4, 5, 6, 7, 8]
    similarity_t = [2, 4, 6, 8, 10, 12, 14, 16]
    seeds = [1,0,2,3,4]
    # for n in nstage:
    #     for t in similarity_t:
    #         if n == 7 and t==12:
    #             print(n, t)
    #             main(n, t, seed=1)
    #             exit(-2)
    # for n in nstage:
    #     for t in similarity_t:
    #         if t == 12 and n != 7:
    #             print(n, t)
    #             main(n, t, seed=1)
    # for seed in seeds:
    #     main(7, 12, seed)
    main(7, 12, seed=0)

