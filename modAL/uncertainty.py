"""
Uncertainty measures and uncertainty based sampling strategies for the active learning models.
"""
from typing import Tuple

import numpy as np
from scipy.stats import entropy
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator

from modAL.utils.data import modALinput
from modAL.utils.selection import multi_argmax, shuffled_argmax


def _proba_uncertainty(proba: np.ndarray) -> np.ndarray:
    """
    Calculates the uncertainty of the prediction probabilities.

    Args:
        proba: Prediction probabilities.

    Returns:
        Uncertainty of the prediction probabilities.
    """

    return 1 - np.max(proba, axis=1)


def _proba_margin(proba: np.ndarray) -> np.ndarray:
    """
    Calculates the margin of the prediction probabilities.

    Args:
        proba: Prediction probabilities.

    Returns:
        Margin of the prediction probabilities.
    """

    if proba.shape[1] == 1:
        return np.zeros(shape=len(proba))

    part = np.partition(-proba, 1, axis=1)
    margin = - part[:, 0] + part[:, 1]

    return margin


def _proba_entropy(proba: np.ndarray) -> np.ndarray:
    """
    Calculates the entropy of the prediction probabilities.

    Args:
        proba: Prediction probabilities.

    Returns:
        Uncertainty of the prediction probabilities.
    """

    return np.transpose(entropy(np.transpose(proba)))


def classifier_uncertainty(classifier: BaseEstimator, X: modALinput, **predict_proba_kwargs) -> np.ndarray:
    """
    Classification uncertainty of the classifier for the provided samples.

    Args:
        classifier: The classifier for which the uncertainty is to be measured.
        X: The samples for which the uncertainty of classification is to be measured.
        **predict_proba_kwargs: Keyword arguments to be passed for the :meth:`predict_proba` of the classifier.

    Returns:
        Classifier uncertainty, which is 1 - P(prediction is correct).
    """
    # calculate uncertainty for each point provided
    try:
        classwise_uncertainty = classifier.predict_proba(X, **predict_proba_kwargs)
    except NotFittedError:
        return np.ones(shape=(X.shape[0], ))

    # for each point, select the maximum uncertainty
    uncertainty = 1 - np.max(classwise_uncertainty, axis=1)
    return uncertainty


def classifier_margin(classifier: BaseEstimator, X: modALinput, **predict_proba_kwargs) -> np.ndarray:
    """
    Classification margin uncertainty of the classifier for the provided samples. This uncertainty measure takes the
    first and second most likely predictions and takes the difference of their probabilities, which is the margin.

    Args:
        classifier: The classifier for which the prediction margin is to be measured.
        X: The samples for which the prediction margin of classification is to be measured.
        **predict_proba_kwargs: Keyword arguments to be passed for the :meth:`predict_proba` of the classifier.

    Returns:
        Margin uncertainty, which is the difference of the probabilities of first and second most likely predictions.
    """
    try:
        classwise_uncertainty = classifier.predict_proba(X, **predict_proba_kwargs)
        # classwise_uncertainty = classifier.decision_function(X, **predict_proba_kwargs)
    except NotFittedError:
        return np.zeros(shape=(X.shape[0], ))

    if classwise_uncertainty.shape[1] == 1:
        return np.zeros(shape=(classwise_uncertainty.shape[0],))

    part = np.partition(-classwise_uncertainty, 1, axis=1)
    margin = - part[:, 0] + part[:, 1]

    return margin


def classifier_entropy(classifier: BaseEstimator, X: modALinput, **predict_proba_kwargs) -> np.ndarray:
    """
    Entropy of predictions of the for the provided samples.

    Args:
        classifier: The classifier for which the prediction entropy is to be measured.
        X: The samples for which the prediction entropy is to be measured.
        **predict_proba_kwargs: Keyword arguments to be passed for the :meth:`predict_proba` of the classifier.

    Returns:
        Entropy of the class probabilities.
    """
    try:
        classwise_uncertainty = classifier.predict_proba(X, **predict_proba_kwargs)
    except NotFittedError:
        return np.zeros(shape=(X.shape[0], ))

    return np.transpose(entropy(np.transpose(classwise_uncertainty)))


def uncertainty_sampling(classifier: BaseEstimator, X: modALinput,
                         n_instances: int = 1, random_tie_break: bool = False,
                         **uncertainty_measure_kwargs) -> np.ndarray:
    """
    Uncertainty sampling query strategy. Selects the least sure instances for labelling.

    Args:
        classifier: The classifier for which the labels are to be queried.
        X: The pool of samples to query from.
        n_instances: Number of samples to be queried.
        random_tie_break: If True, shuffles utility scores to randomize the order. This
            can be used to break the tie when the highest utility score is not unique.
        **uncertainty_measure_kwargs: Keyword arguments to be passed for the uncertainty
            measure function.

    Returns:
        The indices of the instances from X chosen to be labelled;
        the instances from X chosen to be labelled.
    """
    uncertainty = classifier_uncertainty(classifier, X, **uncertainty_measure_kwargs)
    
    if not random_tie_break:
        return multi_argmax(uncertainty, n_instances=n_instances)

    return shuffled_argmax(uncertainty, n_instances=n_instances)

def uncertainty_sampling_clustering(classifier: BaseEstimator, X: modALinput,
                         n_instances: int = 1, random_tie_break: bool = False,
                         clustering_weight: list = None, clustering_label: np.array = None,
                         **uncertainty_measure_kwargs) -> np.ndarray:
    uncertainty = classifier_uncertainty(classifier, X, **uncertainty_measure_kwargs)
    
    if not random_tie_break:
        
        # print("uncertainty_sampling_clustering")
        # 提取前n个不确定instance的下标
        result_arg = multi_argmax(uncertainty, n_instances=500)

        # 使用权重更新disagreement
        for r in result_arg:
            # 判断该instance下标对应的簇类别
            instance_label = int(clustering_label[r])
            # print("instance_label-1:", instance_label-1)
            uncertainty[r] = uncertainty[r] * clustering_weight[instance_label - 1]

        # 使用更新后的disagreement，提取加权后的最大分歧
        # final_arg = multi_argmax(uncertainty, n_instances=n_instances)
        return multi_argmax(uncertainty, n_instances=n_instances)
    
    
    return shuffled_argmax(uncertainty, n_instances=n_instances)


def uncertainty_sampling_weight(classifier: BaseEstimator, X: modALinput, predict_label,
                                    n_instances: int = 1, random_tie_break: bool = False,
                                    weight: list = None,
                                    **uncertainty_measure_kwargs) -> np.ndarray:
    uncertainty = classifier_uncertainty(classifier, X, **uncertainty_measure_kwargs)
    
    if not random_tie_break:
        
        # print("uncertainty_sampling_clustering")
        # 提取前n个不确定instance的下标
        result_arg = multi_argmax(uncertainty, n_instances=uncertainty.shape[0])
        
        if predict_label != []:
            # 使用权重更新disagreement
            for i, r in enumerate(result_arg):
                # 判断该instance下标对应的簇类别
                instance_label = predict_label[i] -1
                # print("instance_label-1:", instance_label-1)
                uncertainty[r] = uncertainty[r] * weight[instance_label]
            
        # 使用更新后的disagreement，提取加权后的最大分歧
        # final_arg = multi_argmax(uncertainty, n_instances=n_instances)
        return multi_argmax(uncertainty, n_instances=n_instances)
    
    return shuffled_argmax(uncertainty, n_instances=n_instances)


def edge_sampling(classifier: BaseEstimator, X: modALinput, predict_label, weight: float = 0.25, class_weight: list = [],
                                    n_instances: int = 1, random_tie_break: bool = False,
                                    adjoin_sp: list = None,
                                    **uncertainty_measure_kwargs) -> np.ndarray:
    edge_count = edge_superpixels(predict_label, adjoin_sp, edge_weight=weight)
    margin = classifier_margin(classifier, X, **uncertainty_measure_kwargs)
    edge_uncertainty = edge_count*margin
    # print(edge_uncertainty.shape)
    # if class_weight != []:
    #     result_arg = multi_argmax(uncertainty, n_instances=uncertainty.shape[0])
    #     for i, r in enumerate(result_arg):
    #         # 判断该instance下标对应的簇类别
    #         instance_label = int(predict_label[i]) - 1
    #         # print("instance_label-1:", instance_label-1)
    #         edge_uncertainty[r] = edge_uncertainty[r] * class_weight[instance_label]
    #     return multi_argmax(edge_uncertainty, n_instances=n_instances)
    

    # edge_count(weight=0.6) ** uncertainty
    return multi_argmax(-edge_uncertainty, n_instances=n_instances)


def uncertainty_sampling_with_adjoin(classifier: BaseEstimator, X: modALinput, adjoin_sp,
                         n_instances: int = 1, random_tie_break: bool = False,
                         **uncertainty_measure_kwargs) -> np.ndarray:

    uncertainty = classifier_uncertainty(classifier, X, **uncertainty_measure_kwargs)
    ad_uncertainty = adjoin_uncertainty(adjoin_sp, uncertainty)
    # print(uncertainty)
    # print(ad_uncertainty)
    # print(uncertainty + 1.0*ad_uncertainty)
    # exit(-2)
    # if not random_tie_break:
    return multi_argmax(uncertainty + 0.1*ad_uncertainty, n_instances=n_instances)
    
    # return shuffled_argmax(uncertainty + 0.1*ad_uncertainty, n_instances=n_instances)


def edge_superpixels(predict_label, adjoin_sp, edge_weight=0.6):
    assert len(predict_label) == len(adjoin_sp), "预测标签数量与超像素数量不等"
    edge = []
    for i, ad in enumerate(adjoin_sp):
        adjoin_labels = []
        for s in ad:
            adjoin_labels.append(predict_label[s-1])
        adjoin_labels = set(adjoin_labels)
        # 边缘超像素edge_rate>1,内部超像素=1
        # edge_rate = np.log10(len(adjoin_labels))
        edge_rate = len(adjoin_labels) ** edge_weight
        edge.append(edge_rate)
        # if edge_rate > 1:
        #     edge.append(2.0)
        # else:
        #     edge.append(1.0)
    return np.array(edge)


def adjoin_uncertainty(adjoin_sp, uncertainty):
    ad_uncertainty = []
    for i, ad in enumerate(adjoin_sp):
        adjoin = []
        for s in ad:
            adjoin.append(uncertainty[s-1])
        ad_uncertainty.append(sum(adjoin))
    return np.array(ad_uncertainty)





def prob_sampling(classifier: BaseEstimator, X: modALinput,
                                    n_instances: int = 1, random_tie_break: bool = False,
                                    **uncertainty_measure_kwargs) -> np.ndarray:
    prob = classifier.predict_proba(X)
    trans_prob = select_prob(prob)
    return multi_argmax(trans_prob, n_instances=n_instances)
    # return shuffled_argmax(trans_prob, n_instances=n_instances)
    

def select_prob(prob):
    min_prob_gap = -1
    prob_list = []
    # select_idx = 0
    for pi in range(prob.shape[0]):
        s = multi_argmax(prob[pi], n_instances=2)
        gap = abs(s[0]-s[1])
        prob_list.append(1-gap)
    return np.array(prob_list)
    
    
    

def margin_sampling(classifier: BaseEstimator, X: modALinput,
                    n_instances: int = 1, random_tie_break: bool = False,
                    **uncertainty_measure_kwargs) -> np.ndarray:
    """
    Margin sampling query strategy. Selects the instances where the difference between
    the first most likely and second most likely classes are the smallest.
    Args:
        classifier: The classifier for which the labels are to be queried.
        X: The pool of samples to query from.
        n_instances: Number of samples to be queried.
        random_tie_break: If True, shuffles utility scores to randomize the order. This
            can be used to break the tie when the highest utility score is not unique.
        **uncertainty_measure_kwargs: Keyword arguments to be passed for the uncertainty
            measure function.
    Returns:
        The indices of the instances from X chosen to be labelled;
        the instances from X chosen to be labelled.
    """
    margin = classifier_margin(classifier, X, **uncertainty_measure_kwargs)

    if not random_tie_break:
        return multi_argmax(-margin, n_instances=n_instances)

    return shuffled_argmax(-margin, n_instances=n_instances)


def MCLU(classifier: BaseEstimator, X: modALinput,
                    n_instances: int = 1, random_tie_break: bool = False,
                    **uncertainty_measure_kwargs) -> np.ndarray:
    multi_dis = classifier.decision_function(X)
    # print(multi_dis[0:10])
    multi_dis = np.abs(multi_dis)
    # print(multi_dis.shape)

    if multi_dis.shape[1] == 1:
        print('multi_dis.shape[1] == 1')
        return np.zeros(shape=(multi_dis.shape[0],))
    part = np.partition(-multi_dis, 1, axis=1)
    # print(part[:, 0].shape)
    margin = abs(- part[:, 0] + part[:, 1])
    return shuffled_argmax(-margin, n_instances=n_instances)


def margin_sampling_density(classifier: BaseEstimator, X: modALinput, euclidean_density,
                    n_instances: int = 1, random_tie_break: bool = False,
                    **uncertainty_measure_kwargs) -> np.ndarray:

    margin = classifier_margin(classifier, X)
    
    
    return multi_argmax(-margin+euclidean_density, n_instances=n_instances)
    
    
def density_sampling(classifier: BaseEstimator, X: modALinput, euclidean_density,
                    n_instances: int = 1, random_tie_break: bool = False,
                    **uncertainty_measure_kwargs) -> np.ndarray:
    return multi_argmax(-euclidean_density, n_instances=n_instances)
    

def margin_sampling_with_adjoin(classifier: BaseEstimator, X: modALinput,adjoin_sp,
                    n_instances: int = 1, random_tie_break: bool = False,
                    **uncertainty_measure_kwargs) -> np.ndarray:
    margin = classifier_margin(classifier, X, **uncertainty_measure_kwargs)
    ad_margin = adjoin_uncertainty(adjoin_sp, margin)
    return multi_argmax(-margin - 0.1 * ad_margin, n_instances=n_instances)



def margin_sampling_with_weight(classifier: BaseEstimator, X: modALinput, predict_label, weight,
                    n_instances: int = 1, random_tie_break: bool = False,
                    **uncertainty_measure_kwargs) -> np.ndarray:
    margin = classifier_margin(classifier, X, **uncertainty_measure_kwargs)
    
    if not random_tie_break:
        
        # print("uncertainty_sampling_clustering")
        # 提取前n个不确定instance的下标
        result_arg = multi_argmax(-margin, n_instances=margin.shape[0])
        
        if predict_label != []:
            # 使用权重更新disagreement
            for i, r in enumerate(result_arg):
                # 判断该instance下标对应的簇类别
                instance_label = predict_label[i] - 1
                # print("instance_label-1:", instance_label-1)
                margin[r] = margin[r] * weight[instance_label]
        
        # 使用更新后的disagreement，提取加权后的最大分歧
        # final_arg = multi_argmax(uncertainty, n_instances=n_instances)
        return multi_argmax(-margin, n_instances=n_instances)
    
    return shuffled_argmax(-margin, n_instances=n_instances)



def margin_adjoin_refine(classifier: BaseEstimator, X: modALinput, adjoin_sp, adjoin_weight=0.1,
                    n_instances: int = 1, random_tie_break: bool = False,
                    **uncertainty_measure_kwargs) -> np.ndarray:
    margin = classifier_margin(classifier, X, **uncertainty_measure_kwargs)
    proba = classifier.predict_proba(X)
    margin_ad = []
    for i in range(len(adjoin_sp)):
        prob_idx = multi_argmax(proba[i], n_instances=2)
        # print('prob_idx', prob_idx.shape)
        difference = []
        for j in range(adjoin_sp[i].shape[0]):
            sp_idx = adjoin_sp[i][j] - 1
            sp_proba = proba[sp_idx]
            # print('sp_proba', sp_proba.shape)
            class_1st = prob_idx[0]
            class_2nd = prob_idx[1]
            difference.append(abs(sp_proba[class_1st]-sp_proba[class_2nd]))
        margin_ad.append(sum(difference))
    margin_ad = np.array(margin_ad)

    
    # ad_margin = adjoin_uncertainty(adjoin_sp, margin)
    return multi_argmax(-margin - adjoin_weight * margin_ad, n_instances=n_instances)

def margin_adjoin_uncertainty(classifier: BaseEstimator, X: modALinput, adjoin_sp,
                    n_instances: int = 1, random_tie_break: bool = False,
                    **uncertainty_measure_kwargs) -> np.ndarray:
    margin = classifier_margin(classifier, X, **uncertainty_measure_kwargs)
    uncertainty = classifier_uncertainty(classifier, X, **uncertainty_measure_kwargs)
    proba = classifier.predict_proba(X)
    uncertainty_ad = []
    for i in range(len(adjoin_sp)):
        prob_idx = multi_argmax(proba[i], n_instances=2)
        # print('prob_idx', prob_idx.shape)
        difference = []
        for j in range(adjoin_sp[i].shape[0]):
            sp_idx = adjoin_sp[i][j] - 1
            # sp_proba = proba[sp_idx]
            # uncert = uncertainty[sp_idx]
            difference.append(uncertainty[sp_idx])
        uncertainty_ad.append(sum(difference))
    uncertainty_ad = np.array(uncertainty_ad)
    return multi_argmax(-margin + 0.1 * uncertainty_ad, n_instances=n_instances)


def entropy_sampling(classifier: BaseEstimator, X: modALinput,
                     n_instances: int = 1, random_tie_break: bool = False,
                     **uncertainty_measure_kwargs) -> np.ndarray:
    """
    Entropy sampling query strategy. Selects the instances where the class probabilities
    have the largest entropy.

    Args:
        classifier: The classifier for which the labels are to be queried.
        X: The pool of samples to query from.
        n_instances: Number of samples to be queried.
        random_tie_break: If True, shuffles utility scores to randomize the order. This
            can be used to break the tie when the highest utility score is not unique.
        **uncertainty_measure_kwargs: Keyword arguments to be passed for the uncertainty
            measure function.

    Returns:
        The indices of the instances from X chosen to be labelled;
        the instances from X chosen to be labelled.
    """
    entropy = classifier_entropy(classifier, X, **uncertainty_measure_kwargs)
    
    if not random_tie_break:
        return multi_argmax(entropy, n_instances=n_instances)

    return shuffled_argmax(entropy, n_instances=n_instances)

def entropy_sampling_with_adjoin(classifier: BaseEstimator, X: modALinput, adjoin_sp,
                     n_instances: int = 1, random_tie_break: bool = False,
                     **uncertainty_measure_kwargs) -> np.ndarray:
    entropy = classifier_entropy(classifier, X, **uncertainty_measure_kwargs)
    ad_entropy = adjoin_uncertainty(adjoin_sp, entropy)
    return multi_argmax(entropy + 0.1*ad_entropy, n_instances=n_instances)