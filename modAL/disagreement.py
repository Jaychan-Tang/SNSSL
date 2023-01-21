"""
Disagreement measures and disagreement based query strategies for the Committee model.
"""
from collections import Counter
from typing import Tuple

import numpy as np
from scipy.stats import entropy
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator

from modAL.utils.data import modALinput
from modAL.utils.selection import multi_argmax, shuffled_argmax
from modAL.models.base import BaseCommittee


def vote_entropy(committee: BaseCommittee, X: modALinput, **predict_proba_kwargs) -> np.ndarray:
    """
    Calculates the vote entropy for the Committee. First it computes the predictions of X for each learner in the
    Committee, then calculates the probability distribution of the votes. The entropy of this distribution is the vote
    entropy of the Committee, which is returned.

    Args:
        committee: The :class:`modAL.models.BaseCommittee` instance for which the vote entropy is to be calculated.
        X: The data for which the vote entropy is to be calculated.
        **predict_proba_kwargs: Keyword arguments for the :meth:`predict_proba` of the Committee.

    Returns:
        Vote entropy of the Committee for the samples in X.
    """
    n_learners = len(committee)
    try:
        votes = committee.vote(X, **predict_proba_kwargs)
    except NotFittedError:
        return np.zeros(shape=(X.shape[0],))

    p_vote = np.zeros(shape=(X.shape[0], len(committee.classes_)))

    for vote_idx, vote in enumerate(votes):
        vote_counter = Counter(vote)

        for class_idx, class_label in enumerate(committee.classes_):
            p_vote[vote_idx, class_idx] = vote_counter[class_label]/n_learners

    entr = entropy(p_vote, axis=1)
    return entr


def consensus_entropy(committee: BaseCommittee, X: modALinput, **predict_proba_kwargs) -> np.ndarray:
    """
    Calculates the consensus entropy for the Committee. First it computes the class probabilties of X for each learner
    in the Committee, then calculates the consensus probability distribution by averaging the individual class
    probabilities for each learner. The entropy of the consensus probability distribution is the vote entropy of the
    Committee, which is returned.

    Args:
        committee: The :class:`modAL.models.BaseCommittee` instance for which the consensus entropy is to be calculated.
        X: The data for which the consensus entropy is to be calculated.
        **predict_proba_kwargs: Keyword arguments for the :meth:`predict_proba` of the Committee.

    Returns:
        Consensus entropy of the Committee for the samples in X.
    """
    try:
        proba = committee.predict_proba(X, **predict_proba_kwargs)
    except NotFittedError:
        return np.zeros(shape=(X.shape[0],))

    entr = np.transpose(entropy(np.transpose(proba)))
    return entr


def KL_max_disagreement(committee: BaseCommittee, X: modALinput, **predict_proba_kwargs) -> np.ndarray:
    """
    Calculates the max disagreement for the Committee. First it computes the class probabilties of X for each learner in
    the Committee, then calculates the consensus probability distribution by averaging the individual class
    probabilities for each learner. Then each learner's class probabilities are compared to the consensus distribution
    in the sense of Kullback-Leibler divergence. The max disagreement for a given sample is the argmax of the KL
    divergences of the learners from the consensus probability.

    Args:
        committee: The :class:`modAL.models.BaseCommittee` instance for which the max disagreement is to be calculated.
        X: The data for which the max disagreement is to be calculated.
        **predict_proba_kwargs: Keyword arguments for the :meth:`predict_proba` of the Committee.

    Returns:
        Max disagreement of the Committee for the samples in X.
    """
    try:
        p_vote = committee.vote_proba(X, **predict_proba_kwargs)
    except NotFittedError:
        return np.zeros(shape=(X.shape[0],))

    p_consensus = np.mean(p_vote, axis=1)

    learner_KL_div = np.zeros(shape=(X.shape[0], len(committee)))
    for learner_idx, _ in enumerate(committee):
        learner_KL_div[:, learner_idx] = entropy(np.transpose(p_vote[:, learner_idx, :]), qk=np.transpose(p_consensus))

    return np.max(learner_KL_div, axis=1)


def vote_entropy_sampling(committee: BaseCommittee, X: modALinput,
                          n_instances: int = 1, random_tie_break=False,
                          **disagreement_measure_kwargs) -> np.ndarray:
    """
    Vote entropy sampling strategy.

    Args:
        committee: The committee for which the labels are to be queried.
        X: The pool of samples to query from.
        n_instances: Number of samples to be queried.
        random_tie_break: If True, shuffles utility scores to randomize the order. This
            can be used to break the tie when the highest utility score is not unique.
        **disagreement_measure_kwargs: Keyword arguments to be passed for the disagreement
            measure function.

    Returns:
        The indices of the instances from X chosen to be labelled;
         the instances from X chosen to be labelled.
    """
    disagreement = vote_entropy(committee, X, **disagreement_measure_kwargs)

    if not random_tie_break:
        return multi_argmax(disagreement, n_instances=n_instances)

    return shuffled_argmax(disagreement, n_instances=n_instances)


def consensus_entropy_sampling(committee: BaseCommittee, X: modALinput,
                               n_instances: int = 1, random_tie_break=False,
                               **disagreement_measure_kwargs) -> np.ndarray:
    """
    Consensus entropy sampling strategy.

    Args:
        committee: The committee for which the labels are to be queried.
        X: The pool of samples to query from.
        n_instances: Number of samples to be queried.
        random_tie_break: If True, shuffles utility scores to randomize the order. This
            can be used to break the tie when the highest utility score is not unique.
        **disagreement_measure_kwargs: Keyword arguments to be passed for the disagreement
            measure function.

    Returns:
        The indices of the instances from X chosen to be labelled;
        the instances from X chosen to be labelled.
    """
    disagreement = consensus_entropy(committee, X, **disagreement_measure_kwargs)

    if not random_tie_break:
        return multi_argmax(disagreement, n_instances=n_instances)

    return shuffled_argmax(disagreement, n_instances=n_instances)


def max_disagreement_sampling(committee: BaseCommittee, X: modALinput,
                              n_instances: int = 1, random_tie_break=False,
                              **disagreement_measure_kwargs) -> np.ndarray:
    """
    Maximum disagreement sampling strategy.

    Args:
        committee: The committee for which the labels are to be queried.
        X: The pool of samples to query from.
        n_instances: Number of samples to be queried.
        random_tie_break: If True, shuffles utility scores to randomize the order. This
            can be used to break the tie when the highest utility score is not unique.
        **disagreement_measure_kwargs: Keyword arguments to be passed for the disagreement
         measure function.

    Returns:
        The indices of the instances from X chosen to be labelled;
        the instances from X chosen to be labelled.
    """
    disagreement = KL_max_disagreement(committee, X, **disagreement_measure_kwargs)

    if not random_tie_break:
        return multi_argmax(disagreement, n_instances=n_instances)

    return shuffled_argmax(disagreement, n_instances=n_instances)


def max_std_sampling(regressor: BaseEstimator, X: modALinput,
                     n_instances: int = 1,  random_tie_break=False,
                     **predict_kwargs) -> np.ndarray:
    """
    Regressor standard deviation sampling strategy.

    Args:
        regressor: The regressor for which the labels are to be queried.
        X: The pool of samples to query from.
        n_instances: Number of samples to be queried.
        random_tie_break: If True, shuffles utility scores to randomize the order. This
            can be used to break the tie when the highest utility score is not unique.
        **predict_kwargs: Keyword arguments to be passed to :meth:`predict` of the CommiteeRegressor.

    Returns:
        The indices of the instances from X chosen to be labelled;
        the instances from X chosen to be labelled.
    """
    _, std = regressor.predict(X, return_std=True, **predict_kwargs)
    std = std.reshape(X.shape[0], )

    if not random_tie_break:
        return multi_argmax(std, n_instances=n_instances)

    return shuffled_argmax(std, n_instances=n_instances)



# 自定义投票熵
def balance_vote_entropy(committee: BaseCommittee, X: modALinput, **predict_proba_kwargs) -> np.ndarray:
    n_learners = len(committee)
    try:
        votes = committee.vote(X, **predict_proba_kwargs)
    except NotFittedError:
        return np.zeros(shape=(X.shape[0],))

    p_vote = np.zeros(shape=(X.shape[0], len(committee.classes_)))
    # print(p_vote.shape)
    predict_label = []
    for vote_idx, vote in enumerate(votes):
        vote_counter = Counter(vote)

        
        vote_most = vote_counter.most_common(1)
        predict_label.append(vote_most[0][0])
        # print(vote_counter)
        # Counter({1.0: 3, 4.0: 2})
        for class_idx, class_label in enumerate(committee.classes_):
            # print(vote_counter[class_label] / n_learners)
            p_vote[vote_idx, class_idx] = vote_counter[class_label]/n_learners

        # exit(-2)
        # print(p_vote)
    entr = entropy(p_vote, axis=1)
    return entr, predict_label


# 自定义委员会投票熵
def balance_vote_entropy_sampling(committee: BaseCommittee, X: modALinput,
                          n_instances: int = 1, random_tie_break=False, weight: np.ndarray = 0,
                          **disagreement_measure_kwargs) -> np.ndarray:
    
    disagreement, predict_label = balance_vote_entropy(committee, X, **disagreement_measure_kwargs)
    # print(disagreement.shape, disagreement)
    # print(type(disagreement))
    if not random_tie_break:
        # 提取前n个不确定instance的下标
        result_arg = multi_argmax(disagreement, n_instances=500)
        
        # 使用权重更新disagreement
        for r in result_arg:
            # 判断该instance下标对应的类别
            instance_label = int(predict_label[r] - 1)
            # print("instance_label-1:", instance_label-1)
            disagreement[r] = disagreement[r] * weight[instance_label-1]
        
        # 使用更新后的disagreement，提取加权后的最大分歧
        final_arg = multi_argmax(disagreement, n_instances=n_instances)
        
        # 权重不由该函数维护
        # final_class = predict_label[final_arg] - 1
        # weight[final_class-1] = 1
        
        return final_arg
    return shuffled_argmax(disagreement, n_instances=n_instances)


