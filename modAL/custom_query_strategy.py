
from modAL.uncertainty import *
# import numpy as np
from modAL.models.base import BaseCommittee
from modAL.utils.data import modALinput
from collections import Counter

def random_sampling(classifier, X_pool):
    n_samples = len(X_pool)
    query_idx = np.random.choice(range(n_samples))
    return [query_idx], X_pool[query_idx]

'''
def custom_vote(learner_list, X_pool):
    prediction = np.zeros(shape=(X_pool.shape[0], len(learner_list)))
    for learner_idx, learner in enumerate(learner_list):
        prediction[:, learner_idx] = learner.predict(X_pool)
    return prediction

def custom_vote_entropy(committee: BaseCommittee, X: modALinput):
    n_learners = len(committee)
    try:
        votes = committee.vote(X)
    except NotFittedError:
        return np.zeros(shape=(X.shape[0],))
    
    p_vote = np.zeros(shape=(X.shape[0], len(committee.classes_)))
    
    for vote_idx, vote in enumerate(votes):
        vote_counter = Counter(vote)
        
        for class_idx, class_label in enumerate(committee.classes_):
            p_vote[vote_idx, class_idx] = vote_counter[class_label] / n_learners
    
    entr = entropy(p_vote, axis=1)
    return entr

def custom_vote_entropy_sampling(committee: BaseCommittee, X: modALinput,
                          n_instances: int = 1)-> np.ndarray:
    disagreement = custom_vote_entropy(committee, X)
    
    return shuffled_argmax(disagreement, n_instances=n_instances)
    
'''