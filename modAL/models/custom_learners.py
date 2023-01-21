import numpy as np

from typing import Callable, Optional, Tuple, List, Any

from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score

from modAL.models.base import BaseLearner, BaseCommittee
from modAL.utils.validation import check_class_labels, check_class_proba
from modAL.utils.data import modALinput, retrieve_rows
from modAL.uncertainty import uncertainty_sampling
from modAL.disagreement import vote_entropy_sampling, max_std_sampling
from modAL.acquisition import max_EI
'''
# from modAL.custom_query_strategy import custom_vote_entropy_sampling

class ActiveLearner(BaseLearner):
    """
    This class is an abstract model of a general active learning algorithm.

    Args:
        estimator: The estimator to be used in the active learning loop.
        query_strategy: Function providing the query strategy for the active learning loop,
            for instance, modAL.uncertainty.uncertainty_sampling.
        X_training: Initial training samples, if available.
        y_training: Initial training labels corresponding to initial training samples.
        bootstrap_init: If initial training data is available, bootstrapping can be done during the first training.
            Useful when building Committee models with bagging.
        on_transformed: Whether to transform samples with the pipeline defined by the estimator
            when applying the query strategy.
        **fit_kwargs: keyword arguments.

    Attributes:
        estimator: The estimator to be used in the active learning loop.
        query_strategy: Function providing the query strategy for the active learning loop.
        X_training: If the model hasn't been fitted yet it is None, otherwise it contains the samples
            which the model has been trained on. If provided, the method fit() of estimator is called during __init__()
        y_training: The labels corresponding to X_training.

    Examples:

        >>> from sklearn.datasets import load_iris
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from modAL.models import ActiveLearner
        >>> iris = load_iris()
        >>> # give initial training examples
        >>> X_training = iris['data'][[0, 50, 100]]
        >>> y_training = iris['target'][[0, 50, 100]]
        >>>
        >>> # initialize active learner
        >>> learner = ActiveLearner(
        ...     estimator=RandomForestClassifier(),
        ...     X_training=X_training, y_training=y_training
        ... )
        >>>
        >>> # querying for labels
        >>> query_idx, query_sample = learner.query(iris['data'])
        >>>
        >>> # ...obtaining new labels from the Oracle...
        >>>
        >>> # teaching newly labelled examples
        >>> learner.teach(
        ...     X=iris['data'][query_idx].reshape(1, -1),
        ...     y=iris['target'][query_idx].reshape(1, )
        ... )
    """

    def __init__(self,
                 estimator: BaseEstimator,
                 query_strategy: Callable = uncertainty_sampling,
                 X_training: Optional[modALinput] = None,
                 y_training: Optional[modALinput] = None,
                 bootstrap_init: bool = False,
                 on_transformed: bool = False,
                 **fit_kwargs
                 ) -> None:
        super().__init__(estimator, query_strategy,
                         X_training, y_training, bootstrap_init, on_transformed, **fit_kwargs)

    def teach(self, X: modALinput, y: modALinput, bootstrap: bool = False, only_new: bool = False, **fit_kwargs) -> None:
        """
        Adds X and y to the known training data and retrains the predictor with the augmented dataset.

        Args:
            X: The new samples for which the labels are supplied by the expert.
            y: Labels corresponding to the new instances in X.
            bootstrap: If True, training is done on a bootstrapped dataset. Useful for building Committee models
                with bagging.
            only_new: If True, the model is retrained using only X and y, ignoring the previously provided examples.
                Useful when working with models where the .fit() method doesn't retrain the model from scratch (e. g. in
                tensorflow or keras).
            **fit_kwargs: Keyword arguments to be passed to the fit method of the predictor.
        """
        self._add_training_data(X, y)
        if not only_new:
            self._fit_to_known(bootstrap=bootstrap, **fit_kwargs)
        else:
            self._fit_on_new(X, y, bootstrap=bootstrap, **fit_kwargs)





class Custom_Committee(BaseCommittee):
    """
    This class is an abstract model of a committee-based active learning algorithm.

    Args:
        learner_list: A list of ActiveLearners forming the Committee.
        query_strategy: Query strategy function. Committee supports disagreement-based query strategies from
            :mod:`modAL.disagreement`, but uncertainty-based ones from :mod:`modAL.uncertainty` are also supported.
        on_transformed: Whether to transform samples with the pipeline defined by each learner's estimator
            when applying the query strategy.

    Attributes:
        classes_: Class labels known by the Committee.
        n_classes_: Number of classes known by the Committee.

    Examples:

        >>> from sklearn.datasets import load_iris
        >>> from sklearn.neighbors import KNeighborsClassifier
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from modAL.models import ActiveLearner, Committee
        >>>
        >>> iris = load_iris()
        >>>
        >>> # initialize ActiveLearners
        >>> learner_1 = ActiveLearner(
        ...     estimator=RandomForestClassifier(),
        ...     X_training=iris['data'][[0, 50, 100]], y_training=iris['target'][[0, 50, 100]]
        ... )
        >>> learner_2 = ActiveLearner(
        ...     estimator=KNeighborsClassifier(n_neighbors=3),
        ...     X_training=iris['data'][[1, 51, 101]], y_training=iris['target'][[1, 51, 101]]
        ... )
        >>>
        >>> # initialize the Committee
        >>> committee = Committee(
        ...     learner_list=[learner_1, learner_2]
        ... )
        >>>
        >>> # querying for labels
        >>> query_idx, query_sample = committee.query(iris['data'])
        >>>
        >>> # ...obtaining new labels from the Oracle...
        >>>
        >>> # teaching newly labelled examples
        >>> committee.teach(
        ...     X=iris['data'][query_idx].reshape(1, -1),
        ...     y=iris['target'][query_idx].reshape(1, )
        ... )
    """
    def __init__(self, learner_list: List[ActiveLearner], query_strategy: Callable = vote_entropy_sampling,
                 on_transformed: bool = False) -> None:
        super().__init__(learner_list, query_strategy, on_transformed)
        self._set_classes()

    def _set_classes(self):
        """
        Checks the known class labels by each learner, merges the labels and returns a mapping which maps the learner's
        classes to the complete label list.
        """
        # assemble the list of known classes from each learner
        try:
            # if estimators are fitted
            known_classes = tuple(learner.estimator.classes_ for learner in self.learner_list)
        except AttributeError:
            # handle unfitted estimators
            self.classes_ = None
            self.n_classes_ = 0
            return

        self.classes_ = np.unique(
            np.concatenate(known_classes, axis=0),
            axis=0
        )
        self.n_classes_ = len(self.classes_)

    def _add_training_data(self, X: modALinput, y: modALinput):
        super()._add_training_data(X, y)

    def fit(self, X: modALinput, y: modALinput, **fit_kwargs) -> 'BaseCommittee':
        """
        Fits every learner to a subset sampled with replacement from X. Calling this method makes the learner forget the
        data it has seen up until this point and replaces it with X! If you would like to perform bootstrapping on each
        learner using the data it has seen, use the method .rebag()!

        Calling this method makes the learner forget the data it has seen up until this point and replaces it with X!

        Args:
            X: The samples to be fitted on.
            y: The corresponding labels.
            **fit_kwargs: Keyword arguments to be passed to the fit method of the predictor.
        """
        super().fit(X, y, **fit_kwargs)
        self._set_classes()

    def teach(self, X: modALinput, y: modALinput, bootstrap: bool = False, only_new: bool = False, **fit_kwargs) -> None:
        """
        Adds X and y to the known training data for each learner and retrains learners with the augmented dataset.

        Args:
            X: The new samples for which the labels are supplied by the expert.
            y: Labels corresponding to the new instances in X.
            bootstrap: If True, trains each learner on a bootstrapped set. Useful when building the ensemble by bagging.
            only_new: If True, the model is retrained using only X and y, ignoring the previously provided examples.
            **fit_kwargs: Keyword arguments to be passed to the fit method of the predictor.
        """
        super().teach(X, y, bootstrap=bootstrap, only_new=only_new, **fit_kwargs)
        self._set_classes()

    def predict(self, X: modALinput, **predict_proba_kwargs) -> Any:
        """
        Predicts the class of the samples by picking the consensus prediction.

        Args:
            X: The samples to be predicted.
            **predict_proba_kwargs: Keyword arguments to be passed to the :meth:`predict_proba` of the Committee.

        Returns:
            The predicted class labels for X.
        """
        # getting average certainties
        proba = self.predict_proba(X, **predict_proba_kwargs)
        # finding the sample-wise max probability
        max_proba_idx = np.argmax(proba, axis=1)
        # translating label indices to labels
        return self.classes_[max_proba_idx]

    def predict_proba(self, X: modALinput, **predict_proba_kwargs) -> Any:
        """
        Consensus probabilities of the Committee.

        Args:
            X: The samples for which the class probabilities are to be predicted.
            **predict_proba_kwargs: Keyword arguments to be passed to the :meth:`predict_proba` of the Committee.

        Returns:
            Class probabilities for X.
        """
        return np.mean(self.vote_proba(X, **predict_proba_kwargs), axis=1)

    def score(self, X: modALinput, y: modALinput, sample_weight: List[float] = None) -> Any:
        """
        Returns the mean accuracy on the given test data and labels.

        Todo:
            Why accuracy?

        Args:
            X: The samples to score.
            y: Ground truth labels corresponding to X.
            sample_weight: Sample weights.

        Returns:
            Mean accuracy of the classifiers.
        """
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred, sample_weight=sample_weight)

    def vote(self, X: modALinput, **predict_kwargs) -> Any:
        """
        Predicts the labels for the supplied data for each learner in the Committee.

        Args:
            X: The samples to cast votes.
            **predict_kwargs: Keyword arguments to be passed to the :meth:`predict` of the learners.

        Returns:
            The predicted class for each learner in the Committee and each sample in X.
        """
        # print("custom vote")
        prediction = np.zeros(shape=(X.shape[0], len(self.learner_list)))

        for learner_idx, learner in enumerate(self.learner_list):
            prediction[:, learner_idx] = learner.predict(X, **predict_kwargs)

        return prediction

    def vote_proba(self, X: modALinput, **predict_proba_kwargs) -> Any:
        """
        Predicts the probabilities of the classes for each sample and each learner.

        Args:
            X: The samples for which class probabilities are to be calculated.
            **predict_proba_kwargs: Keyword arguments for the :meth:`predict_proba` of the learners.

        Returns:
            Probabilities of each class for each learner and each instance.
        """

        # get dimensions
        n_samples = X.shape[0]
        n_learners = len(self.learner_list)
        proba = np.zeros(shape=(n_samples, n_learners, self.n_classes_))

        # checking if the learners in the Committee know the same set of class labels
        if check_class_labels(*[learner.estimator for learner in self.learner_list]):
            # known class labels are the same for each learner
            # probability prediction is straightforward

            for learner_idx, learner in enumerate(self.learner_list):
                proba[:, learner_idx, :] = learner.predict_proba(X, **predict_proba_kwargs)

        else:
            for learner_idx, learner in enumerate(self.learner_list):
                proba[:, learner_idx, :] = check_class_proba(
                    proba=learner.predict_proba(X, **predict_proba_kwargs),
                    known_labels=learner.estimator.classes_,
                    all_labels=self.classes_
                )

        return proba

'''