import numbers
import threading
from numbers import Integral, Real
from warnings import warn
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

import numpy as np
from scipy.sparse import issparse

from sklearn.base import OutlierMixin
from sklearn.utils._chunking import get_chunk_n_rows
#from sklearn.utils.validation import _fit_context
from sklearn.tree import ExtraTreeRegressor
from sklearn.tree._tree import DTYPE as tree_dtype
from sklearn.utils import (
    check_array,
    check_random_state,
    gen_batches,
)
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import _num_samples, check_is_fitted
from sklearn.ensemble._bagging import BaseBagging
from sklearn.ensemble._base import _partition_estimators

def get_node_path(tree, node_id):
    """
    Given a tree and a node_id, returns the path from the node to the root.

    Parameters:
    - tree: The tree structure (tree_ attribute of a DecisionTree).
    - node_id: The ID of the node for which the path is to be found.

    Returns:
    - path: A list of node IDs representing the path from the node to the root.
    """
    path = []
    current_node = node_id
    
    while current_node != 0:  # 0 is the root node
        path.append(current_node)
        parent_node = None
        for node in range(tree.node_count):
            if (tree.children_left[node] == current_node or 
                tree.children_right[node] == current_node):
                parent_node = node
                break
        if parent_node is None:
            raise ValueError(f"Parent node not found for node {current_node}")
        current_node = parent_node
    
    path.append(0)  # Append the root node
    path.reverse()  # Reverse the path to have it from root to the node
    
    return path

def distance_from_root(tree, node_id):
    """
    Finds the distance from a given node to the root in a decision tree.
    
    Parameters:
    - tree: The tree structure (tree_ attribute of a DecisionTree).
    - node_id: The ID of the node for which the distance is to be found.
    
    Returns:
    - distance: The number of edges between the node and the root.
    """
    path = get_node_path(tree, node_id)
    distance = len(path) - 1  # Subtract 1 because the path includes the root node itself
    return distance

def find_lca_for_sublist(tree, sublist):
    """
    Find the lowest common ancestor (LCA) of the nodes in the given sublist.

    Parameters:
    - tree: The tree structure (tree_ attribute of a DecisionTree).
    - sublist: A list of node IDs for which the LCA is to be found.

    Returns:
    - lca: The node ID of the lowest common ancestor.
    """
    if len(sublist) == 1:
        # If the sublist contains only one node, return that node as the LCA
        return sublist[0]
    
    # Get the paths from each node to the root
    paths = [get_node_path(tree, node_id) for node_id in sublist]
    
    # Initialize the LCA as the first node in the path (root)
    lca = paths[0][0]
    
    # Iterate over the paths and find the common ancestor
    path1 = paths[0]
    path2 = paths[-1]
    
    # Find the minimum length between the two paths
    min_length = min(len(path1), len(path2))
    
    # Find the last common node in the two paths
    for j in range(min_length):
        if path1[j] != path2[j]:
            break
        lca = path1[j]
    
    return lca

def compute_lcas(tree, leaves_sublists):
    
    """
    Compute the LCA for each sublist in leaves_sublists.
    
    Parameters:
    - tree: The tree structure (tree_ attribute of a DecisionTree).
    - leaves_sublists: A list of sublists where each sublist contains node IDs.
    
    Returns:
    - lcas: A list of LCAs, one for each sublist.
    """
    lcas = []
    
    for sublist in leaves_sublists:
        lca = find_lca_for_sublist(tree.tree_, sublist)
        lcas.append(lca)
    
    return lcas

def average_path_length_per_tree(n_samples_leaf):
    """
    Compute the average path length in a n_samples iTree, which is equal to
    the average path length of an unsuccessful BST search since the
    latter has the same structure as an isolation tree.

    Parameters
    ----------
    n_samples_leaf : int
        The number of training samples in the leaf.

    Returns
    -------
    average_path_length : float
        The average path length for the given number of samples.
    """
    printx('n_smaples_leaf', n_samples_leaf)
    if len(n_samples_leaf) <= 1:
        return 0.0
    elif n_samples_leaf == 2:
        return 1.0
    else:
        return (
            2.0 * (np.log(n_samples_leaf - 1.0) + np.euler_gamma)
            - 2.0 * (n_samples_leaf - 1.0) / n_samples_leaf
        )

def _average_path_length(n_samples_leaf):
    """
    The average path length in a n_samples iTree, which is equal to
    the average path length of an unsuccessful BST search since the
    latter has the same structure as an isolation tree.
    Parameters
    ----------
    n_samples_leaf : array-like of shape (n_samples,)
        The number of training samples in each test sample leaf, for
        each estimators.

    Returns
    -------
    average_path_length : ndarray of shape (n_samples,)
    """

    n_samples_leaf = check_array(n_samples_leaf, ensure_2d=False)

    n_samples_leaf_shape = n_samples_leaf.shape
    n_samples_leaf = n_samples_leaf.reshape((1, -1))
    average_path_length = np.zeros(n_samples_leaf.shape)

    mask_1 = n_samples_leaf <= 1
    mask_2 = n_samples_leaf == 2
    not_mask = ~np.logical_or(mask_1, mask_2)

    average_path_length[mask_1] = 0.0
    average_path_length[mask_2] = 1.0
    average_path_length[not_mask] = (
        2.0 * (np.log(n_samples_leaf[not_mask] - 1.0) + np.euler_gamma)
        - 2.0 * (n_samples_leaf[not_mask] - 1.0) / n_samples_leaf[not_mask]
    )

    return average_path_length.reshape(n_samples_leaf_shape)

def map_leaves_to_datapoints(leaves_index):
    """
    Function to map each leaf to the set of data points that map to it.

    Parameters:
    - tree: A fitted tree object (e.g., ExtraTreeRegressor).
    - X: The input data used to apply to the tree.

    Returns:
    - leaf_mapping: A dictionary where keys are leaf indices and values are sets of data point indices.
    """

    # Initialize a dictionary to store the mapping from leaves to data points
    leaf_mapping = {}

    # Loop through each data point and its corresponding leaf index
    for data_point_idx, leaf_idx in enumerate(leaves_index):
        # If the leaf index is not in the dictionary, add it with an empty set
        if leaf_idx not in leaf_mapping:
            leaf_mapping[leaf_idx] = set()
        
        # Add the data point index to the set corresponding to this leaf
        leaf_mapping[leaf_idx].add(data_point_idx)

    return leaf_mapping

def subdivide_leaves_index(reordered_leaves_index, ipv4_sublists):
    leaves_sublists = []
    current_index = 0
    
    for sublist in ipv4_sublists:
        # Get the length of the current ipv4 sublist
        sublist_length = len(sublist)
        
        # Slice the reordered_leaves_index according to the sublist length
        leaves_sublists.append(reordered_leaves_index[current_index:current_index + sublist_length])
        
        # Move the current index forward by the length of the current sublist
        current_index += sublist_length
    
    return leaves_sublists

def printx(*args, **kwargs):
    if should_print:
        print(*args, **kwargs)

def divide_into_sublists(ipv4_list):
    sublists = []
    current_sublist = [ipv4_list[0]]  # Start with the first element

    for i in range(1, len(ipv4_list)):
        if ipv4_list[i] == ipv4_list[i-1]:
            # If the current element is equal to the previous, add to current sublist
            current_sublist.append(ipv4_list[i])
        else:
            # If not, end the current sublist and start a new one
            sublists.append(current_sublist)
            current_sublist = [ipv4_list[i]]

    # Add the last sublist
    sublists.append(current_sublist)
    
    return sublists

def _parallel_compute_tree_depths(
    tree,
    X,
    features,
    tree_decision_path_lengths,
    tree_avg_path_lengths,
    depths,
    ipv4_depths,
    lock,
    df,
):
    """Parallel computation of isolation tree depth."""
    if features is None:
        X_subset = X
    else:
        X_subset = X[:, features]

    leaves_index = tree.apply(X_subset, check_input=False)
    #printx(leaves_index)
    leaves_mapping = map_leaves_to_datapoints(leaves_index)
    #printx(leaves_mapping)

    leaves_list = []
    for j in sorted(leaves_mapping.keys()):
        s = leaves_mapping[j]
        leaves_list += s
        #printx('leaves_list:', leaves_list)
        #ipv4_list = [df.loc[l, 'ipv4'] for l in leaves_list]
        ipv4_list = [df.loc[l, 'ip_id'] for l in leaves_list]
        printx('ipv4_list:', ipv4_list)
        ipv4_sublists = divide_into_sublists(ipv4_list)
        #printx('ipv4_sublists ',ipv4_sublists)
        ipv4_keys = [l[0] for l in ipv4_sublists]
        #printx("ipv4_keys", ipv4_keys)

    reordered_leaves_index = [leaves_index[l] for l in leaves_list]
    #printx("Reordered leaves_index:", reordered_leaves_index)

    leaves_sublists = subdivide_leaves_index(reordered_leaves_index, ipv4_sublists)
    printx('leaves_sublists (parallel)',leaves_sublists)

    # Compute the LCAs for each sublist in leaves_sublists
    lcas = compute_lcas(tree, leaves_sublists)
    printx('lcas',lcas)

    ipv4_distances_to_root = [distance_from_root(tree.tree_, n) for n in lcas]
    printx('ipv4_distances_to_root',ipv4_distances_to_root)

    
    with lock:
        #num_samples_per_leaf.append([len(l) for l in leaves_sublists])
        printx("{ipv4_keys[i]:len(leaves_sublists[i]) for i in range(len(ipv4_keys))}", {ipv4_keys[i]:len(leaves_sublists[i]) for i in range(len(ipv4_keys))})
        printx('ipv4_keys',ipv4_keys)
        printx('leaves_sublists',leaves_sublists)
        printx('range(len(ipv4_keys))',range(len(ipv4_keys)))
        #num_samples_per_leaf.append({ipv4_keys[i]:len(leaves_sublists[i]) for i in range(len(ipv4_keys))})
        '''num_samples_per_leaf.append([(ipv4_keys[i], len(leaves_sublists[i])) for i in range(len(ipv4_keys))])
        #ipv4_avg_path_length_per_tree += ([_average_path_length([len(l) for l in leaves_sublists])])
        ipv4_avg_path_length_per_tree += [[(ipv4_keys[i], avg_path_lengths[i]) for i in range(len(ipv4_keys))]]'''
        avg_path_lengths = _average_path_length([len(l) for l in leaves_sublists])

        #ipv4_depths.append([(ipv4_keys[i], len(leaves_sublists[i]+avg_path_lengths[i]-1)) for i in range(len(ipv4_keys))])
        ipv4_depths.append([(ipv4_keys[i], ipv4_distances_to_root[i]+avg_path_lengths[i]-1) for i in range(len(ipv4_keys))])

    with lock:
        depths += (
            tree_decision_path_lengths[leaves_index]
            + tree_avg_path_lengths[leaves_index]
            - 1.0
        )

should_print = False
should_draw = False

__all__ = ["CustomIsolationForest"]

class CustomIsolationForest(OutlierMixin, BaseBagging):
    """
    Isolation Forest Algorithm.

    Return the anomaly score of each sample using the IsolationForest algorithm

    The IsolationForest 'isolates' observations by randomly selecting a feature
    and then randomly selecting a split value between the maximum and minimum
    values of the selected feature.

    Since recursive partitioning can be represented by a tree structure, the
    number of splittings required to isolate a sample is equivalent to the path
    length from the root node to the terminating node.

    This path length, averaged over a forest of such random trees, is a
    measure of normality and our decision function.

    Random partitioning produces noticeably shorter paths for anomalies.
    Hence, when a forest of random trees collectively produce shorter path
    lengths for particular samples, they are highly likely to be anomalies.

    Read more in the :ref:`User Guide <isolation_forest>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    n_estimators : int, default=100
        The number of base estimators in the ensemble.

    max_samples : "auto", int or float, default="auto"
        The number of samples to draw from X to train each base estimator.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.
            - If "auto", then `max_samples=min(256, n_samples)`.

        If max_samples is larger than the number of samples provided,
        all samples will be used for all trees (no sampling).

    contamination : 'auto' or float, default='auto'
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. Used when fitting to define the threshold
        on the scores of the samples.

            - If 'auto', the threshold is determined as in the
              original paper.
            - If float, the contamination should be in the range (0, 0.5].

        .. versionchanged:: 0.22
           The default value of ``contamination`` changed from 0.1
           to ``'auto'``.

    max_features : int or float, default=1.0
        The number of features to draw from X to train each base estimator.

            - If int, then draw `max_features` features.
            - If float, then draw `max(1, int(max_features * n_features_in_))` features.

        Note: using a float number less than 1.0 or integer less than number of
        features will enable feature subsampling and leads to a longer runtime.

    bootstrap : bool, default=False
        If True, individual trees are fit on random subsets of the training
        data sampled with replacement. If False, sampling without replacement
        is performed.

    n_jobs : int, default=None
        The number of jobs to run in parallel for both :meth:`fit` and
        :meth:`predict`. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors. See :term:`Glossary <n_jobs>` for more details.

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo-randomness of the selection of the feature
        and split values for each branching step and each tree in the forest.

        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    verbose : int, default=0
        Controls the verbosity of the tree building process.

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`the Glossary <warm_start>`.

        .. versionadded:: 0.21

    Attributes
    ----------
    estimator_ : :class:`~sklearn.tree.ExtraTreeRegressor` instance
        The child estimator template used to create the collection of
        fitted sub-estimators.

        .. versionadded:: 1.2
           `base_estimator_` was renamed to `estimator_`.

    estimators_ : list of ExtraTreeRegressor instances
        The collection of fitted sub-estimators.

    estimators_features_ : list of ndarray
        The subset of drawn features for each base estimator.

    estimators_samples_ : list of ndarray
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator.

    max_samples_ : int
        The actual number of samples.

    offset_ : float
        Offset used to define the decision function from the raw scores. We
        have the relation: ``decision_function = score_samples - offset_``.
        ``offset_`` is defined as follows. When the contamination parameter is
        set to "auto", the offset is equal to -0.5 as the scores of inliers are
        close to 0 and the scores of outliers are close to -1. When a
        contamination parameter different than "auto" is provided, the offset
        is defined in such a way we obtain the expected number of outliers
        (samples with decision function < 0) in training.

        .. versionadded:: 0.20

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    sklearn.covariance.EllipticEnvelope : An object for detecting outliers in a
        Gaussian distributed dataset.
    sklearn.svm.OneClassSVM : Unsupervised Outlier Detection.
        Estimate the support of a high-dimensional distribution.
        The implementation is based on libsvm.
    sklearn.neighbors.LocalOutlierFactor : Unsupervised Outlier Detection
        using Local Outlier Factor (LOF).

    Notes
    -----
    The implementation is based on an ensemble of ExtraTreeRegressor. The
    maximum depth of each tree is set to ``ceil(log_2(n))`` where
    :math:`n` is the number of samples used to build the tree
    (see (Liu et al., 2008) for more details).

    References
    ----------
    .. [1] Liu, Fei Tony, Ting, Kai Ming and Zhou, Zhi-Hua. "Isolation forest."
           Data Mining, 2008. ICDM'08. Eighth IEEE International Conference on.
    .. [2] Liu, Fei Tony, Ting, Kai Ming and Zhou, Zhi-Hua. "Isolation-based
           anomaly detection." ACM Transactions on Knowledge Discovery from
           Data (TKDD) 6.1 (2012): 3.

    Examples
    --------
    >>> from sklearn.ensemble import IsolationForest
    >>> X = [[-1.1], [0.3], [0.5], [100]]
    >>> clf = IsolationForest(random_state=0).fit(X)
    >>> clf.predict([[0.1], [0], [90]])
    array([ 1,  1, -1])

    For an example of using isolation forest for anomaly detection see
    :ref:`sphx_glr_auto_examples_ensemble_plot_isolation_forest.py`.
    """

    _parameter_constraints: dict = {
        "n_estimators": [Interval(Integral, 1, None, closed="left")],
        "max_samples": [
            StrOptions({"auto"}),
            Interval(Integral, 1, None, closed="left"),
            Interval(Real, 0, 1, closed="right"),
        ],
        "contamination": [
            StrOptions({"auto"}),
            Interval(Real, 0, 0.5, closed="right"),
        ],
        "max_features": [
            Integral,
            Interval(Real, 0, 1, closed="right"),
        ],
        "bootstrap": ["boolean"],
        "n_jobs": [Integral, None],
        "random_state": ["random_state"],
        "verbose": ["verbose"],
        "warm_start": ["boolean"],
    }

    def __init__(
        self,
        *,
        n_estimators=100,
        max_samples="auto",
        contamination="auto",
        max_features=1.0,
        bootstrap=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ipv4_index=None,
        min_score=False,
        df=None
    ):
        super().__init__(
            estimator=None,
            # here above max_features has no links with self.max_features
            bootstrap=bootstrap,
            bootstrap_features=False,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )
        self.base_estimator_ = ExtraTreeRegressor()
        self.contamination = contamination
        self.df=df 

    def _get_estimator(self):
        return ExtraTreeRegressor(
            # here max_features has no links with self.max_features
            max_features=1,
            splitter="random",
            random_state=self.random_state,
        )

    def _set_oob_score(self, X, y):
        raise NotImplementedError("OOB score not supported by iforest")

    def _parallel_args(self):
        # ExtraTreeRegressor releases the GIL, so it's more efficient to use
        # a thread-based backend rather than a process-based backend so as
        # to avoid suffering from communication overhead and extra memory
        # copies. This is only used in the fit method.
        return {"prefer": "threads"}

    #@_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None, sample_weight=None):
        """
        Fit estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Use ``dtype=np.float32`` for maximum
            efficiency. Sparse matrices are also supported, use sparse
            ``csc_matrix`` for maximum efficiency.

        y : Ignored
            Not used, present for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        def check_finite(X):
            if not np.all(np.isfinite(X)):
                raise ValueError("Input contains NaN or infinite values")
            return X
        #print("################################################################")
        #print(X)
        X = check_finite(X)
        X = self._validate_data(
            X, accept_sparse=["csc"], dtype=tree_dtype
        )
        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()
        printx('X: ', X)
        rnd = check_random_state(self.random_state)
        y = rnd.uniform(size=X.shape[0])

        # ensure that max_sample is in [1, n_samples]:
        n_samples = X.shape[0]

        if isinstance(self.max_samples, str) and self.max_samples == "auto":
            max_samples = min(256, n_samples)

        elif isinstance(self.max_samples, numbers.Integral):
            if self.max_samples > n_samples:
                warn(
                    "max_samples (%s) is greater than the "
                    "total number of samples (%s). max_samples "
                    "will be set to n_samples for estimation."
                    % (self.max_samples, n_samples)
                )
                max_samples = n_samples
            else:
                max_samples = self.max_samples
        else:  # max_samples is float
            max_samples = int(self.max_samples * X.shape[0])

        self.max_samples_ = max_samples
        max_depth = int(np.ceil(np.log2(max(max_samples, 2))))
        super()._fit(
            X,
            y,
            max_samples,
            max_depth=max_depth,
            sample_weight=sample_weight,
            check_input=False,
        )

        if should_draw:
            for tree in self.estimators_:
                plt.figure(figsize=(20, 10))  # Adjust the figure size as needed
                plot_tree(tree, filled=True, feature_names=None, rounded=True)
                plt.show()

        self._average_path_length_per_tree, self._decision_path_lengths = zip(
            *[
                (
                    _average_path_length(tree.tree_.n_node_samples),
                    tree.tree_.compute_node_depths(),
                )
                for tree in self.estimators_
            ]
        )

        if self.contamination == "auto":
            # 0.5 plays a special role as described in the original paper.
            # we take the opposite as we consider the opposite of their score.
            self.offset_ = -0.5
            return self

        # Else, define offset_ wrt contamination parameter
        # To avoid performing input validation a second time we call
        # _score_samples rather than score_samples.
        # _score_samples expects a CSR matrix, so we convert if necessary.
        if issparse(X):
            X = X.tocsr()
        self.offset_ = np.percentile(self._score_samples(X), 100.0 * self.contamination)

        return self

    def predict(self, X, min_score=False):
        """
        Predict if a particular sample is an outlier or not.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        is_inlier : dict
            For each observation, tells whether or not (+1 or -1) it should
            be considered as an inlier according to the fitted model.
            The keys are the sample identifiers and the values are +1 or -1.
        """
        self.min_score = min_score
        check_is_fitted(self)
        decision_func = self.decision_function(X)

        # Initialize a dictionary to hold the inlier/outlier predictions
        is_inlier = {}

        # Iterate over each sample's decision function output
        for sample_id, score in decision_func.items():
            if score < 0:
                is_inlier[sample_id] = -1  # Mark as outlier if the score is negative
            else:
                is_inlier[sample_id] = 1   # Mark as inlier if the score is non-negative
        
        return is_inlier

    def decision_function(self, X, most_anom_score=False):
        """
        Average anomaly score of X of the base classifiers.

        The anomaly score of an input sample is computed as
        the mean anomaly score of the trees in the forest.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        adjusted_scores : dict
            The adjusted anomaly scores of the input samples.
        """

        # Step 1: Compute the raw scores
        raw_scores = self.score_samples(X)
        printx("raw_scores", raw_scores)
        
        if most_anom_score:
            # Return the lowest (most anomalous) score for each sample
            adjusted_scores = {sample_id: min(score) for sample_id, score in raw_scores.items()}
        else:
            # Extract all scores into a list for calculating the offset
            all_scores = list(raw_scores.values())
            printx("all_scores", all_scores)
            
            # Step 2: Calculate the offset (e.g., using the median)
            self.offset_ = np.median(all_scores)
            printx("self.offset_", self.offset_)
            
            # Step 3: Adjust each sample's score by subtracting the offset
            adjusted_scores = {sample_id: float(score - self.offset_) for sample_id, score in raw_scores.items()}
        
        printx("adjusted_scores", adjusted_scores)
        return adjusted_scores
    
    def score_samples(self, X):
        """
        Opposite of the anomaly score defined in the original paper.

        The anomaly score of an input sample is computed as
        the mean anomaly score of the trees in the forest.

        The measure of normality of an observation given a tree is the depth
        of the leaf containing this observation, which is equivalent to
        the number of splittings required to isolate this point. In case of
        several observations n_left in the leaf, the average path length of
        a n_left samples isolation tree is added.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            The anomaly score of the input samples.
            The lower, the more abnormal.

        Notes
        -----
        The score function method can be parallelized by setting a joblib context. This
        inherently does NOT use the ``n_jobs`` parameter initialized in the class,
        which is used during ``fit``. This is because, calculating the score may
        actually be faster without parallelization for a small number of samples,
        such as for 1000 samples or less.
        The user can set the number of jobs in the joblib context to control the
        number of parallel jobs.

        .. code-block:: python

            from joblib import parallel_backend

            # Note, we use threading here as the score_samples method is not CPU bound.
            with parallel_backend("threading", n_jobs=4):
                model.score(X)
        """
        # Check data
        X = self._validate_data(
            X,
            accept_sparse="csr",
            dtype=tree_dtype,
            reset=False,
            #ensure_all_finite=False,
        )
        printx("self._score_samples(X)", self._score_samples(X))
        return self._score_samples(X)

    def _score_samples(self, X):
        """Private version of score_samples without input validation.

        Input validation would remove feature names, so we disable it.
        """
        # Code structure from ForestClassifier/predict_proba

        check_is_fitted(self)

        # Take the opposite of the scores as bigger is better (here less abnormal)
        chunk_scores = self._compute_chunked_score_samples(X)
        printx("Chunk scores", chunk_scores)

        def swap_signs(scores_dict):
            printx('scores_dict',scores_dict)
            return defaultdict(float, {k: -v for k, v in scores_dict.items()})

        # Swap the signs
        chunk_scores_swapped = swap_signs(chunk_scores)
        printx("Chunk scores swapped", chunk_scores_swapped)
        
        return chunk_scores_swapped

    def _compute_chunked_score_samples(self, X):
        # todo: to implement chunks, we need to maintain information on how many samples each ip leaf represents

        return self._compute_score_samples(X, False, min_score=self.min_score)

    def _compute_score_samples(self, X, subsample_features, min_score=False):
        """
        Compute the score of each samples in X going through the extra trees.

        Parameters
        ----------
        X : array-like or sparse matrix
            Data matrix.

        subsample_features : bool
            Whether features should be subsampled.

        min_score : bool, optional, default=False
            If True, return the minimum score of the unique IPs leaves.
            If False, return the average score.

        Returns
        -------
        scores : defaultdict of shape (n_samples,)
            The score of each sample in X.
        """
        min_score = self.min_score
        n_samples = X.shape[0]
        printx('n_samples', n_samples)


        depths = np.zeros(n_samples, order="f")

        average_path_length_max_samples = _average_path_length([self._max_samples])

        ipv4_depths = np.empty((n_samples,), dtype=object, order="f")
        for i in range(n_samples):
            ipv4_depths = []

        n_jobs, _, _ = _partition_estimators(self.n_estimators, None)
        lock = threading.Lock()
        Parallel(
            n_jobs=n_jobs,
            verbose=self.verbose,
            require="sharedmem",
        )(
            delayed(_parallel_compute_tree_depths)(
                tree,
                X,
                features if subsample_features else None,
                self._decision_path_lengths[tree_idx],
                self._average_path_length_per_tree[tree_idx],
                depths,
                ipv4_depths,
                lock,
                self.df,
            )
            for tree_idx, (tree, features) in enumerate(
                zip(self.estimators_, self.estimators_features_)
            )
        )

        # Initialize dictionaries to accumulate total depths and counts
        combined_depths = defaultdict(lambda: np.full(len(ipv4_depths), np.inf if min_score else 0))
        counts = defaultdict(lambda: np.zeros(len(ipv4_depths)))

        # Aggregate depths and counts for each sample ID for each tree
        for tree_idx, tree_depths in enumerate(ipv4_depths):
            for sample_id, depth in tree_depths:
                if min_score:
                    combined_depths[sample_id][tree_idx] = min(combined_depths[sample_id][tree_idx], depth)
                else:
                    combined_depths[sample_id][tree_idx] += depth
                counts[sample_id][tree_idx] += 1

        # Calculate scores
        scores = defaultdict(float)
        num_tree_leaves = {tree_idx: sum([counts[sample_id][tree_idx] for sample_id in combined_depths]) for tree_idx in range(len(ipv4_depths))}
        printx('num_tree_leaves', num_tree_leaves)
        for sample_id in combined_depths:
            sample_scores = []
            for tree_idx in range(len(ipv4_depths)):
                if counts[sample_id][tree_idx] > 0:
                    if min_score:
                        depth = combined_depths[sample_id][tree_idx]
                    else:
                        depth = combined_depths[sample_id][tree_idx] / counts[sample_id][tree_idx]
                else:
                    depth = 0

                denominator = _average_path_length([num_tree_leaves[tree_idx]])

                if denominator != 0:
                    sample_score = 2 ** (-depth / denominator)
                    sample_scores.append(sample_score)

            # Average or take the minimum of the scores from all trees
            if sample_scores:
                if min_score:
                    scores[sample_id] = np.min(sample_scores)
                else:
                    scores[sample_id] = np.mean(sample_scores)

        printx("_compute_score_samples Scores:", scores)

        return scores

    def _compute_score_samples_old(self, X, subsample_features):
        """
        Compute the score of each samples in X going through the extra trees.

        Parameters
        ----------
        X : array-like or sparse matrix
            Data matrix.

        subsample_features : bool
            Whether features should be subsampled.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            The score of each sample in X.
        """
        n_samples = X.shape[0]

        depths = np.zeros(n_samples, order="f")

        average_path_length_max_samples = _average_path_length([self._max_samples])

        ipv4_depths = np.empty((n_samples,), dtype=object, order="f")
        for i in range(n_samples):
            ipv4_depths = []

        n_jobs, _, _ = _partition_estimators(self.n_estimators, None)
        lock = threading.Lock()
        Parallel(
            n_jobs=n_jobs,
            verbose=self.verbose,
            require="sharedmem",
        )(
            delayed(_parallel_compute_tree_depths)(
                tree,
                X,
                features if subsample_features else None,
                self._decision_path_lengths[tree_idx],
                self._average_path_length_per_tree[tree_idx],
                depths,
                ipv4_depths,
                lock,
                self.df,
            )
            for tree_idx, (tree, features) in enumerate(
                zip(self.estimators_, self.estimators_features_)
            )
        )

        #printx("num_samples_per_leaf: ", num_samples_per_leaf)

        #printx('ipv4_avg_path_length_per_tree: ',ipv4_avg_path_length_per_tree)

        '''ipv4_depths = num_samples_per_leaf + ipv4_avg_path_length_per_tree - 1.0 
        printx('ipv4_depths',ipv4_depths)'''
        
        '''ipv4_depths = []
        for tree_samples, tree_path_lengths in zip(num_samples_per_leaf, ipv4_avg_path_length_per_tree):
            # Element-wise addition between corresponding entries
            tree_depths = np.array(tree_samples) + np.array(tree_path_lengths) - 1.0
            ipv4_depths.append(tree_depths)
    
        printx('ipv4_depths:', ipv4_depths)'''
        printx('ipv4_depths:', ipv4_depths)

        '''denominator = len(self.estimators_) * average_path_length_max_samples
        scores = 2 ** (
            # For a single training sample, denominator and depth are 0.
            # Therefore, we set the score manually to 1.
            -np.divide(
                ipv4_depths, denominator, out=np.ones_like(ipv4_depths), where=denominator != 0
            )
        )
        printx("scores: ", scores)'''

        # Initialize dictionaries to accumulate total depths and counts
        total_depths = defaultdict(lambda: np.zeros(len(ipv4_depths)))
        counts = defaultdict(lambda: np.zeros(len(ipv4_depths)))

        # Aggregate depths and counts for each sample ID for each tree
        for tree_idx, tree_depths in enumerate(ipv4_depths):
            for sample_id, depth in tree_depths:
                total_depths[sample_id][tree_idx] += depth
                counts[sample_id][tree_idx] += 1
        printx('counts',dict(counts))
        # Calculate scores
        scores = defaultdict(float)
        num_tree_leaves = {tree_idx:sum([counts[sample_id][tree_idx] for sample_id in total_depths]) for tree_idx in range(len(ipv4_depths))}
        printx("num_tree_leaves", num_tree_leaves)
        for sample_id in total_depths:
            sample_scores = []
            for tree_idx in range(len(ipv4_depths)):
                # Average depth for the sample ID in the current tree
                printx("counts[sample_id][tree_idx]", counts[sample_id][tree_idx])
                if counts[sample_id][tree_idx] > 0:
                    avg_depth = total_depths[sample_id][tree_idx] / counts[sample_id][tree_idx]
                else:
                    avg_depth = 0
                
                # Calculate the denominator (normalization factor) for the current tree
                #denominator = len(self.estimators_) * average_path_length_max_samples
                #average_path_length_max_samples = _average_path_length([self._max_samples])
                denominator = _average_path_length([num_tree_leaves[tree_idx]])
                printx("denominator", denominator)
                
                # Avoid division by zero
                if denominator == 0:
                    continue
                else:
                    sample_score = 2 ** (-avg_depth / denominator)
                
                sample_scores.append(sample_score)
            
            # Average the scores from all trees
            printx("sample_scores", sample_scores)
            scores[sample_id] = np.mean(sample_scores)

        printx("_compute_score_samples Scores:", scores)

        return scores

    def _more_tags(self):
        return {
            "_xfail_checks": {
                "check_sample_weights_invariance": (
                    "zero sample_weight is not equivalent to removing samples"
                ),
            },
            "allow_nan": True,
        }