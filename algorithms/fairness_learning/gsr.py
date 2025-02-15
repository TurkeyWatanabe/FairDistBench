"""
The code for GridSearchReduction wraps the source class
fairlearn.reductions.GridSearch
available in the https://github.com/fairlearn/fairlearn library
licensed under the MIT Licencse, Copyright Microsoft Corporation
"""
from logging import warning
import copy
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

try:
    import fairlearn.reductions as red
except ImportError as error:
    warning("{}: GridSearchReduction will be unavailable. To install, run:\n"
            "pip install 'aif360[Reductions]'".format(error))
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder

from aif360.algorithms import Transformer


class GridSearchReduction(Transformer):
    """Grid search reduction for fair classification or regression.

    Grid search is an in-processing technique that can be used for fair
    classification or fair regression. For classification it reduces fair
    classification to a sequence of cost-sensitive classification problems,
    returning the deterministic classifier with the lowest empirical error
    subject to fair classification constraints [#agarwal18]_ among the
    candidates searched. For regression it uses the same priniciple to return a
    deterministic regressor with the lowest empirical error subject to the
    constraint of bounded group loss [#agarwal19]_.

    References:
        .. [#agarwal18] `A. Agarwal, A. Beygelzimer, M. Dudik, J. Langford, and
           H. Wallach, "A Reductions Approach to Fair Classification,"
           International Conference on Machine Learning, 2018.
           <https://arxiv.org/abs/1803.02453>`_
        .. [#agarwal19] `A. Agarwal, M. Dudik, and Z. Wu, "Fair Regression:
           Quantitative Definitions and Reduction-based Algorithms,"
           International Conference on Machine Learning, 2019.
           <https://arxiv.org/abs/1905.12843>`_
    """
    def __init__(self,
                 estimator,
                 constraints,
                 constraint_weight=0.5,
                 grid_size=10,
                 grid_limit=2.0,
                 grid=None,
                 loss="ZeroOne",
                 min_val=None,
                 max_val=None):
        """
        Args:
            estimator: An estimator implementing methods ``fit(X, y,
                sample_weight)`` and ``predict(X)``, where ``X`` is the matrix
                of features, ``y`` is the vector of labels, and
                ``sample_weight`` is a vector of weights; labels ``y`` and
                predictions returned by ``predict(X)`` are either 0 or 1 -- e.g.
                scikit-learn classifiers/regressors.
            constraints (str or fairlearn.reductions.Moment): If string, keyword
                denoting the :class:`fairlearn.reductions.Moment` object
                defining the disparity constraints -- e.g., "DemographicParity"
                or "EqualizedOdds". For a full list of possible options see
                `self.model.moments`. Otherwise, provide the desired
                :class:`~fairlearn.reductions.Moment` object defining the
                disparity constraints.
            constraint_weight: When the ``selection_rule`` is
                "tradeoff_optimization" (default, no other option currently)
                this float specifies the relative weight put on the constraint
                violation when selecting the best model. The weight placed on
                the error rate will be ``1-constraint_weight``.
            grid_size (int): The number of Lagrange multipliers to generate in
                the grid.
            grid_limit (float): The largest Lagrange multiplier to generate. The
                grid will contain values distributed between ``-grid_limit`` and
                ``grid_limit`` by default.
            grid (pandas.DataFrame): Instead of supplying a size and limit for
                the grid, users may specify the exact set of Lagrange
                multipliers they desire using this argument in a DataFrame.
            loss (str): String identifying loss function for constraints.
                Options include "ZeroOne", "Square", and "Absolute."
            min_val: Loss function parameter for "Square" and "Absolute,"
                typically the minimum of the range of y values.
            max_val: Loss function parameter for "Square" and "Absolute,"
                typically the maximum of the range of y values.
        """
        super(GridSearchReduction, self).__init__()

        #init model, set prot_attr during fit

        self.model = skGridSearchRed(estimator, constraints,
                constraint_weight, grid_size, grid_limit, grid,
                loss, min_val, max_val)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resnet50 = models.resnet50(pretrained=True).to(self.device)
        self.resnet50 = torch.nn.Sequential(*list(self.resnet50.children())[:-1])
        self.resnet50.eval()


    def fit(self, dataset):
        """Learns model with less bias

        Args:
            dataset : Dataset containing true output.

        Returns:
            GridSearchReduction: Returns self.
        """


        data = dataset.data
        batch_size = 256
        data = torch.tensor(data).permute(0, 3, 1, 2).float()
        data = TensorDataset(data)
        data = DataLoader(data, batch_size=batch_size, shuffle=False)
        features_list = []
        with torch.no_grad():
            for batch in data:
                batch = batch[0].to(self.device) 
                features = self.resnet50(batch) 
                features_list.append(features.squeeze().cpu()) 
        X = torch.cat(features_list, dim=0).numpy()

        Y = dataset.labels
        A = dataset.sensitive_attribute

        self.model.fit(X, Y, A)

        return self


    def predict(self, dataset):
        """Obtain the predictions for the provided dataset using the model
        learned.

        Args:
            dataset: Dataset containing output values that need to be
                transformed.

        Returns:
            dataset: Transformed dataset.
        """
        data = dataset.data
        batch_size = 256
        data = torch.tensor(data).permute(0, 3, 1, 2).float()
        data = TensorDataset(data)
        data = DataLoader(data, batch_size=batch_size, shuffle=False)
        features_list = []
        with torch.no_grad():
            for batch in data:
                batch = batch[0].to(self.device) 
                features = self.resnet50(batch) 
                features_list.append(features.squeeze().cpu()) 
        X = torch.cat(features_list, dim=0).numpy()

        dataset_new = copy.deepcopy(dataset)
        dataset_new.labels = self.model.predict(X).reshape(-1, 1)

        # if isinstance(self.model.moment_, red.ClassificationMoment):
        #     fav = int(dataset.favorable_label)
        #     try:
        #         # Probability of favorable label
        #         scores = self.model.predict_proba(X)[:, fav]
        #         dataset_new.scores = scores.reshape(-1, 1)
        #     except (AttributeError, NotImplementedError):
        #         warning("dataset.scores not updated, underlying model does not "
        #                 "support predict_proba")

        return dataset_new
    

class skGridSearchRed(BaseEstimator, ClassifierMixin):
    """Grid search reduction for fair classification or regression.

    Grid search is an in-processing technique that can be used for fair
    classification or fair regression. For classification it reduces fair
    classification to a sequence of cost-sensitive classification problems,
    returning the deterministic classifier with the lowest empirical error
    subject to fair classification constraints [#agarwal18]_ among the
    candidates searched. For regression it uses the same priniciple to return a
    deterministic regressor with the lowest empirical error subject to the
    constraint of bounded group loss [#agarwal19]_.

    References:
        .. [#agarwal18] `A. Agarwal, A. Beygelzimer, M. Dudik, J. Langford, and
           H. Wallach, "A Reductions Approach to Fair Classification,"
           International Conference on Machine Learning, 2018.
           <https://arxiv.org/abs/1803.02453>`_
        .. [#agarwal19] `A. Agarwal, M. Dudik, and Z. Wu, "Fair Regression:
           Quantitative Definitions and Reduction-based Algorithms,"
           International Conference on Machine Learning, 2019.
           <https://arxiv.org/abs/1905.12843>`_
    """
    def __init__(self,
                estimator,
                constraints,
                constraint_weight=0.5,
                grid_size=10,
                grid_limit=2.0,
                grid=None,
                loss="ZeroOne",
                min_val=None,
                max_val=None
                ):
        """
        Args:
            estimator: An estimator implementing methods ``fit(X, y,
                sample_weight)`` and ``predict(X)``, where ``X`` is the matrix
                of features, ``y`` is the vector of labels, and
                ``sample_weight`` is a vector of weights; labels ``y`` and
                predictions returned by ``predict(X)`` are either 0 or 1 -- e.g.
                scikit-learn classifiers/regressors.
            constraints (str or fairlearn.reductions.Moment): If string, keyword
                denoting the :class:`fairlearn.reductions.Moment` object
                defining the disparity constraints -- e.g., "DemographicParity"
                or "EqualizedOdds". For a full list of possible options see
                `self.model.moments`. Otherwise, provide the desired
                :class:`~fairlearn.reductions.Moment` object defining the
                disparity constraints.
            constraint_weight: When the ``selection_rule`` is
                "tradeoff_optimization" (default, no other option currently)
                this float specifies the relative weight put on the constraint
                violation when selecting the best model. The weight placed on
                the error rate will be ``1-constraint_weight``.
            grid_size (int): The number of Lagrange multipliers to generate in
                the grid.
            grid_limit (float): The largest Lagrange multiplier to generate. The
                grid will contain values distributed between ``-grid_limit`` and
                ``grid_limit`` by default.
            grid (pandas.DataFrame): Instead of supplying a size and limit for
                the grid, users may specify the exact set of Lagrange
                multipliers they desire using this argument in a DataFrame.
            loss (str): String identifying loss function for constraints.
                Options include "ZeroOne", "Square", and "Absolute."
            min_val: Loss function parameter for "Square" and "Absolute,"
                typically the minimum of the range of y values.
            max_val: Loss function parameter for "Square" and "Absolute,"
                typically the maximum of the range of y values.
        """
        self.estimator = estimator
        self.constraints = constraints
        self.constraint_weight = constraint_weight
        self.grid_size = grid_size
        self.grid_limit = grid_limit
        self.grid = grid
        self.loss = loss
        self.min_val = min_val
        self.max_val = max_val

    def fit(self, X, y, A):
        """Train a less biased classifier or regressor with the given training
        data.

        Args:
            X (pandas.DataFrame): Training samples.
            y (array-like): Training output.

        Returns:
            self
        """
        self.estimator_ = clone(self.estimator)

        moments = {
            "DemographicParity": red.DemographicParity,
            "EqualizedOdds": red.EqualizedOdds,
            "TruePositiveRateParity": red.TruePositiveRateParity,
            "FalsePositiveRateParity": red.FalsePositiveRateParity,
            "ErrorRateParity": red.ErrorRateParity,
            "BoundedGroupLoss": red.BoundedGroupLoss,
        }
        if isinstance(self.constraints, str):
            if self.constraints not in moments:
                raise ValueError(f"Constraint not recognized: {self.constraints}")
            if self.constraints == "BoundedGroupLoss":
                losses = {
                    "ZeroOne": red.ZeroOneLoss,
                    "Square": red.SquareLoss,
                    "Absolute": red.AbsoluteLoss
                }
                if self.loss == "ZeroOne":
                    self.loss_ = losses[self.loss]()
                else:
                    self.loss_ = losses[self.loss](self.min_val, self.max_val)

                self.moment_ = moments[self.constraints](loss=self.loss_)
            else:
                self.moment_ = moments[self.constraints]()
        elif isinstance(self.constraints, red.Moment):
            self.moment_ = self.constraints
        else:
            raise ValueError("constraints must be a string or Moment object.")

        self.model_ = red.GridSearch(estimator=self.estimator_,
                constraints=self.moment_,
                constraint_weight=self.constraint_weight,
                grid_size=self.grid_size, grid_limit=self.grid_limit,
                grid=self.grid)

        if isinstance(self.model_.constraints, red.ClassificationMoment):
            le = LabelEncoder()
            y = le.fit_transform(y)
            self.classes_ = le.classes_

        self.model_.fit(X, y, sensitive_features=A)

        return self

    def predict(self, X):
        """Predict output for the given samples.

        Args:
            X (pandas.DataFrame): Test samples.

        Returns:
            numpy.ndarray: Predicted output per sample.
        """

        return self.model_.predict(X)


    def predict_proba(self, X):
        """Probability estimates.

        The returned estimates for all classes are ordered by the label of
        classes for classification.

        Args:
            X (pandas.DataFrame): Test samples.

        Returns:
            numpy.ndarray: returns the probability of the sample for each class
            in the model, where classes are ordered as they are in
            ``self.classes_``.
        """

        if isinstance(self.model_.constraints, red.ClassificationMoment):
            return self.model_.predict_proba(X)

        raise NotImplementedError("Underlying model does not support "
                                  "predict_proba")