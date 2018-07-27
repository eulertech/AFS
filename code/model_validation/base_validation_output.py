
""" Module imports """

import numpy as np
import pandas as pd
import pickle

""" Attribute imports """

from abc import ABCMeta, abstractmethod, abstractproperty
from sklearn.pipeline import Pipeline

""" User-defined attribute imports """

from timing import timing

class BaseValidationOutput(object):
    __metaclass__ = ABCMeta

    """
    Wrapper for classifier output, containing predictions, truth values, statistics, and feature information
    """

    # Constants for column names

    true_col = 'y_true'
    pred_col = 'y_pred'

    output_columns = [true_col, pred_col]

    cumulative_target_col = 'cumulative_target'
    cumulative_population_col = 'cumulative_population'

    # Columns for feature summary output

    feature_name_col = 'feature_name'

    def __init__(self, predictions, fi=None):

        """
        Constructor: takes in properly formatted truth/prediction dataframe and optional feature importance dataframe
        """

        self._predictions = predictions if type(predictions) is pd.core.frame.DataFrame else pd.DataFrame(predictions)
        self._feature_importances = fi if type(fi) is pd.core.frame.DataFrame else pd.core.frame.DataFrame(fi)

        for col in [col for col in [BaseValidationOutput.true_col, BaseValidationOutput.pred_col] if col not in self._predictions.columns]:
            raise ValueError('No column \'{0}\''.format(col))

        if any([col not in self._predictions.columns for col in [BaseValidationOutput.cumulative_target, BaseValidationOutput.cumulative_population]]):
            self._calculate_cumulative_response()

    def _calculate_cumulative_response(self):

        """
        Set cumulative population and target by descending prediciton
        """

        self.predictions.sort_values(BaseValidationOutput.pred_col, ascending=False, inplace=True)
        self.predictions.loc[:, BaseValidationOutput.cumulative_target_col] = self.true.cumsum() / self.true.sum()
        self.predictions.loc[:, BaseValidationOutput.cumulative_population_col] = [float(i + 1) / self.num_predictions for i in range(self.num_predictions)] 

    @abstractmethod
    def from_classifier():

        """
        Create output class from classifier
        """

        return

    def quantize_pred(self, threshold=.25):

        """
        Convert prediction probabilities to 0 or 1 based on provided threshold
        """

        return [int(i) for i in self.pred >= self.model_score_at_threshold(threshold)]

    def model_score_at_threshold(self, population_threshold=.25):

        """
        Return target captured at population threshold
        """

        return self.predictions[self.cumulative_population >= population_threshold][BaseValidationOutput.pred_col].values[0]

    def captured_target(self, population_threshold=.25):

        """
        Return target captured at population threshold
        """

        return self.predictions[self.cumulative_population >= population_threshold][BaseValidationOutput.cumulative_target_col].values[0]

    def to_csv(self, predictions_path, feature_importances_path):

        """
        Save component dataframes to csvs
        """

        self.predictions.to_csv(predictions_path, index=False)
        self.feature_importances.to_csv(feature_importances_path, index=False)

    @property
    def predictions(self):

        """
        Return output dataframe
        """

        return self._predictions

    @property
    def feature_importances(self):

        """
        Return output dataframe
        """

        return self._feature_importances

    @feature_importances.setter
    def feature_importances(self,val):
        self._feature_importances = val
        return 
        
    @abstractproperty
    def feature_importance_summary(self):

        """
        Return summary of feature importances
        """

        return  

    @property
    def true(self):

        """
        Predictions frame true_col
        """

        return self.predictions[BaseValidationOutput.true_col]

    @property
    def pred(self):

        """
        Predictions frame pred_col
        """

        return self.predictions[BaseValidationOutput.pred_col]

    @property
    def cumulative_target(self):

        """
        Predictions frame cumulative_target
        """

        return self.predictions[BaseValidationOutput.cumulative_target_col]

    @property
    def cumulative_population(self):

        """
        Predictions frame cumulative_population
        """

        return self.predictions[BaseValidationOutput.cumulative_population_col]

    @property
    def num_predictions(self):

        """
        Count of prediction frame rows
        """

        return len(self.predictions.index)

    @property
    def num_features(self):

        """
        Number of features from classifier
        """

        return len(self.feature_importances.index)

    @property
    def perfect_target(self):

        """
        Proportion of population that is target
        """

        return float(self.true.sum()) / self.num_predictions

    def dump(self, path):

        """
        Convenience method to pickle BaseValidationOutput
        """

        with open(path, 'wb+') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):

        """
        Convenience method to load from pickle
        """

        with open(path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def get_feature_importances(model):

        """
        Helper method to extract model parameters from RF/LR pipelines/classifiers
        """

        # Get classifier if pipeline provided
        classifier = model if not isinstance(model, Pipeline) else model.get_params()[model.steps[-1][0]]

        # For random forest model
        if hasattr(classifier, 'feature_importances_'):
            return classifier.feature_importances_.squeeze()

        # For logistic regression model
        elif hasattr(classifier, 'coef_'):
            return classifier.coef_.squeeze()
        
        else:
            return None