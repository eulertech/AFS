
import logging
import numpy as np
import pandas as pd

from base_validation_output import BaseValidationOutput

# Create logger with module name
logger = logging.getLogger(__name__)

class HoldoutValidationOutput(BaseValidationOutput):

    """
    Class for prediction output when running trained classifier against new data set
    """

    weight_col = 'weight'

    summary_cols = [BaseValidationOutput.feature_name_col, weight_col]

    def __init__(self, *args, **kwargs):

        """
        Constructor for HoldoutValdiationOutput
        """

        super(self.__class__, self).__init__(*args, **kwargs)

    @classmethod
    def from_classifier(cls, classifier, X, y, pos_label=1):

        """
        Use trained classifier on test feature data and compare against test target
        """

        X = X if type(X) is pd.core.frame.DataFrame else pd.DataFrame(X)
        y = y if type(y) is pd.core.series.Series else pd.Series(y)

        true_col = [int(yi) for yi in y == pos_label]
        pred_col = classifier.predict_proba(X)[:, pos_label].tolist()

        data = np.matrix([true_col, pred_col])
        predictions = pd.DataFrame(data.T, columns=cls.output_columns)

        predictions[cls.true_col] = predictions[cls.true_col].astype(int)

        feature_importances = BaseValidationOutput.get_feature_importances(classifier)
        feature_importances = pd.DataFrame([feature_importances], columns=X.columns)

        return cls(predictions, feature_importances)

    @property
    def feature_importance_summary(self):
        
        """
        Return summary of feature importances
        """

        summary = pd.DataFrame(columns=HoldoutValidationOutput.summary_cols)
        summary.loc[:, BaseValidationOutput.feature_name_col] = self.feature_importances.columns
        summary.loc[:, HoldoutValidationOutput.weight_col] = self.feature_importances.T.iloc[:, 0]

        return summary