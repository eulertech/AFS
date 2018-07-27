
import logging
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

from base_validation_output import BaseValidationOutput
from timing import timing

# Create logger with module name
logger = logging.getLogger(__name__)

class CrossValidationOutput(BaseValidationOutput):

    def __init__(self, *args, **kwargs):

        super(self.__class__, self).__init__(*args, **kwargs)

    fold_col = 'y_fold'

    output_columns = [BaseValidationOutput.true_col, BaseValidationOutput.pred_col, fold_col]

    feature_name = 'feature_name'
    coef_mean = 'coef_mean'
    coef_std = 'coef_std'
    coef_rsd = 'coef_rsd'
    coef_abs = 'coef_abs'

    @classmethod
    def from_classifier(cls, classifier, X, y, n_folds=5, pos_label=1):

        """
        Create BaseValidationOutput object from cross validation of classifier
        """

        X = X if type(X) is pd.core.frame.DataFrame else pd.DataFrame(X)
        y = y if type(y) is pd.core.series.Series else pd.Series(y)

        df = pd.DataFrame(columns=cls.output_columns)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=False)

        feature_importances = []

        for i, (train, test) in enumerate(skf.split(X, y)):

            fold_label = i + 1
            logger.debug('Fitting fold %i' % fold_label)
            with timing(logger, 'Fold %i fit' % fold_label):

                classifier.fit(X.iloc[train], y.iloc[train])

                true_col = [int(x) for x in pd.Series(y.iloc[test].tolist()) == pos_label]
                pred_col = classifier.predict_proba(X.iloc[test])[:, pos_label].tolist()
                fold_col = [i] * len(test)

                data = np.matrix([true_col, pred_col, fold_col])
                df = df.append(pd.DataFrame(data.T, columns=cls.output_columns), ignore_index=True)

                feature_importances.append(BaseValidationOutput.get_feature_importances(classifier))

        df[cls.true_col] = df[cls.true_col].astype(int)
        df[cls.fold_col] = df[cls.fold_col].astype(int)

        feature_importances = pd.DataFrame(feature_importances, columns=X.columns)

        return cls(df, feature_importances)

    @classmethod
    def from_classifier_no_cv(cls, classifier, X, y, pos_label=1):

        """
        Create BaseValidationOutput object from cross validation of classifier
        """

        X = X if type(X) is pd.core.frame.DataFrame else pd.DataFrame(X)
        y = y if type(y) is pd.core.series.Series else pd.Series(y)

        df = pd.DataFrame(columns=cls.output_columns)

        feature_importances = []

        logger.debug('Fitting model')
        with timing(logger, 'model fit'):

            classifier.fit(X, y)

            true_col = [int(x) for x in pd.Series(y.tolist()) == pos_label]
            pred_col = classifier.predict_proba(X)[:, pos_label].tolist()
            fold_col = [0] * len(true_col)

            data = np.matrix([true_col, pred_col, fold_col])
            df = df.append(pd.DataFrame(data.T, columns=cls.output_columns), ignore_index=True)

            feature_importances.append(BaseValidationOutput.get_feature_importances(classifier))

        df[cls.true_col] = df[cls.true_col].astype(int)
        df[cls.fold_col] = df[cls.fold_col].astype(int)

        feature_importances = pd.DataFrame(feature_importances, columns=X.columns)

        return cls(df, feature_importances)

    @property
    def feature_importance_summary(self):

        """
        Return summary of feature importances
        """
        
        param_means = np.mean(self.feature_importances, axis=0)
        param_stds = np.std(self.feature_importances, axis=0)
        
        data = {}
        data[BaseValidationOutput.feature_name_col] = self.feature_importances.columns
        data[CrossValidationOutput.coef_mean] = param_means
        data[CrossValidationOutput.coef_std] = param_stds
        summary = pd.DataFrame(data)
        
        summary[CrossValidationOutput.coef_rsd] = summary[CrossValidationOutput.coef_std] / summary[CrossValidationOutput.coef_mean].abs()
        summary[CrossValidationOutput.coef_abs] = summary[CrossValidationOutput.coef_mean].abs()

        return summary

    @property
    def folded(self):

        """
        Whether or not output frame contains fold distinction
        """

        return CrossValidationOutput.fold_col in self._predictions.columns and self.num_folds > 1

    @property
    def num_folds(self):

        """
        Number of folds
        """

        return len(self.fold.unique())

    @property
    def fold_range(self):

        """
        List of folds
        """

        return range(self.num_folds)

    def get_fold(self, n):

        """
        Return specified fold frame
        """

        return CrossValidationOutput(self.predictions[self.predictions[CrossValidationOutput.fold_col] == n])   

    @property
    def fold(self):

        """
        Predictions frame fold_col
        """

        return self.predictions[CrossValidationOutput.fold_col]

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