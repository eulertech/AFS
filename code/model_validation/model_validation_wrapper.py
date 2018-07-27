
""" Module imports """

import logging
import numpy as np
import pandas as pd
import pickle
import re

from copy import deepcopy
from sklearn.externals import joblib

""" User-defined attribute imports """

from cross_validation_output import CrossValidationOutput
from holdout_validation_output import HoldoutValidationOutput
from timing import timing

# Create logger with module name
logger = logging.getLogger(__name__)

class ModelValidationWrapper(object):

    def __init__(self, classifier, trained=False):
        super(self.__class__, self).__init__()

        self._classifier = deepcopy(classifier)
        self._cross_validation_output = None
        self._holdout_validaiton_output = None
        self._trained = trained

    def train(self, X, y, n_folds=5, pos_label=1,
         roll_up_feature_importances=True):

        """
        Instantiate the wrapper's cross validation output and train the model on full feature data set.
        If run with n_folds=0 it skips the CV part.
        """

        # This section deals with non-numeric data types (one-hot encoding, casting)
        ohe_bool = False
        X_copy = deepcopy(X)

        if type(X) is pd.core.frame.DataFrame:
            # one-hot encode strings if there are 16 or fewer unique values
            # TODO: 16 should not be hard-coded
            ohe_cols_base = []
            for idx,dtyp in X.dtypes.iteritems():
                if str(dtyp) == 'object':
                    if X[idx].nunique() > 16:
                        X_copy[idx] = 1.0
                    else:
                        ohe_cols_base.append(idx)
                elif str(dtyp) == 'bool':
                    X_copy[idx] = X_copy[idx].astype(float)
                elif str(dtyp) == 'datetime64[ns]':
                    X_copy[idx] = X_copy[idx].dt.dayofyear.astype(float)

            ohe_bool = len(ohe_cols_base) != 0

            # TODO: # of digits to use in naming scheme shouldn't be hard-coded
            ohe_prefixes = ['OHE_{0:02d}_{1}'.format(i,col) for i, col in enumerate(ohe_cols_base)]
            X_ohe = pd.get_dummies(
                X_copy,
                columns=ohe_cols_base,
                prefix=ohe_prefixes,
                drop_first=True,
                )

            self.ohe_cols_base = ohe_cols_base
            self.all_ohe_cols = X_ohe.columns
            # self.cat_dict = dict()
            # for col in ohe_cols_base:
            #     self.cat_dict[col] = list(X[col].unique())


        
        if n_folds == 0:
            logger.debug('Skipping cross-validation')
            temp_cross_validation_output = \
                CrossValidationOutput.from_classifier_no_cv(
                                                        self.classifier,
                                                        X_ohe,
                                                        y,
                                                        pos_label)

        else:
            logger.debug('Running cross-validation')
            with timing(logger, 'Cross-validation complete'):

                temp_cross_validation_output =\
                    CrossValidationOutput.from_classifier(
                                                    self.classifier,
                                                    X_ohe,
                                                    y,
                                                    n_folds,
                                                    pos_label)

        if ohe_bool and roll_up_feature_importances:
            feature_importances_full = temp_cross_validation_output.feature_importances
            feature_importances = pd.DataFrame(
                                    feature_importances_full.mean()).transpose()
            f_i_cols = feature_importances.columns

            ohe_cols = []
            non_ohe_cols = []
            for col in f_i_cols:
                if re.match("^OHE_\d\d_",col) is not None:
                    ohe_cols.append(col)
                else:
                    non_ohe_cols.append(col)

            ohe_feat_imp = feature_importances[ohe_cols]\
                                    .transpose()\
                                    .rename(columns={0:'FEAT_IMP'})\
                                    .reset_index()

            top_ohe_index = ohe_feat_imp.groupby(
                ohe_feat_imp['index'].str.slice(0,6))['FEAT_IMP'].idxmax().values
            top_ohe_col_names = list(
                ohe_feat_imp.loc[top_ohe_index,:]['index'].values)

            ohe_decode = {}
            for col in ohe_prefixes:
                ohe_decode[col[:6]] = col[7:]

            full_cols_with_OHE = non_ohe_cols + top_ohe_col_names

            full_cols_no_OHE = non_ohe_cols \
                       + [ohe_decode[col[:6]] for col in ohe_prefixes]

            feature_importances = feature_importances_full[full_cols_with_OHE]
            feature_importances.columns = full_cols_no_OHE
            feature_importances = feature_importances.reindex(columns=X.columns)

            new_cross_val_output = CrossValidationOutput(
                temp_cross_validation_output.predictions,
                fi=feature_importances)
        else:
            new_cross_val_output = temp_cross_validation_output

        self.cross_validation_output = new_cross_val_output


        logger.debug('Fitting classifier on full training set')
        with timing(logger, 'Fitting complete'):
            self.classifier.fit(X_ohe, y)

        self._trained = True

    def test(self, X, y, pos_label=1):

        """
        Instantiate the wrapper's holdout validation output using trained model
        """
        X_copy = X

        if type(X) is pd.core.frame.DataFrame:
            ohe_cols_base = []
            make_dummy = []
            # one-hot encode strings if there are 16 or fewer unique values
            ohe_cols_base = []
            make_dummy = []
            for idx,dtyp in X.dtypes.iteritems():
                if str(dtyp) == 'object':
                    if (X[idx].nunique() > 16) or (idx in self.ohe_cols_base):
                        make_dummy.append(idx)
                    else:
                        ohe_cols_base.append(idx)
                elif str(dtyp) == 'bool':
                    X_copy[idx] = X_copy[idx].astype(float)
                elif str(dtyp) == 'datetime64[ns]':
                    X_copy[idx] = X_copy[idx].dt.dayofyear.astype(float)

            for dummy in make_dummy:
                X_copy[dummy] = 1.0

            ohe_prefixes = ['OHE_{0:02d}_{1}'.format(i,col) for i, col in enumerate(ohe_cols_base)]
            X_ohe = pd.get_dummies(
                X_copy,
                columns=ohe_cols_base,
                prefix=ohe_prefixes,
                drop_first=True,
                )

            for col in self.all_ohe_cols:
                if col not in X_ohe.columns:
                    X_ohe[col] = 0
            droplist = []
            for col in X_ohe.columns:
                if col not in self.all_ohe_cols:
                    droplist.append(col)
            X_ohe = X_ohe.drop(droplist,axis=1, inplace=False)

        if not self.trained:
            raise ValueError('Classifier not fit on train set')

        logger.debug('Predicting on holdout set')
        with timing(logger, 'Prediction complete'):

            self._holdout_validaiton_output = HoldoutValidationOutput.from_classifier(self.classifier, X_ohe, y, pos_label)

    @property
    def classifier(self):

        """
        Classifier used for all validations 
        """

        return self._classifier

    @classifier.setter
    def classifier(self,value):

        self._classifier = value

    @property
    def cross_validation_output(self):

        """
        Result of cross-validation run
        """

        if self._cross_validation_output is None:
            raise ValueError('Cross validation output not produced')

        return self._cross_validation_output

    @cross_validation_output.setter
    def cross_validation_output(self,val):
       self._cross_validation_output = val
       return


    @property
    def holdout_validation_output(self):

        """
        Result of holdout-validation run
        """

        if self._holdout_validaiton_output is None:
            raise ValueError('Holdout validation output not produced')

        return self._holdout_validaiton_output

    @property
    def trained(self):

        """
        Flag if classifier has been trained on train data
        """

        return self._trained

    def dump(self, path):

        """
        Convenience method to pickle ModelValidationWrapper
        """

        with open(path, 'wb+') as f:
            joblib.dump(self, f)

    @staticmethod
    def load(path):

        """
        Loads any pickle object from the path, here for convenience
        """

        with open(path, 'rb') as f:
            return joblib.load(f)