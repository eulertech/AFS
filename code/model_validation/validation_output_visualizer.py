
""" Module imports """

import itertools
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import numpy as np
import pandas as pd

""" Attribute imports """

from sklearn.metrics import auc, confusion_matrix, f1_score, roc_curve, roc_auc_score

""" User-defined attribute imports """

from base_validation_output import BaseValidationOutput
from cross_validation_output import CrossValidationOutput
from holdout_validation_output import HoldoutValidationOutput
from timing import timing

# Create logger with module name
logger = logging.getLogger(__name__)

class ValidationOutputVisualizer(object):

    """
    Helper for visual displays of model output
    """

    default_population_threshold = 0.25

    def __init__(self, validation_output):

        self._validation_output = validation_output

    @property
    def validation_output(self):
        return self._validation_output

    def roc_add_to_plot(self, series_label):
        """
        Plot roc curve for validation output
        """

        if type(self.validation_output).__name__ == 'HoldoutValidationOutput':

            fpr, tpr, thresholds = roc_curve(self.validation_output.true, self.validation_output.pred, pos_label=1)

            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=1, label=series_label+' (area = %0.2f)' % (roc_auc))

        elif type(self.validation_output).__name__ == 'CrossValidationOutput':

            mean_tpr = 0.0
            mean_fpr = np.linspace(0, 1, 100)
            allProbs = np.zeros_like(self.validation_output.true, dtype=float)

            for i in self.validation_output.fold_range:
                fold = self.validation_output.get_fold(i)

                """ Compute ROC curve and area the curve """

                fpr, tpr, thresholds = roc_curve(fold.true, fold.pred, pos_label=1)
                mean_tpr += np.interp(mean_fpr, fpr, tpr)
                mean_tpr[0] = 0.0

                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

            mean_tpr /= self.validation_output.num_folds
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            plt.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    def roc(self, title='No title', path=None):

        """
        Plot roc curve for validation output
        """

        plt.clf()

        if type(self.validation_output).__name__ == 'HoldoutValidationOutput':

            fpr, tpr, thresholds = roc_curve(self.validation_output.true, self.validation_output.pred, pos_label=1)

            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)' % (roc_auc))

        elif type(self.validation_output).__name__ == 'CrossValidationOutput':

            mean_tpr = 0.0
            mean_fpr = np.linspace(0, 1, 100)
            allProbs = np.zeros_like(self.validation_output.true, dtype=float)

            for i in self.validation_output.fold_range:
                fold = self.validation_output.get_fold(i)

                """ Compute ROC curve and area the curve """

                fpr, tpr, thresholds = roc_curve(fold.true, fold.pred, pos_label=1)
                mean_tpr += np.interp(mean_fpr, fpr, tpr)
                mean_tpr[0] = 0.0

                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

            mean_tpr /= self.validation_output.num_folds
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            plt.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC: ' + title)
        plt.legend(loc="lower right")
        if path is not None:
            plt.savefig(path, bbox_inches = 'tight')
            plt.close()
        else:
            plt.show()

    def confusion_matrix(self, threshold=None):

        """
        Return confusion matrix based on provided threshold
        """

        threshold = threshold if threshold is not None else ValidationOutputVisualizer.default_population_threshold

        return confusion_matrix(y_true=self.validation_output.true, y_pred=self.validation_output.quantize_pred(threshold))

    def plot_confusion_matrix(self, threshold=None, title='Confusion Matrix', path=None):

        """
        Plot confusion matrix based on provided threshold
        """

        matrix = self.confusion_matrix(threshold)
        
        plt.clf()

        plt.imshow(matrix, interpolation='nearest'
            , cmap=plt.cm.Blues
            , norm=clrs.LogNorm(vmin=1
            , vmax=np.max(matrix)
            , clip=True))# cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, [0, 1])
        plt.yticks(tick_marks, [0, 1])

        color_threshold = np.log(matrix.max()) / 2.
        for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
            plt.text(j, i, matrix[i, j], fontsize=16,
                 horizontalalignment="center",
                 color="white" if matrix[i, j] > np.exp(color_threshold) else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        if path is not None:
            plt.savefig(path, bbox_inches = 'tight')
            plt.close()
        else:
            plt.show()

    def f1_score(self, threshold=None):

        """
        Return F1 score
        """

        threshold = threshold if threshold is not None else ValidationOutputVisualizer.default_population_threshold

        return f1_score(y_true=self.validation_output.true, y_pred=self.validation_output.quantize_pred(threshold))

    def f1_score_max_threshold_backtracking(self, ratio=0.5,tolerance=0.001, guess=0.2):
        assert ratio > 0 and ratio < 1.0
        debug_lvl = 4
        lim = 5
        top = 1.0
        best_f1 = -np.inf
        best_guess = np.nan
        done = False
        x = 1.0
        while not done:
            x = x * ratio
            multiplier = 1 - x
            guess = guess * multiplier
            f1 = self.f1_score(guess)
            done = (guess / multiplier - guess) < tolerance
            #debug_pr(str(best_guess) + ': ' + str(best_f1),4)
            #debug_pr(str(guess) + ': ' + str(f1),4)
            k=0
            while k < lim and  guess > tolerance:
                if(f1 >= best_f1):
                    best_f1 = f1
                    best_guess = guess
                guess = guess * multiplier
                f1 = self.f1_score(guess)
                #debug_pr(str(guess) + ': ' + str(f1),4)
                k = k + 1
            guess = best_guess / multiplier
            #debug_pr('___',4)
        return best_guess

    def f1_score_max_threshold(self, tolerance=0.001, guess=0.5):

        """
        Return threshold that generates the highest F1 score
        """
        assert tolerance > 0

        best_threshold = guess*1.0
        step_size = best_threshold * 2
        steps = 10
        best_f1 = -np.inf

        while (step_size >= tolerance):
            step_size = step_size / steps
            new_best_threshold = -1
            for k in range(-steps, steps + 1):
                thresh = best_threshold + k * step_size
                if(thresh < 0 or thresh > 1):
                    continue
                f1 = self.f1_score(thresh)
                if f1 > best_f1:
                    best_f1 = f1
                    new_best_threshold = thresh
            if new_best_threshold >= 0:
                best_threshold = new_best_threshold
        return best_threshold

    def auc_score(self):

            """
            Return AUC of ROC curve
            """
            score = roc_auc_score(
                y_true=self.validation_output.true,
                y_score=self.validation_output.pred
                )
            return score

    def cumulative_response_curve(self, population_threshold=None
                                      , title='target'
                                      , path=None):

        """
        Plot cumulative response curve of model output
        """

        population_threshold = population_threshold if population_threshold is not None else ValidationOutputVisualizer.default_population_threshold

        plt.clf()
        plt.plot([0, self.validation_output.perfect_target, 1], [0, 1, 1], 'b-', color='0.75')
        
        plt.plot([0, 1], [0, 1], '--', color='0.75')
        plt.plot(self.validation_output.cumulative_population, self.validation_output.cumulative_target, 'g-')

        plt.legend(['Perfect', 'Luck', 'Model'], loc=4)
        plt.xlabel('Proportion of Total Population')
        plt.ylabel('Proportion of Target')
        plt.title('Cume. Resp. Curve: {0}'.format(title))
        
        print 'Proportion of target captured at {0} population: {1}'.format(population_threshold, self.validation_output.captured_target(population_threshold))

        if path is not None:
            plt.savefig(path, bbox_inches = 'tight')
            plt.close()
        else:
            plt.show()

    def plot_feature_importance(self, title='target', path=None):

        """
        Plots logistic regression feature importance
        """

        if type(self.validation_output).__name__ == 'CrossValidationOutput':

            summary = self.validation_output.feature_importance_summary

            summary.sort_values(CrossValidationOutput.coef_abs, axis=0, ascending=False, inplace=True)
            summary.drop(CrossValidationOutput.coef_abs, axis=1, inplace=True)
            summary.reset_index(drop=True, inplace=True)
            summary.iloc[:10, :].plot(kind='bar', x=BaseValidationOutput.feature_name_col, y=CrossValidationOutput.coef_mean, yerr=CrossValidationOutput.coef_std)

        elif type(self.validation_output).__name__ == 'HoldoutValidationOutput':

            summary = self.validation_output.feature_importance_summary

            summary.sort_values(HoldoutValidationOutput.weight_col, axis=0, ascending=False, inplace=True)
            summary.iloc[:10, :].plot(kind='bar', x=BaseValidationOutput.feature_name_col, y=HoldoutValidationOutput.weight_col)

        plt.title('Top 10 features: {0}'.format(title))
        if path is not None:
            plt.savefig(path, bbox_inches = 'tight')
            plt.close()
        else:
            plt.show()
        
        return summary

def roc_setup(title='No Title'):
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC: ' + title)

def plt_save(filename):
    if filename is not None:
        plt.legend(loc="lower right")
        plt.savefig(filename, bbox_inches = 'tight')
        plt.close()
