import numpy as np
import pandas as pd
import numbers
import sys

from global_definitions import AFS_HOME, DATA_DIR, GLOBAL_DEBUG_LEVEL, my_timer, debug_pr, tic, toc
from customTools import compute_correlation, get_feature_correlation, equals
from model_validation.model_validation_wrapper import ModelValidationWrapper
from util import utils



def evaluateCorrelation(population, afs_tool, comparison_set='bfs'):
    """
    Method for evaluation population fitness using correlation with the target variable
    
    @param population: population to evaluate
    @param afs_tool: AFSTool object (holds the data for the given population)
    @param comparison_set: collection of individuals to compare for correlation penalty calculations
        None: no comparison, compute a univariate correlation
        'bfs': compare to current Best Feature Set
        'hof': compare to the current Hall of Fame

    Return number of .apply operations needed (params population and afs_tool are modified)
    
    """
    evals_before = afs_tool.eval_counter
    debug_pr("Starting population evaluate.",3)
    if comparison_set is None:
        fitnesses = map(lambda x: evaluateIndividualCorrelation(x, afs_tool), population)
    else:
        fitnesses = map(lambda x: evaluateIndividualDiversityScore(x, afs_tool, comparison_set), population)
    for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
    evals_after = afs_tool.eval_counter

    return evals_after - evals_before

def evaluateIndividualCorrelation(individual, afs_tool): # evaluation function for DEAP
    debug_pr("Getting feature {}...".format(utils.string_replace_using_dict(str(individual), afs_tool.arg_dict)),4)
    col = afs_tool.get_feature_col(individual)
    toc(debug_level=4)
    debug_pr("Computing correlation...".format(str(individual)),4)
    if(type(col) is np.ndarray):
        type_ = type(col[0])
    else:
        type_ = type(col.values[0])
    if issubclass(type_, np.number):
        corr_coef = compute_correlation(col, afs_tool.target)
    elif issubclass(type_, bool) or issubclass(type_, np.bool_):
        col = col.astype(float)
        corr_coef = compute_correlation(col, afs_tool.target)
    else: # issubclass(type, str):
        if (col.nunique() > 16): # TODO: parameterize the maximum number of columns
            return (0.,)

        ohe_df = pd.get_dummies(col)

        # Taking the maximum correlation coefficient among OHE columns
        corr_coef = -1
        for col_key in ohe_df:
            temp = compute_correlation(ohe_df[col_key], afs_tool.target)
            if (temp > corr_coef):
                corr_coef = temp
    return (corr_coef,)
    toc(debug_level=4)
def evaluateIndividualDiversityScore(individual, afs_tool, comparison_set_str = 'bfs'): # evaluation function for DEAP
    '''
    Finds a "diversity score" based on the difference
     (correlation with target) - (correlation with most similar HOF feature)
    '''

    if type(comparison_set_str) is str:
        if comparison_set_str == 'bfs':
            comparison_set = afs_tool.bfs
        elif comparison_set_str == 'hof':
            comparison_set = afs_tool.hof
        else:
            raise ValueError('comparison_set_str value must be \'bfs\' or \'hof\'.')
    else:
        raise TypeError('comparison_set_str must be a string')
    comparison_set = list(comparison_set)
    debug_pr("Getting feature {}...".format(str(individual)),4)
    
    # tic()

    corr_coef = get_feature_correlation(individual, afs_tool.target_name, afs_tool)

    # if individual in comparison_set or len(comparison_set)<1:
    if len(comparison_set) < 1:
        score = corr_coef
    else:
        most_similar_individual = None
        most_similar_comparison_indexx = 0
        highest_corr = -1
        for comparison_index in range(len(comparison_set)):
            comparison_indiv = comparison_set[comparison_index]
            if (equals(individual, comparison_indiv)):
                continue
            corr = get_feature_correlation(individual, comparison_indiv, afs_tool)
            if corr > highest_corr:
                highest_corr = corr
                most_similar_individual = comparison_indiv
                most_similar_comparison_indexx = comparison_index
        if most_similar_individual is None:
            score = corr_coef
        else:
            similar_feature_target_corr = get_feature_correlation(most_similar_individual, afs_tool.target_name, afs_tool)
            if (comparison_set_str is 'hof'):
                if corr_coef > similar_feature_target_corr and not (individual in afs_tool.hof):
                    afs_tool.hof.remove(most_similar_comparison_indexx)
                    afs_tool.hof.insert(individual)
                    most_similar_individual.fitness.values = ( (similar_feature_target_corr - highest_corr), )
                    score = corr_coef
                else:
                    score = corr_coef - highest_corr
            elif corr_coef > similar_feature_target_corr:
                    score = corr_coef
            else:
                score = corr_coef - highest_corr
    # print(score)
    # toc()
    return (score,)

def evaluateModelWrapper(population, afs_tool, classifier):
    """
    Method for evaluation population fitness using feature importance from a model
    
    Parameters
    ----------
    population: population to evaluate
    afs_tool: AFSTool object (holds the data for the given population)
    classifier: sklearn classifier (i.e. random forest)

    Returns
    -------
    Return number of .apply operations needed (params population and afs_tool are modified)
    
    """

    evals_before = afs_tool.eval_counter

    wrapper = ModelValidationWrapper(classifier)
    data_frame = afs_tool.get_feature_set_dataframe(population)
    wrapper.train(
        data_frame,
        afs_tool.target,
         pos_label=1
        )

    fitnesses = wrapper.cross_validation_output.\
                    feature_importance_summary.coef_abs.values.tolist()

    fitness_dict = dict()
    for col, fit in zip(data_frame,fitnesses):
        fitness_dict[col] = fit

    for ind in population:
        key = str(ind)
        if(key in fitness_dict):
            fit_tuple = (fitness_dict[key],)
        else:
            fit_tuple = (0.0,)
            debug_pr('WARNING: feature {} did not have a fitness score.'.format(key),1)
        ind.fitness.values = fit_tuple
    evals_after = afs_tool.eval_counter

    return evals_after - evals_before
