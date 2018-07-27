from collections import deque
import functools
import math
import numpy as np
import operator
import os
import pandas as pd
import random
import numbers

import sys
if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import _pickle as pickle

from deap import algorithms, base, creator, gp, tools

##################################
# User-defined imports
##################################
from global_definitions import GLOBAL_DEBUG_LEVEL, my_timer, debug_pr, tic, toc

from custom import customTools
from custom import customAlgorithms
from custom import customFitnessEvaluators
from custom import customGp
from custom.population import PopulationBuilder
from model_validation.model_validation_wrapper import ModelValidationWrapper
from model_validation.validation_output_visualizer import ValidationOutputVisualizer
from util import utils

def quickload(qualified_path_str):
    """
    Method for importing and returning a refrence to a python identifier
    
    @param qualified_path_str: fully-qualified string

    Return: the class, object, or function reference

    For example, if qualified_path_str is 'sklearn.linear_model.LogisticRegression',
     this method will import LogisticRegression from the module sklearn.linear_model.
     The value returned is the LogisticRegression class object.
     If qualified_path_str is 'custom.customGp.full_uniform_mutation', this method
     will import the function full_uniform_mutation from the module custom.customGp.
     The value returned is the full_uniform_mutation function handle.
    """
    import_str, target_str, target_obj = utils.source_decode(qualified_path_str)
    return target_obj

class AFSTool(object):

    _PICKLE_SAVE_VERSION = 2 # update this if/when a change to pickle file formatting is made

    DEFAULT_SEED = 0

    DEFAULT_FNAME_ZERO_PADDING = 3
    DEFAULT_SPLIT_TEST_DATA_IF_NOT_PROVIDED =True
    DEFAULT_PCT_TEST_DATA = 0.25

    DEFAULT_POP_SIZE = 200
    DEFAULT_HOF_SIZE = 20
    DEFAULT_GENERATIONS_PER_EPOCH = 10
    DEFAULT_CROSSOVER_PROB = 0.7
    DEFAULT_MUTATION_PROB = 0.3

    DEFAULT_TARGET_NAME = 'TARGET'
    DEFAULT_COLUMNS_TO_DROP = ['ID', 'POLICY_NUMBER']
    DEFAULT_DELIMITER = '|'

    DEFAULT_DOWNSAMPLE_N_TRAIN = 'None'
    DEFAULT_DOWNSAMPLE_N_TEST = 'None'
    DEFAULT_BALANCE_TRAINING_SET = 'False'
    DEFAULT_BALANCE_POSITIVE_CLASS_RATIO = 0.5

    DEFAULT_OPERATOR_DICT = \
    {
        'operator.mul': '([float, float], float)'
        , 'custom.customOperators.encode_labels': '([str],float)'
        , 'custom.customOperators.z_score_by_group': '([float,str],float)'
        , 'custom.customOperators.categorical_concat': '([str,str],str)'
        , 'custom.customOperators.bin_4': '([float],str)'
        , 'custom.customOperators.bin_10': '([float],str)'
        , 'custom.customOperators.max_by_group': '([float,str],float)'
        , 'custom.customOperators.min_by_group': '([float,str],float)'
        , 'custom.customOperators.conjuction': '([bool,bool],bool)'
        , 'custom.customOperators.disjuction': '([bool,bool],bool)'
        , 'custom.customOperators.negation': '([bool],bool)'
    }
    DEFAULT_RETURN_TYPE_LIST = '[float, bool, str]'

    DEFAULT_BUILDER_GEN_FCN = 'custom.customGp.genHalfAndHalfTyped'
    DEFAULT_BUILDER_MIN_DEPTH = 0
    DEFAULT_BUILDER_MAX_DEPTH = 0
    DEFAULT_BUILDER_WEIGHT = 1

    DEFAULT_HOF_RECALCULATE_FLAG = True
    DEFAULT_EVOLUTIONARY_ALGORITHM = 'custom.customAlgorithms.eaMuPlusLambdaExclusive'
    @property
    def DEFAULT_ALGO_ARGS(self):
        return \
        {
            'mu': self.pop_size
            , 'lambda_': self.pop_size / 2
        }

    DEFAULT_EVALUATE_METHOD = 'corr_target_minus_bfs' # 'wrapper'
    DEFAULT_WRAPPER_CLASSIFIER = 'None'
    DEFAULT_WRAPPER_CLASSIFIER_ARGS = \
    {
        'n_estimators': 100
        , 'random_state': 0
        , 'max_features': 4
    }
    DEFAULT_SELECT_FCN = 'deap.tools.selBest'
    DEFAULT_MUTATE_FCN = 'custom.customGp.full_uniform_mutation'
    DEFAULT_MUTATE_MIN_HEIGHT = 0
    DEFAULT_MUTATE_MAX_HEIGHT = 0
    DEFAULT_MATE_FCN = 'custom.customTools.cxJoinWithRandomOp'
    DEFAULT_MATE_MIN_HEIGHT = 0
    DEFAULT_MATE_MAX_HEIGHT = 2

    DEFAULT_BFS_METRIC = 'auc'
    DEFAULT_BFS_MIN_SIZE = 5
    DEFAULT_BFS_MAX_SIZE = 50
    DEFAULT_BFS_INITIAL_SIZE = 5
    DEFAULT_BFS_MAX_REMOVE_EVALS = 10
    DEFAULT_BFS_MAX_ADD_EVALS = 10
    DEFAULT_BFS_ADD_THRESHOLD = 0.0005
    DEFAULT_BFS_REMOVE_THRESHOLD = 0.0005
    DEFAULT_BFS_REPLACE_THRESHOLD = 0.0
    DEFAULT_BFS_N_CV_FOLDS = 3
    DEFAULT_BFS_ADD_FROM_POP = True
    DEFAULT_BFS_ADD_FROM_HOF = True
    DEFAULT_BFS_CLASSIFIER = 'sklearn.linear_model.LogisticRegression'
    DEFAULT_BFS_CLASSIFIER_ARGS = {}
    DEFAULT_BFS_INITIALIZE_METHOD = 'hof' # ['hof','abt']
    DEFAULT_VALIDATION_CLASSIFIER = 'sklearn.linear_model.LogisticRegression'
    DEFAULT_VALIDATION_CLASSIFIER_ARGS = {}

    DEFAULT_BFS_CONVERGENCE_N_EPOCHS = 3 # Consider algo converged if no improvemnt seen over this many epochs
    DEFAULT_POP_CONVERGENCE_EVALS_PER_GEN = 3 # consider population converged if fewer than this many evals per generation in any epoch

    DEFAULT_OUTPUT_SETTINGS_DICT = \
    {
        'title': 'Binary Classification'
        , 'save_feature_list': True
        , 'save_confusion_matrix': True
        , 'save_roc': True
        , 'save_crc': False
        , 'threshold': None
    }
        
    data_stats_dict = dict() # Static dictionary to keep track of training population stats

    def __init__(self, abt_path, population_directory, config={}, abt_test_path=None, resume_file_prefix=None):
        """
        Initialization for AFSTool objects
        
        @param abt_path: location of the training data
        @param population_directory: output directory for population save files
        @param config: configuration dictionary
        @param abt_test_path: location of the testing data
        @param resume_file_prefix: naming pattern for population save files (ex. 'my_afs_run')
        
        Returns: nothing

        """
        clear_stats_dict()

        config['abt_path'] = abt_path
        config['population_directory'] = population_directory
        config['abt_test_path'] = abt_test_path
        config['resume_file_prefix'] = resume_file_prefix

        self.config=config

        self._seed = config.get('seed', AFSTool.DEFAULT_SEED)
        random.seed(self._seed)
        np.random.seed(self._seed)

        self._eval_counter = 0
        self.epoch = 0

        self.pop_size = config.get('pop_size', AFSTool.DEFAULT_POP_SIZE)
        self.hof_size = config.get('hof_size', AFSTool.DEFAULT_HOF_SIZE)
        
        self.generations_per_epoch = config.get('generations_per_epoch', AFSTool.DEFAULT_GENERATIONS_PER_EPOCH)
        self.crossover_prob = config.get('crossover_prob', AFSTool.DEFAULT_CROSSOVER_PROB)
        self.mutation_prob = config.get('mutation_prob', AFSTool.DEFAULT_MUTATION_PROB)
            
        self.abt_path = abt_path
        self.abt_test_path = abt_test_path

        self.pop_dir = population_directory

        downsample_n_train = config.get( 'downsample_n_train', AFSTool.DEFAULT_DOWNSAMPLE_N_TRAIN )
        downsample_n_test = config.get( 'downsample_n_test', AFSTool.DEFAULT_DOWNSAMPLE_N_TEST )
        balance_training_set = config.get( 'balance_training_set', AFSTool.DEFAULT_BALANCE_TRAINING_SET )
        balance_positive_class_ratio = config.get( 'balance_positive_class_ratio', AFSTool.DEFAULT_BALANCE_POSITIVE_CLASS_RATIO )
        test_train_split_flag = config.get('split_test_data_if_not_provided', AFSTool.DEFAULT_SPLIT_TEST_DATA_IF_NOT_PROVIDED)
        pct_train = 1 - config.get('pct_test_data', AFSTool.DEFAULT_PCT_TEST_DATA)
        target_name = config.get('target_name', AFSTool.DEFAULT_TARGET_NAME)
        self.read_abt(downsample_n_train=downsample_n_train
            , downsample_n_test=downsample_n_test
            , test_train_split_flag=test_train_split_flag
            , pct_train=pct_train
            , target_name=target_name
            , balance_training_set=balance_training_set
            , balance_positive_class_ratio=balance_positive_class_ratio)
        self.drop_non_training_columns(config.get('columns_to_drop', AFSTool.DEFAULT_COLUMNS_TO_DROP))
        self.data_eval_dict = dict() # Keeps track of every already-evaluated feature column
        self.data_eval_dict_test = dict() # Ditto, but for the test data
        self.feature_set_score_dict = dict() # keeps track of alredy-scored feature sets
            
        self.arg_dict = dict() # dictionary to map "ARG0", "ARG1" to descriptive names "FEATURE_1", "FEATURE_2", etc.
        self.internal_prefix = 'ARG'
        i = 0
        for feature in self.data:
            key = self.internal_prefix + str(i)
            self.arg_dict[key] = feature
            i=i+1
        
        ret_types = eval( config.get('return_type_list', AFSTool.DEFAULT_RETURN_TYPE_LIST) )
        self.make_primitive_set_typed(ret_types
                    , operator_dict=config.get('operator_dict', AFSTool.DEFAULT_OPERATOR_DICT))

        self.evolutionary_algorithm = quickload(config.get('evolutionary_algorithm'
                                , AFSTool.DEFAULT_EVOLUTIONARY_ALGORITHM))
        self.ea_args = config.get('algo_args', self.DEFAULT_ALGO_ARGS)
        self.hof_recalculate_flag = config.get('hof_recalculate_flag', AFSTool.DEFAULT_HOF_RECALCULATE_FLAG)

        builder = PopulationBuilder()

        builder.pset = self.pset
        builder.gen = quickload( config.get('builder_gen_fcn', AFSTool.DEFAULT_BUILDER_GEN_FCN) )
        builder.min_depth = config.get('builder_min_depth', AFSTool.DEFAULT_BUILDER_MIN_DEPTH)
        builder.max_depth = config.get('builder_max_depth', AFSTool.DEFAULT_BUILDER_MAX_DEPTH)
        builder.weight = config.get('builder_weight', AFSTool.DEFAULT_BUILDER_WEIGHT)

        mutate_fcn = quickload( config.get('mutate_fcn', AFSTool.DEFAULT_MUTATE_FCN) )
        mutate_min_height = config.get('mutate_min_height', AFSTool.DEFAULT_MUTATE_MIN_HEIGHT)
        mutate_max_height = config.get('mutate_max_height', AFSTool.DEFAULT_MUTATE_MAX_HEIGHT)
        mate_fcn = quickload( config.get('mate_fcn', AFSTool.DEFAULT_MATE_FCN) )
        mate_min_height = config.get('mate_min_height', AFSTool.DEFAULT_MATE_MIN_HEIGHT)
        mate_max_height = config.get('mate_max_height', AFSTool.DEFAULT_MATE_MAX_HEIGHT)
        builder.add_individual_function('mutate', mutate_fcn, pset=self.pset, min_=mutate_min_height, max_=mutate_max_height)
        builder.add_individual_function('mate', mate_fcn, pset=self.pset)
        
        #-------------------------------------------------------------------------------
        # Decorate functions for individuals
        #-------------------------------------------------------------------------------

        height_limit_decorator = lambda max_value: gp.staticLimit(key=operator.attrgetter("height"), max_value=max_value)
        builder.decorate_individual_function('mutate', height_limit_decorator, max_value=mutate_max_height)
        builder.decorate_individual_function('mate', height_limit_decorator, max_value=mate_max_height)

        builder.add_population_function('select', quickload( config.get('select_fcn', AFSTool.DEFAULT_SELECT_FCN) ))
        
        wrapper_classifier = config.get('wrapper_classifier', AFSTool.DEFAULT_WRAPPER_CLASSIFIER)
        if wrapper_classifier is None or wrapper_classifier == 'None':
            # use filter method
            method = config.get('evaluate_method',AFSTool.DEFAULT_EVALUATE_METHOD)
            if (method == 'corr_target_minus_bfs'):
                builder.add_population_function('evaluate'
                    , customFitnessEvaluators.evaluateCorrelation
                    , afs_tool=self
                    , comparison_set = 'bfs')
            elif (method == 'corr_target_minus_hof'):
                builder.add_population_function('evaluate'
                    , customFitnessEvaluators.evaluateCorrelation
                    , afs_tool=self
                    , comparison_set = 'hof')
            elif (method == 'corr_target'):
                builder.add_population_function('evaluate'
                    , customFitnessEvaluators.evaluateCorrelation
                    , afs_tool=self
                    , comparison_set = None)
        else:
            wrapper_args = config.get('wrapper_classifier_args', AFSTool.DEFAULT_WRAPPER_CLASSIFIER_ARGS)
            wrapper_classifier_type = quickload(wrapper_classifier)
            wrapper_classifier_instance = wrapper_classifier_type(**wrapper_args)
            builder.add_population_function('evaluate', customFitnessEvaluators.evaluateModelWrapper
                , afs_tool=self, classifier=wrapper_classifier_instance)

        self.population = builder.build(n=self.pop_size)

        #populate the output settings dictionary
        self.output_settings=config.get('output_settings', AFSTool.DEFAULT_OUTPUT_SETTINGS_DICT)
        if (not 'threshold' in self.output_settings) or (not isinstance(self.output_settings['threshold'], numbers.Number)):
            self.output_settings['threshold'] = None
    
        ##################################
        # Initialize the BFS
        ##################################
        self.bfs_metric = config.get('bfs_metric', AFSTool.DEFAULT_BFS_METRIC)                # which metric to use for scoring
        self.bfs_min_size = config.get('bfs_min_size', AFSTool.DEFAULT_BFS_MIN_SIZE)            # Don't allow the BFS to shrink beyond this size
        self.bfs_max_size = config.get('bfs_max_size', AFSTool.DEFAULT_BFS_MAX_SIZE)           # Don't allow the BFS to grow beyond this size
        self.bfs_initial_size = config.get('bfs_initial_size', AFSTool.DEFAULT_BFS_INITIAL_SIZE)
        self.bfs_max_remove_evals = config.get('bfs_max_remove_evals', AFSTool.DEFAULT_BFS_MAX_REMOVE_EVALS)   # Number of features to evaluate from the remove queue
        self.bfs_max_add_evals = config.get('bfs_max_add_evals', AFSTool.DEFAULT_BFS_MAX_ADD_EVALS)      # Number of features to evaluate from the add queue
        self.bfs_add_threshold = config.get('bfs_add_threshold', AFSTool.DEFAULT_BFS_ADD_THRESHOLD)      # The higher the threshold, the more difficult it is to add features
        self.bfs_remove_threshold = config.get('bfs_remove_threshold', AFSTool.DEFAULT_BFS_REMOVE_THRESHOLD)    # The higher the threshold, the more easily features get dropped
        self.bfs_replace_threshold = config.get('bfs_replace_threshold', AFSTool.DEFAULT_BFS_REPLACE_THRESHOLD)     # The higher the threshold, the more difficult it is to replace a feature
        self.bfs_n_cv_folds = config.get('bfs_n_cv_folds', AFSTool.DEFAULT_BFS_N_CV_FOLDS)                 # Number of CV folds for BFS wrapper scoring method
        self.bfs_add_from_pop = config.get('bfs_add_from_pop', AFSTool.DEFAULT_BFS_ADD_FROM_POP)          # Whether or not to put individuals from the population into the BFS add queue
        self.bfs_add_from_hof = config.get('bfs_add_from_hof', AFSTool.DEFAULT_BFS_ADD_FROM_HOF)          # Whether or not to put individuals from the hall of fame into the BFS add queue 
        self.bfs_score = 0

        # Initialize current "best feature set"
        bfs_initialize_method = config.get('bfs_initialize_method', AFSTool.DEFAULT_BFS_INITIALIZE_METHOD)
        if (bfs_initialize_method == 'hof'):
            self.bfs = set()                   # Setting BFS to empty set will iniitialize later with first epoch HOF
        elif (bfs_initialize_method == 'abt'):
            sample_size = np.min([self.bfs_initial_size, len(self.data.columns)])
            self.bfs = set(random.sample(self.data.columns, sample_size))
        else:
            self.bfs = set()                   # Setting BFS to empty set will iniitialize later with first epoch HOF
        
        self.best_seen_bfs = None
        self.best_seen_bfs_score = None
        self.epochs_since_bfs_improvement = 0
        self.evals_per_gen = None

        self.pop_convergence_evals_per_gen = config.get('pop_convergence_evals_per_gen',AFSTool.DEFAULT_POP_CONVERGENCE_EVALS_PER_GEN)
        self.bfs_convergence_n_epochs = config.get('bfs_convergence_n_epochs',AFSTool.DEFAULT_BFS_CONVERGENCE_N_EPOCHS)

        self.pop_is_converged = False
        self.bfs_is_converged = False


        # Initialize the queues for adding and removing features to the BFS 
        self.candidate_feature_add_queue = deque()
        self.candidate_feature_remove_queue = deque()

        cls_type = quickload( config.get('bfs_classifier', AFSTool.DEFAULT_BFS_CLASSIFIER) )
        cls_args = config.get('bfs_classifier_args', AFSTool.DEFAULT_BFS_CLASSIFIER_ARGS)
        self.bfs_classifier = cls_type(**cls_args)
        cls_type = quickload( config.get('validation_classifier', AFSTool.DEFAULT_VALIDATION_CLASSIFIER) )
        cls_args = config.get('validation_classifier_args', AFSTool.DEFAULT_VALIDATION_CLASSIFIER_ARGS)
        self.validation_classifier = cls_type(**cls_args)
        
        self.hof = tools.HallOfFame(self.hof_size, similar=customTools.equals)

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats_size = tools.Statistics(len)
        self.mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        self.mstats.register("avg", np.mean)
        self.mstats.register("std", np.std)
        self.mstats.register("min", np.min)
        self.mstats.register("max", np.max)

        if (resume_file_prefix is None):
            self.save_file_root = self.create_save_file_root(leading_zeros=config.get('fname_zero_padding', AFSTool.DEFAULT_FNAME_ZERO_PADDING))
        else:
            resume_file_prefix = resume_file_prefix + '.p'
            self.save_file_root = self.pop_dir+'/'+resume_file_prefix
            load_file = utils.new_filenames(self.save_file_root)[1]
            if load_file is None:
                debug_pr('Starting a new population.', 2)
            else:
                debug_pr('Loading an existing population.', 2)
                self.load(load_file)
                config['loaded_file'] = load_file

    def run(self, n_epochs=1):
        """
        Method for running one or more epochs of the AFS Tool
        
        @param n_epochs: the number of epochs to run

        Return: nothing

        The each epoch consists of:
        - (if population hasn't converged) run Genetic Program for some number of generations (self.generations_per_epoch) 
        - determine whether population is converged
        - run Best Feature Set algorithm
        - determine whether Best Feature Set has converged
        """
        for epoch_ind in range(n_epochs):
            epoch = self.epoch
            debug_pr('Epoch {}, generations = {}'.format(epoch, self.generations_per_epoch),1)
            
            if(self.pop_is_converged):
                debug_pr('Population is converged... Skipping Evolutionary Algorithm.',1)
            else:
                evals = self.eval_counter
                self.population, log = self.evolutionary_algorithm(self.population
                                , cxpb=self.crossover_prob
                                , mutpb=self.mutation_prob
                                , ngen=self.generations_per_epoch
                                , stats=self.mstats
                                , halloffame=self.hof
                                , verbose=GLOBAL_DEBUG_LEVEL>=1
                                ,  hof_recalculate_flag=self.hof_recalculate_flag
                                , **self.ea_args)
                evals = self.eval_counter - evals
                self.evals_per_gen = (1.0*evals)/self.generations_per_epoch
                debug_pr('Evals per Generation = {}'.format(self.evals_per_gen),2)
                debug_pr('Size of BFS = {}'.format(len(self.bfs)), 2)
                if self.evals_per_gen <= self.pop_convergence_evals_per_gen:
                    self.pop_is_converged = True
            epoch = epoch + 1
            self.epoch = epoch

            # If necessary, initialize the BFS from the HOF
            if self.bfs is None or len(self.bfs) < 1:
                self.bfs = set()
                for i in range(self.bfs_initial_size):
                    if (i > len(self.hof)):
                        break
                    self.bfs.add(str(self.hof[i]))

            self.epochs_since_bfs_improvement += 1
            self.maintain_bfs()
            if (self.epochs_since_bfs_improvement >= self.bfs_convergence_n_epochs):
                self.bfs_is_converged = True
            else:
                self.bfs_is_converged = False
            debug_pr('Epochs sincee BFS improvement = {}'.format(self.epochs_since_bfs_improvement),2)
            self.save(utils.no_overwrite_filename(self.save_file_root))

    def save(self, filename):
        """
        Method for dumping the necessary data to a pickle file
        
        @param filename: file to write (will overwrite if exists)

        Return: nothing

        Generally speaking, we pickle enough to re-produce the features 
        that were in the population, hof, and bfs this epoch, as well as
        some stats about performance of the algorithm.

        When changing the items included in the pickled file, please
        increment the _PICKLE_SAVE_VERSION to indicate the change.
        """
        import abc
        pickle_save_obj = dict()
        pickle_save_obj['population'] = self.population
        pickle_save_obj['abt_columns'] = self.data.columns
        pickle_save_obj['hof'] = self.hof
        pickle_save_obj['bfs'] = self.bfs
        pickle_save_obj['bfs_score'] = self.bfs_score
        pickle_save_obj['best_seen_bfs'] = self.best_seen_bfs
        pickle_save_obj['best_seen_bfs_score'] = self.best_seen_bfs_score
        pickle_save_obj['pop_is_converged'] = self.pop_is_converged
        pickle_save_obj['bfs_is_converged'] = self.bfs_is_converged
        pickle_save_obj['evals_per_gen'] = self.evals_per_gen
        pickle_save_obj['epochs_since_bfs_improvement'] = self.epochs_since_bfs_improvement
        pickle_save_obj['config'] = self.config
        pickle_save_obj['version'] = AFSTool._PICKLE_SAVE_VERSION
        debug_pr('WRITING {}'.format(filename), 2)
        pickle_file_writer = open(filename, 'wb')
        pickle.dump(pickle_save_obj, pickle_file_writer, -1)
        debug_pr('SAVED SUCCESSFULLY.', 2)

    def load(self, filename):
        """
        Method for loading pre-existing results into the AFSTool
        
        @param filename: file to load

        Return: True if success, otherwise False

        """
        if(os.path.exists(filename)):
            # Load the population data from the pickle
            debug_pr('LOADING {}'.format(filename), 2)
            pickle_file_reader = open(filename, 'rb')
            pickle_obj = pickle.load(pickle_file_reader)
            self.population = pickle_obj['population']
            self.hof = pickle_obj['hof']
            self.bfs = pickle_obj['bfs']
            # TODO: Check for metric change. We are assuming here that the evaluation metric has not changed
            if ('bfs_score' in pickle_obj):
                self.bfs_score = pickle_obj['bfs_score']
            if ('best_seen_bfs' in pickle_obj):
                self.best_seen_bfs = pickle_obj['best_seen_bfs']
            if ('best_seen_bfs_score' in pickle_obj):
                self.best_seen_bfs_score = pickle_obj['best_seen_bfs_score']
            debug_pr('LOADED SUCCESSFULLY.', 2)
            return True
        else:
            debug_pr('File {} does not exist.'.format(filename), 2)
            return False

    def maintain_bfs(self):
        """
        Method for executing the BFS feature selection algorithm

        Return: nothing

        Steps in this method:
        - update the queues for deletion, addition of BFS features
        - attempt to remove up to bfs_max_remove_evals from the BFS
        -- if the wrapper model score does not drop, the feature is
            "unneeded" and is dropped from the BFS
        - attempt to add or replace up to bfs_max_add_evals to the BFS
        -- if the wrapper model score improves by either addition (or
            replacement of the most highly-correlated BFS feature),
            that feature is either inserted or replaces the correlated
            feature, depending on which score was higher

        Once insertion/deletion/replacement no longer improves the
         score, the BFS will stop changing.
        Tuning parameters allow thresholds to be set how easy or
         difficult it is for a feature to be added or dropped.
        - bfs_remove_threshold
        - bfs_add_threshold
        - bfs_replace_threshold
        """
        # Add all individuals in the BFS to the queue for potential removal
        for individual in self.bfs:
            feature_key = str(individual)
            if (feature_key not in self.candidate_feature_remove_queue):
                self.candidate_feature_remove_queue.append(feature_key)
        # Add all individuals in the current population to the queue for potential addition
        if self.bfs_add_from_hof:
            for individual in self.hof:
                feature_key = str(individual)
                if (feature_key not in self.candidate_feature_add_queue):
                    self.candidate_feature_add_queue.append(feature_key)
        if self.bfs_add_from_pop:
            for individual in self.population:
                feature_key = str(individual)
                if (feature_key not in self.candidate_feature_add_queue):
                    self.candidate_feature_add_queue.append(feature_key)

        # Score the baseline BFS
        score_baseline = self.score_bfs_set(self.bfs, metric=self.bfs_metric)

        debug_pr('BFS '+self.bfs_metric.upper()+' Score: ' + str(score_baseline),2)
        debug_pr('',2)
        debug_pr('Removing unneeded features from BFS.',2)
        for i in range(self.bfs_max_remove_evals):
            # Don't try to remove anything if mimimum size has been reached
            # Don't try to remove anything if queue of candidates is empty
            if (len(self.bfs) <= self.bfs_min_size or len(self.candidate_feature_remove_queue) < 1):
                break

            # Take the first candidate from the queue
            individual = self.candidate_feature_remove_queue.popleft()

            # If individual is no longer in BFS, skip it
            if not (individual in self.bfs):
                continue

            # Create temporary set (with individual removed) to compare with the BFS
            bfs_tmp = self.bfs - set([individual])

            # Score the temporary set
            score_tmp = self.score_bfs_set(bfs_tmp, metric=self.bfs_metric)
            
            # delete the feature if unneeded
            if (score_tmp > score_baseline - self.bfs_remove_threshold):
                score_baseline = score_tmp
                debug_pr('Deleting feature '+utils.string_replace_using_dict(individual, self.arg_dict)+' from BFS. '+str(score_tmp),2)
                self.bfs = bfs_tmp
        debug_pr('', 2)
        debug_pr("Adding potential features to BFS.", 2)
        for i in range(self.bfs_max_add_evals):
            # Don't try to add anything if maximum size has been reached
            # Don't try to add anything if queue of candidates is empty
            if(len(self.bfs) >= self.bfs_max_size or len(self.candidate_feature_add_queue) < 1):
                break
                
            # Take the first candidate from the queue
            individual = self.candidate_feature_add_queue.popleft()
            
            # If individual is already in BFS, skip it and take another from the queue
            while (individual in self.bfs):
                if (len(self.candidate_feature_add_queue) < 1):
                    break
                # Take the first candidate from the queue
                individual = self.candidate_feature_add_queue.popleft()

            debug_pr('Individual = '+str(individual), 3)
            bfs_tmp = self.bfs | set([individual])

            # Score the set with this new feature added
            score_tmp_add = self.score_bfs_set(bfs_tmp, metric=self.bfs_metric)

            # Also Score the set with the most similar feature replaced
            similar_feature_key = self.most_similar_feature(individual, self.bfs)
            bfs_tmp = bfs_tmp - set([similar_feature_key])
            score_tmp_rep = self.score_bfs_set(bfs_tmp, metric=self.bfs_metric)

            debug_pr("Baseline = "+str(score_baseline),3)
            debug_pr("Add      = "+str(score_tmp_add),3)
            debug_pr("Replace  = "+str(score_tmp_rep),3)
            # ADD to BFS or REPLACE a BFS feature... If either improves the BFS score
            if (score_tmp_add - self.bfs_add_threshold > score_tmp_rep - self.bfs_replace_threshold):
                if (score_tmp_add > score_baseline + self.bfs_add_threshold):
                    score_baseline = score_tmp_add
                    debug_pr('Adding feature '+utils.string_replace_using_dict(individual, self.arg_dict)+' to BFS. '+str(score_tmp_add),2)
                    self.bfs = self.bfs | set([individual])
            else:
                if (score_tmp_rep > score_baseline + self.bfs_replace_threshold):
                    score_baseline = score_tmp_rep
                    debug_pr('Replacing feature '+utils.string_replace_using_dict(similar_feature_key, self.arg_dict)\
                     +' with '+utils.string_replace_using_dict(individual, self.arg_dict)+' in BFS. '+str(score_tmp_rep),2)
                    self.bfs = bfs_tmp
        
        # update the recorded bfs_score
        self.bfs_score = score_baseline

        debug_pr('',2)
        self.bfs_summary()

    def bfs_summary(self):
        """
        Method for printing a summary of the BFS features

        Return: nothing

        """
        debug_pr("BFS Summary for epoch {}:".format(self.epoch),2)
        for feature in self.bfs:
            # replace generic argument names with descriptive ones
            pretty_feature = utils.string_replace_using_dict(str(feature), self.arg_dict)
            debug_pr(pretty_feature,2)

    def score_bfs_set(self
        , feature_set
        , metric='auc'
        , classifier=None
        , is_validation_run=False
        , save_output=False
        , output_settings=None
        , save_suffix='NA'):
        """
        Method for scoring a feature set
        
        @param feature_set: the iterable collection of features (could be a Population, a set, or a list)
        @param classifier: the classifier to use. If None, score_bfs_set will use self.bfs_classifier
        @param is_validation_run: when False, use score from cross-validated training data
                                  when True, use training data for training, and test data for scoring
        @param plot_settings: a dictionary defining directory name for the feature list
                              and labels for plottig. When None, no plotting or file-list is produced

        Return: AUC Score (floating point value)

        """

        if save_output and not output_settings:
            if not self.output_settings:
                output_settings = DEFAULT_OUTPUT_SETTINGS_DICT
            else:
                output_settings = self.output_settings


        if (len(feature_set)==0):
            msg = 'Set to be score must have a nonzero size.'
            raise ValueError(msg)

        if (classifier is None):
            classifier = self.bfs_classifier

        if (is_validation_run):
            X_test = self.get_feature_set_dataframe(feature_set, is_validation_run=True)
            y_test = self.target_test
            X_test=X_test.fillna(0)
        else:
            feature_set_str = self.feature_set_to_string(feature_set)
            if feature_set_str in self.feature_set_score_dict:
                debug_pr('skipping known feature set '+feature_set_str,3)
                return self.feature_set_score_dict[feature_set_str]
        
        X = self.get_feature_set_dataframe(feature_set)
        y = self.target
        X=X.fillna(0)

        wrapper = ModelValidationWrapper(classifier)

        wrapper.train(X, y, self.bfs_n_cv_folds)

        if is_validation_run:
            wrapper.test(X_test,y_test)
            viz = ValidationOutputVisualizer(wrapper.holdout_validation_output)
            thresh = viz.f1_score_max_threshold(tolerance=0.0001, guess=0.002)
            confusion_mat = viz.confusion_matrix(thresh)
            precision = 1.0*confusion_mat[1,1]/(confusion_mat[1,1]+confusion_mat[0,1])
            recall = 1.0*confusion_mat[1,1]/(confusion_mat[1,1]+confusion_mat[1,0])
            f1_validation_score = viz.f1_score(thresh)
            auc_validation_score = viz.auc_score()
            debug_pr("Precision at threshold {}: \n{}".format(thresh,precision),2)
            debug_pr("Recall at threshold {}: \n{}".format(thresh,recall),2)
            debug_pr("Confusion Matrix at threshold {}: \n{}".format(thresh,confusion_mat),2)
            debug_pr('{} Feature Set {} Score: {}'.format(save_suffix.upper(),'AUC',auc_validation_score),2)
            debug_pr('{} Feature Set {} Score: {}'.format(save_suffix.upper(),'F1',f1_validation_score),2)
            if metric=='auc':
                score = auc_validation_score
            elif metric=='f1':
                score = f1_validation_score


            # tic()
            # thresh = viz.f1_score_max_threshold(tolerance=0.0001, guess=0.002)
            # print(thresh)
            # toc()
            # tic()
            # thresh = viz.f1_score_max_threshold_backtracking(ratio=0.75,tolerance=0.0001, guess=0.002)
            # print(thresh)
            # toc()
            # print('')
            # print('')
                
                
        else:
            viz = ValidationOutputVisualizer(wrapper.cross_validation_output)
            if metric=='auc':
                score = viz.auc_score()
            elif metric=='f1':
                thresh = viz.f1_score_max_threshold(tolerance=0.0001, guess=0.002)
                score = viz.f1_score(thresh)
            if (self.best_seen_bfs is None) or (self.best_seen_bfs_score < score):
                self.best_seen_bfs = feature_set
                self.best_seen_bfs_score = score
                self.epochs_since_bfs_improvement = 0
            feature_set_str = self.feature_set_to_string(feature_set)
            self.feature_set_score_dict[feature_set_str] = score
            debug_pr('.',3)

        if save_output:
            output_filepath_root = utils.new_filenames(self.save_file_root)[1]
            output_filepath_root, discard = os.path.splitext(output_filepath_root)
            if output_settings['save_feature_list']:
                title = output_settings['title']
                fl_filepath = output_filepath_root + '_features_' + save_suffix + '.txt'
                feature_list_writer = open(fl_filepath, 'w')
                sorted_fl = self.sorted_feature_list(feature_set)
                for feature in sorted_fl:
                    feature_list_writer.write(feature+'\n')
                feature_list_writer.close()

            if output_settings['save_confusion_matrix']:
                tile = output_settings['title'] + ' ' + save_suffix.upper() +' CONFUSION MATRIX'
                confmat_filepath = output_filepath_root + '_confmat_' + save_suffix + '.png'
                viz.plot_confusion_matrix(threshold=thresh, title=title, path=confmat_filepath)

            if output_settings['save_roc']:
                tile = output_settings['title'] + ' ' + save_suffix.upper() +' ROC'
                roc_filepath = output_filepath_root + '_roc_' + save_suffix + '.png'
                viz.roc(title=title, path=roc_filepath)
            
            if output_settings['save_crc']:
                tile = output_settings['title'] + ' ' + save_suffix.upper() +' CRC'
                crc_filepath = output_filepath_root + '_crc_' + save_suffix + '.png'
                viz.cumulative_response_curve(population_threshold=0.01, title=title, path=crc_filepath)
        return score

    def sorted_feature_list(self, feature_set):
        sorted_fl = list()
        for feature in feature_set:
            feature_str = utils.string_replace_using_dict(str(feature), self.arg_dict)
            sorted_fl.append(feature_str)
        sorted_fl.sort()
        return sorted_fl

    def feature_set_to_string(self, feature_set):
        return '|'.join(self.sorted_feature_list(feature_set))

    def hof_summary(self):
        """
        Method for printing a summary of the current Hall of Fame

        Return: nothing

        """
        debug_pr('HOF Summary',2)
        for h in self.hof:
            pretty_h = utils.string_replace_using_dict(str(h), self.arg_dict)
            debug_pr('{}\t{}'.format(h.fitness.values[0], pretty_h),2)

    def read_abt(self
        , downsample_n_train=None
        , downsample_n_test=None
        , test_train_split_flag=True
        , pct_train=0.75
        , target_name='TARGET'
        , balance_training_set=False
        , balance_positive_class_ratio=0.5):
        """
        Method for reading the abt
        
        @param downsample_n_train: downsample to this number of training rows
        @param downsample_n_test: downsample to this number of testing rows
        @param test_train_split_flag: when True and test abt is not given, create test set via test/train split
        @param pct_train: ratio of training to testing rows
                          (only used when test_train_split_flag=True and test abt is not given)
        @param target_name: name of the target column
                          (currently no stratified sampling is done, but this could be added later)

        Return: nothing

        """
        abt_pathname, abt_pathext = os.path.splitext(self.abt_path)
        abt_test_data = None
        if (abt_pathext in ['.p']):
            abt_data = pd.read_pickle(self.abt_path)
            if not (self.abt_test_path is None):
                abt_test_data = pd.read_pickle(self.abt_test_path)
        elif (abt_pathext in ['.h5']):
            H5_KEY_NAME = 'table'
            abt_data = pd.read_hdf(self.abt_path, H5_KEY_NAME)
            if not (self.abt_test_path is None):
                abt_test_data = pd.read_hdf(self.abt_test_path, H5_KEY_NAME)
        else:
            delimiter = self.config.get('delimiter',AFSTool.DEFAULT_DELIMITER)
            abt_data = pd.read_csv(self.abt_path
                            , warn_bad_lines=True
                            , error_bad_lines=False
                            , sep=delimiter)
            if not (self.abt_test_path is None):
                abt_test_data = pd.read_csv(self.abt_test_path
                            , warn_bad_lines=True
                            , error_bad_lines=False
                            , sep=delimiter)

        # create holdout data using test train split if needed
        
        if (self.abt_test_path is None) and test_train_split_flag:
            # Perform a train/test split
            msk = np.random.rand(len(abt_data)) < pct_train
            train = abt_data[msk]
            test = abt_data[~msk]
            abt_data = train
            abt_test_data = test

        # Use the downsample_n_train to make the training data set smaller for debugging purposes
        if not (downsample_n_train is None or str(downsample_n_train) == 'None'):
            if balance_training_set:
                stratified = abt_data.head(0)
                counts = abt_data[target_name].value_counts()
                classes = abt_data[target_name].unique()
                n_classes = abt_data[target_name].nunique()
                for clas in classes:
                    # Assuming binary classification here
                    strat_n_equal = downsample_n_train / n_classes
                    if counts[clas] < strat_n_equal:
                        weight = balance_positive_class_ratio
                    else:
                        weight = 1 - balance_positive_class_ratio
                    strat_n = int(downsample_n_train * weight)
                    sub_abt = abt_data[abt_data[target_name] == clas]
                    strat_sample = sub_abt.sample(n=strat_n
                            , axis=0
                            , replace = counts[clas] < strat_n)
                    stratified = stratified.append(
                        strat_sample)
                self.abt_data = stratified
            else:
                self.abt_data = abt_data.sample(n=downsample_n_train, axis=0)
        else:
            self.abt_data = abt_data

        if not (abt_test_data is None):
            if not (downsample_n_test is None or str(downsample_n_test) == 'None'):
                self.abt_test_data = abt_test_data.sample(n=downsample_n_test)
            else:
                self.abt_test_data = abt_test_data
        else:
            self.abt_test_data = None

        self.target_name = target_name
    
    def drop_non_training_columns(self, extra_columns_to_drop=[]):
        """
        Method for dropping the target column and additional columns not to be used by the tool
        
        @param extra_columns_to_drop: list of columns to ignore for feature engineering
                                      (for example, ID number column)
        Returns: nothing

        """
        self.data = self.abt_data.iloc[:, :]
        self.target = self.abt_data[self.target_name]

        # drop columns not to be used in feature engineering
        columns_to_drop = [self.target_name] + extra_columns_to_drop

        for drop_col in columns_to_drop:
            if drop_col in self.data:
                del self.data[drop_col]

        if not (self.abt_test_data is None):  
            self.data_test = self.abt_test_data.iloc[:, :]
            self.target_test = self.abt_test_data[self.target_name]
            for drop_col in columns_to_drop:
                if drop_col in self.data_test:
                    del self.data_test[drop_col]

    def most_similar_feature(self, feature_key, feature_key_set):
        """
        Method for finding the most highly correlated feature in a given set
        
        @param feature_key: feature to compare to the set
        @param feature_key_set: the feature set

        Returns: string key of the most similar feature from the set

        """
        highest_corr = -1
        matched_key = None
        for compare_key in feature_key_set:
            try:
                corr = customTools.get_feature_correlation(self.get_feature_col(feature_key), self.get_feature_col(compare_key), self)
            except:
                corr = 0
            if (corr > highest_corr):
                highest_corr = corr
                matched_key = compare_key
        return matched_key

    def get_feature_col(self, individual, df=None, eval_dict=None):
        """
        Method for retrieving a data column representing a single feature
        
        @param individual: feature to retrieve (could be Individual object OR string key)
        @param df: the abt dataframe to use (if none, use the training data)
        @param eval_dict: the eval dictionary to use (if none, use the training data)
                          this allows each engineered column to be computed only once.
                          Subsequent calls will load the column from the dictionary.
        
        Returns: Pandas Series representing the feature as a data column

        """
        if (eval_dict is None):
            eval_dict = self.data_eval_dict
        if (df is None):
            df = self.data
        key = str(individual).strip()
        if (key in eval_dict):          # if it is already in dictionary, look it up
            col = eval_dict[key]
        elif (key in df.columns):          # if it is a base column, look it up
            col = df[key]
            eval_dict[key] = col
        elif (key in self.arg_dict and self.arg_dict[key] in df.columns):          # if it is a base column, look it up
            col = df[self.arg_dict[key]]
            eval_dict[key] = col
        else:                                # compile the transformation
            col = customTools.apply_expression(key, self, df=df, eval_dict=eval_dict)
            eval_dict[key] = col
            self._eval_counter = self._eval_counter + 1
        return pd.core.series.Series(col)
    
    def get_feature_set_dataframe(self, individual_set, is_validation_run=False):
        """
        Method for retrieving a data frame representing a set of features
        
        @param individual_set: iterable collection of features feature to retrieve (could be Individual objects OR string keys)
        @param is_validation_run: when True, derive columns from the testing abt
                                  when False, derive columns from the training abt
        
        Returns: Pandas DataFrame representing the feature set

        """
        df = pd.DataFrame()
        for individual in individual_set:
            if (is_validation_run):
                df[str(individual)] = self.get_feature_col(str(individual), df=self.data_test, eval_dict=self.data_eval_dict_test)
            else:
                df[str(individual)] = self.get_feature_col(str(individual))
        return df
    
    def make_primitive_set_typed(self, return_types, operator_dict, ephemeral_constant_dict={}):
        """
        Method for creating the DEAP primitive set for Strongly Typed Genetic Programming
        
        @param return_types: list of the valid types of an engineered feature (i.e. root of the primitiveTree)
        @param operator_dict: dictionary defining GP operators (see afs_example.yaml for an example)
        @param ephemeral_constant_dict: dictionary defining GP ephemeral constants
                                        These are leaves of the PrimitiveTree that are some constant value
                                        instead of an ABT column
        
        Returns: nothing

        """
        # get the types of the input data
        arg_types_raw = self.data.dtypes
        arg_types = []
        for arg_type in arg_types_raw:
            if issubclass(arg_type.type, np.number):
                arg_types.append(np.float)
            elif issubclass(arg_type.type, bool) or issubclass(arg_type.type, np.bool_):
                arg_types.append(np.bool)
            elif issubclass(arg_type.type, str):
                arg_types.append(str)
            else:
                debug_pr('WARNING: type '+str(arg_type)+' is not supported. Attempting to treat it as a string.',5)
                arg_types.append(str)

        #defining a typed primitive set
        self.pset = gp.PrimitiveSetTyped('main'
            , in_types=arg_types
            , ret_type=return_types
            , prefix=self.internal_prefix)
        
        self.add_to_pset(operator_dict, ephemeral_constant_dict)

    def make_primitive_set(self):
        pass

    def add_to_pset(self, operator_dict={}, ephemeral_constant_dict={}):
        """
        Helper method for adding operators and constants to the primitive set

        @param operator_dict: dictionary defining GP operators (see afs_example.yaml for an example)
        @param ephemeral_constant_dict: dictionary defining GP ephemeral constants
                                        These are leaves of the PrimitiveTree that are some constant value
                                        instead of an ABT column
        
        Returns: nothing

        """
        
        ##################################
        # Add Ephemeral Constants
        ##################################
        for key in ephemeral_constant_dict:
            self.pset.addEphemeralConstant(key, eval(ephemeral_constant_dict[key]))

        ##################################
        # Add Operators
        ##################################
        for key in operator_dict:
            op_obj = quickload(key)
            value = eval(operator_dict[key])
            self.pset.addPrimitive(op_obj, value[0], value[1])
    
    def create_save_file_root(self, run_id=None, save_dir=None, leading_zeros=3):
        """
        Method for default save prefixes
        
        @param run_id: run id used in the filename
                       when None, create one by counting up (will never overwrite an existing series of save files)
        @param save_dir: base directory for save file
        @param leading_zeros: number of digits to use for generated run_id
        
        Returns: the save file root (with .p appended)

        """
        if (save_dir is None):
            save_dir = self.pop_dir
        if (run_id is None):
            id_format_pattern = '%0'+str(leading_zeros)+'d'
            path_tail = '_epoch_{}.p'.format(id_format_pattern % 0)
            id_ = 0
            file_root = save_dir + '/afs_run_' + (id_format_pattern % id_) + '_epoch'
            fname = file_root + '_{}.p'.format(id_format_pattern % 0)
            while os.path.exists(fname):
                id_ += 1
                file_root = save_dir + '/afs_run_' + (id_format_pattern % id_) + '_epoch'
                fname = file_root + '_{}.p'.format(id_format_pattern % 0)
        else:
            file_root = save_dir + '/afs_run_' + run_id + '_epoch'
        return file_root+'.p'

    @property
    def population(self):
        '''
        The population object
        '''

        return self._population

    @population.setter
    def population(self, value):
        self._population = value
    
    
    @property
    def eval_counter(self):
        '''
        A running count of .apply evaluations needed by get_feature_col
        '''

        return self._eval_counter


def clear_stats_dict():
    """
    Method for clearing the data_stats_dict

    This method can be used to clear old stats (such a means and standard deviations)
    when retraining on a new data set.
    
    Returns: nothing

    """
    AFSTool.data_stats_dict = dict() # clear the stats dict to allow training on new data
