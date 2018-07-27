
#------------------------------------------------------------------------------
# Automated feature selection script
#
# For help, run: python afs.py -h
#------------------------------------------------------------------------------

import argparse, logging, logging.config, os, sys, yaml
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

#------------------------------------------------------------------------------
# Define default arguments
#------------------------------------------------------------------------------

default_afs_home_path = os.path.dirname(os.path.realpath(sys.argv[0]))
default_python_package_directory_path = 'py_package_local_install/'
default_logging_config_path = 'logging.yaml'
default_afs_config_path = 'afs.yaml'
default_num_epochs = 5

#------------------------------------------------------------------------------
# Parse command line arguments
#------------------------------------------------------------------------------

args_yaml_path = os.path.join(default_afs_home_path, 'args.yaml')
arg_config = yaml.load(open(args_yaml_path, 'r'))

parser = argparse.ArgumentParser(arg_config.get('description', None))

for arg in arg_config.get('args', []):
    key_list = list(arg.keys())
    val_list = list(arg.values())
    if len(arg.items()) != 1 or 'flag' not in val_list[0]:
        raise IOError('Malformed args.yaml')
    name, config = key_list[0], val_list[0]
    flag, options = config['flag'], config.get('options', {})
    parser.add_argument('-{}'.format(flag), '--{}'.format(name), **options)

args = parser.parse_args()

#------------------------------------------------------------------------------
# Validate AFS home directory and logging configuration file
#------------------------------------------------------------------------------

afs_home_path = args.afs_home_path or default_afs_home_path

if not os.path.isdir(afs_home_path):
    raise IOError('AFS home directory at path %(path)s does not exit' % \
        {'path': afs_home_path})

logging_config_path = args.logging_config_path or \
    os.path.join(afs_home_path, default_logging_config_path)

if not os.path.isfile(logging_config_path):
    raise IOError(
    	'Logging configuration file at path %(path)s does not exit' % \
        {'path': logging_config_path})

#------------------------------------------------------------------------------
# Configure logger
#------------------------------------------------------------------------------

logging_config = yaml.load(open(logging_config_path, 'r'))
logging.config.dictConfig(logging_config)

logger = logging.getLogger(__name__)

# something funky going on with the logger module, until it's fixed just create
# the logger by hand

# logger.handlers = [logging.StreamHandler(sys.stdout)]
# logger.setLevel(logging.DEBUG)

#------------------------------------------------------------------------------
# Load source modules
#------------------------------------------------------------------------------

sys.path.append(afs_home_path)

from util import utils
from afs_gp.afsTool import AFSTool

#------------------------------------------------------------------------------
# Validate remaining arguments
#------------------------------------------------------------------------------

# Validate data file path

data_path = args.data_path

if not os.path.isfile(data_path):
    error_message = 'Data file at path %(path)s does not exist' % \
        {'path': data_path}
    logger.error(error_message)
    raise IOError(error_message)

# Validate test data file path

test_data_path = args.test_data_path or None

if not (test_data_path is None or os.path.isfile(test_data_path) ):
    error_message = 'Data file at path %(path)s does not exist' % \
        {'path': test_data_path}
    logger.error(error_message)
    raise IOError(error_message)

# Validate configuration path

afs_config_path = args.afs_config_path or \
    os.path.join(afs_home_path, default_afs_config_path)

afs_config = yaml.load(open(afs_config_path, 'r')) if \
	os.path.isfile(afs_config_path) else {}

# Validate python package install directory

python_package_directory_path = args.python_package_directory_path or \
    os.path.join(afs_home_path, default_python_package_directory_path)

if os.path.isdir(python_package_directory_path):
	sys.path.append(python_package_directory_path)

# Validate output file path

output_directory_path = args.output_directory_path
utils.make_sure_path_exists(output_directory_path)

if not os.access(output_directory_path, os.W_OK):
    error_message = \
    	'Output file write location at path %(path)s is inaccessible' % \
        {'path': output_directory_path}
    logger.error(error_message)
    raise IOError(error_message)

resume_file_prefix = args.resume_file_prefix or None

#------------------------------------------------------------------------------
# Configure timer
#------------------------------------------------------------------------------

my_timer = utils.Timer()
tic = lambda: my_timer.tic()
toc = lambda: my_timer.toc()

#------------------------------------------------------------------------------
# Create automated feature selection tool
#------------------------------------------------------------------------------

tic()

afs_tool = AFSTool(data_path
            , output_directory_path
            , config=afs_config
            , abt_test_path=test_data_path
            , resume_file_prefix=resume_file_prefix)

toc()

#------------------------------------------------------------------------------
# Run automated feature selection
#------------------------------------------------------------------------------

tic()

time_limit_minutes = afs_config.get('time_limit_minutes', None)
algorithm_timer = utils.Timer()
algorithm_timer.tic()

for epoch in range(afs_config.get('num_epochs', default_num_epochs)):

    afs_tool.run()
    afs_tool.hof_summary()
    if (afs_tool.bfs_is_converged and afs_tool.pop_is_converged):
        logger.info('Terminating because BFS is converged.')
        break
    total_time_minutes = algorithm_timer.toc(verbose=True).seconds / 60.0
    if time_limit_minutes and (total_time_minutes > time_limit_minutes):
        logger.info('Terminating because time limit has passed.')
        break

#------------------------------------------------------------------------------
# Evaluate
#------------------------------------------------------------------------------
if not afs_tool.abt_test_data is None:

    logger.debug('Running final evaluations')

    output_settings = afs_tool.output_settings
    save_output = output_settings['save_feature_list'] or output_settings['save_confusion_matrix'] or output_settings['save_roc'] or output_settings['save_crc']

    # Score the original feature set

    score_final = afs_tool.score_bfs_set(afs_tool.data.columns
        , classifier=afs_tool.validation_classifier
        , is_validation_run=True, save_output=save_output, output_settings=None, save_suffix='ORIGINAL')
    print('')
    print('')

    # Score the final BFS

    score_final = afs_tool.score_bfs_set(afs_tool.bfs
        , classifier=afs_tool.validation_classifier
        , is_validation_run=True, save_output=save_output, output_settings=None, save_suffix='BFS')
    print('')
    print('')

    # Score the Best Seen BFS

    score_final = afs_tool.score_bfs_set(afs_tool.best_seen_bfs
        , classifier=afs_tool.validation_classifier
        , is_validation_run=True, save_output=save_output, output_settings=None, save_suffix='BBFS')
    print('')
    print('')

    # Score the final HOF

    score_final = afs_tool.score_bfs_set(afs_tool.hof
        , classifier=afs_tool.validation_classifier
        , is_validation_run=True, save_output=save_output, output_settings=None, save_suffix='HOF')

    logger.info('Total Constructed Feature Evals = {}'.format(afs_tool.eval_counter))
    logger.info('Total Unique BFS Evals = {}'.format( len(afs_tool.feature_set_score_dict) ))