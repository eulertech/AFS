DATA_DIR = 'C:\\data\\AFS'
AFS_HOME = '.'
GLOBAL_DEBUG_LEVEL = 2
##################################
# Nice-to-have global definitions
##################################
import sys
from util import utils
my_timer = utils.Timer()

def debug_pr(x, level):
    utils.debug_print(x,level,GLOBAL_DEBUG_LEVEL)

def tic():
    my_timer.tic()

def toc(debug_level=2):
    if(GLOBAL_DEBUG_LEVEL >= debug_level):
        return my_timer.toc()