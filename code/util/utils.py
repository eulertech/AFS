import os.path
import errno
from datetime import datetime
import logging
import re

logger = logging.getLogger('__main__')

def debug_print(x, message_level, debug_level):
    if(debug_level >= message_level):
        print(x)
class Timer(object):
    def __init__(self):
        super(self.__class__, self).__init__()
        self._timer_tic_time = datetime.now()

    def tic(self):
        self._timer_tic_time = datetime.now()

    def toc(self, verbose=True):
        elapsed_time = datetime.now() - self._timer_tic_time
        if(verbose):
            print('elapsed time: ' + str(elapsed_time))
        return elapsed_time

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def no_overwrite_filename(s, leading_zeros = 3):
    """
    Appends _000, _001, _002, etc. just before the file extension in order
    to produce a non-existing file name.
    Does a binary search for O(LOG(n)) speed
    """
    return new_filenames(s,leading_zeros)[0]


def new_filenames(s, leading_zeros = 3):
    """
    Appends _000, _001, _002, etc. just before the file extension in order
    to produce a non-existing file name.
    Does a binary search for O(LOG(n)) speed

    Returns a tuple (a, b) where
     a is the next unused file
     b is the latest existing file (None if a is file zero)
    """
    pathname, pathext = os.path.splitext(s)
    format_pattern = '%0'+str(leading_zeros)+'d'
    STEP_START = 8
    grow = True
    step=STEP_START
    tail = 0
    fname = pathname + '_' + (format_pattern % tail) + pathext

    if not os.path.exists(fname):
        return (fname, None)
    while step >= 1:
        if os.path.exists(fname):
            tail += step
        else:
            if grow:
                grow = False
                step /= 4
            tail -= step
        fname = pathname + '_' + (format_pattern % tail) + pathext
        if grow:
            step *= 2
        else:
            step /= 2
    while os.path.exists(fname):
        tail += 1
        fname = pathname + '_' + (format_pattern % tail) + pathext
    fname_existing = pathname + '_' + (format_pattern % (tail-1)) + pathext
    return (fname, fname_existing)

def string_replace_using_dict(str_, replacement_dict):
    # Using a regular expression to replace 'ARG0', 'ARG1', etc. with actual feature names
    pattern = re.compile(r'\b(' + '|'.join(replacement_dict.keys()) + r')\b')
    ret_str = pattern.sub(lambda x: replacement_dict[x.group()], str_)
    return ret_str

def source_decode(sourcecode):
    """Decode operator source and import operator class.
    Parameters
    ----------
    sourcecode: string
        a string of operator source (e.g 'sklearn.feature_selection.RFE')
    Returns
    -------
    import_str: string
        a string of operator class source (e.g. 'sklearn.feature_selection')
    op_str: string
        a string of operator class (e.g. 'RFE')
    op_obj: object
        operator class (e.g. RFE)
    """
    tmp_path = sourcecode.split('.')
    op_str = tmp_path.pop()
    import_str = '.'.join(tmp_path)
    try:
        exec('from {} import {}'.format(import_str, op_str))
        op_obj = eval(op_str)
    except ImportError:
        logger.error('Error: {} is not available for import.'.format(sourcecode))
        op_obj = None

    return import_str, op_str, op_obj
