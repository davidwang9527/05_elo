'''This module includes some universal functions'''
import logging
import datetime
import sys
import os
import importlib
import pickle
import re
import lib.config as config
from pathlib import Path

log_filename='{}/log{}.log'.format(config.file_dir_path['log'],datetime.date.today().strftime('%Y%m%d'))
file_handler = logging.FileHandler(filename=log_filename)
stdout_handler = logging.StreamHandler(sys.stdout)
logging.basicConfig(
		level='INFO', 
		format='[%(asctime)s] %(filename)s:%(lineno)d:%(levelname)s - %(message)s',
		handlers=[file_handler, stdout_handler])
logger = logging.getLogger(__name__)
'''
    logger which writes to both file and stdout
'''

def check_file_exist(filename):
    """Check whether a file exists in OS. 

    Parameters:
        filename(str):filename with absolute path or relative path

    Returns:
        bool:True if file exists.False if file doesn't exist

    """
    if not os.path.exists(filename):
        return False
    else:
        return True

def initialize_configs(filename):
    """Dynamically load python module from a file

    Dynamically load python module from a file and then execute the module. 

    Parameters:
        filename(str):filename with absolute path or relative path

    Returns:
        module : module which is loaded from filename  
    """

    if not filename:
        return None

    f1=Path(filename)
    if not f1.exists():
        raise ValueError("Spec file {spec_file} does not exist".format(spec_file=f1._str))
    spec = importlib.util.spec_from_file_location(name=f1.stem, location=f1)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def save_pickle(filename,obj):
    """Save an obj to a file as a pickle.

    If the file already exist, it'll be overwritten. 

    Parameters:
        filename(str):Filename with absolute path or relative path
        obj(object)  :The object to save.

    """
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
        logger.info('save obj to {}'.format(filename))
    return True

def load_pickle(filename):
    """Load an obj from a file.

    Parameters:
        filename(str) : Filename with absolute path or relative path

    Returns:
        object : The object to load. If the file doesn't exist, return None.

    """

    if not check_file_exist(filename):
        return None
    with open(filename, 'rb') as f:
        logger.info('load object from {}'.format(filename))
        return pickle.load(f)

def re_split(delimiters, string, maxsplit=0):
    """Split a string with multiple delimiters.

    Parameters:
        delimiters(list) : multiple delimiters
        string(str) : The string which will be delimited.
        maxsplit(int) : If maxsplit is nonzero, at most maxsplit splits occur, and the remainder of the string is returned as the final element of the list.

    Returns:
       list: resulting list after delimited.
    """
    regexPattern = '|'.join(map(re.escape, delimiters))
    return re.split(regexPattern, string, maxsplit)

def get_elapsed_days(first_active_month):
    """Get elapsed days from first activate month to 1 Feb,2018.
     
    This function is specially for Elo project.

    Parameters:
        first_active_month(date) : first_active_month

    Returns:
        timedelta : elapsed days from first activate month to 1 Feb,2018.
    """
    return (datetime.date(2018, 2, 1) - first_active_month.dt.date).dt.days
