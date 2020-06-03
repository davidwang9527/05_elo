'''This model includes some configs which need modification for every project.'''
import os
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.model_selection import TimeSeriesSplit

main_table_pk = ['card_id']
'''
   Primary key columns of main table. When training a model on train data or predict on test data,
   the key columns should be removed. This  usually needs to be changed for every project.
'''

debug_nrows  = 50000
'''
    In debug mode, the final debug rows is the minimum of debug_rows and data rows.
'''

chunksize=1000000
'''
    Chunk size for loading from csv.
'''

file_dir_path = {
    'cache'   : './cache',
    'configs' : './configs',
    'doc'     : './doc',
    'input'   : './input',
    'log'     : './log',
    'model'   : './model',
    'params'  : './params',
    'output'  : './output'
}
'''
    File directories.
'''

data_refresh_configs = {
    'from_csv'        : {'level': 1, 'filename': None},
    'from_raw'        : {'level': 2, 'filename':'{prefix}_{datatype}_raw.pkl'},
    'from_train_test' : {'level': 3, 'filename':'{prefix}_{datatype}_final.pkl'},
}
'''
   When refresh_cache is set to False, data is loaded from train_test first(level=3),then
   data will be loaded from level 2 if there's no files in level 3, and so on.
   When refresh_cache is set to True, data will be reloaded from csv(level 1) to refresh 
   the cache from scratch.
'''

model_selection_object = {
    'KFold'                  : KFold,
    'StratifiedKFold'        : StratifiedKFold,
    'ShuffleSplit'           : ShuffleSplit,
    'StratifiedShuffleSplit' : StratifiedShuffleSplit,
    'TimeSeriesSplit'        : TimeSeriesSplit,
}
'''
    Model selection objects.
'''

filename_hpo_intermediate = '{loc}/skopt_{prefix}_{stem}_hyperparameters_iter{iter_num:04d}.pkl'
filename_hpo_result       = '{loc}/skopt_{prefix}_{stem}_hyperparameters_result.pkl'
filename_model_result     = '{loc}/skopt_{prefix}_{modelname}_result.pkl'
'''
    File names for hyperparameter optimization and model result.
'''

#create paths which haven't been there yet
for path in file_dir_path.values():
    if not os.path.exists(path):
        os.makedirs(path)
