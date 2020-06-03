import numpy as np
from lightgbm import LGBMRegressor

model_configs = {
    'LGBM': {
        'model': LGBMRegressor,
        #model intilization 
        'initialize': {
            'boosting_type'     : 'gbdt',
            'n_jobs'            : -1,
        },

        #model search space
        'search_space': {
            'num_leaves'        : np.arange(2,8,1),
            'n_estimators'      : np.arange(30,50,2)
        },
    },  # LGBM
}
