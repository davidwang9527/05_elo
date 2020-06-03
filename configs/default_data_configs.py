import numpy as np
from lib.utility import get_elapsed_days

data_configs = {
    #This section is used for loading data, which usually should be changed for 
    #every project,thus it's NOT included in process_sequence.There must be 
    # 'fact_train' for training set and  'fact_test' for testing set.
    'input':{
        'fact_train'    : {'name': 'train.csv'},
        'fact_test'     : {'name': 'test.csv'},
    },

    'process_sequence':['fact_transform','x_y'],

    'fact_transform':{
        'action_sequence':['get_data','simple_impute','change_dtype','interaction_columns','drop_columns','result'],
        
        'get_data':[{'dict':'raw','key':'fact','how_to':'first_table'}],
        
        'simple_impute': {
            'first_active_month':{'missing_values':np.NaN,'strategy':'most_frequent'},
        },

        'change_dtype':
        {
            'first_active_month':{'type':'datetime','format':'%Y-%m'},
            'feature_1':'category',
            'feature_2':'category',
            'feature_3':'category',
        },

        'interaction_columns':
        [
            {'name':'elapsed_days','mode':'function','dtype':'int','a':'first_active_month','f':get_elapsed_days},
        ],

        'drop_columns':['first_active_month'],

        'result':[{'dict':'processed','key':'fact'}],
    },

    'x_y': {
        #get_data and result are mandantory in action_sequence. 
        'action_sequence':['get_data','factorize_columns','result'],
        'get_data':[
            {'dict':'processed','key':'fact','how_to':'first_table'},
         ],
        
        #all categorical and object colomns will be factorized except exclude_columns
        'factorize_columns':{
            'exclude_columns':[]
        },
        
        'result':[{'dict':'train_test','key':'x','exclude_columns':['target']},
                  {'dict':'train_test','key':'y','include_columns':['target']},
        ],
    },
}
