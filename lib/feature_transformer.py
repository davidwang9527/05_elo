import numpy as np
import pandas as pd
from datetime import datetime
from pandas.api.types import is_categorical_dtype,is_object_dtype
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from lib.utility import logger

class _FlexibleLabelEncoder(LabelEncoder):
    """add unseen labels from test data to training classes for online training"""
    def transform(self,y):
        try:
            return super().transform(y)
        except ValueError as v:
            logger.info(str(v))
            diff = np.setdiff1d(y, self.classes_)
            self.classes_=np.append(self.classes_,diff)
            return super().transform(y)

class FeatureTransformer(TransformerMixin):
    """Class FeatureTransformer is built for feature engineering."""

    def __init__(self):
        #dict for LabelEncoder
        self.le_dict={}
        #dict for Imputer 
        self.imp_dict={}
        #dict for One-hot-encoder 
        self.ohe_dict={}
        #dict for clipping outliers
        self.clipping_dict={}
        #dict for normalization
        self.norm_dict={}
        #dict for pca
        self.pca_dict={}
        #dict for kbins
        self.kbins_dict={}

    @staticmethod
    def reduce_mem_usage(df, verbose=True):
        """Reduce memory usage for dataframes.

        Pandas is a memory monster because DataFrame tends to use 64bit integer or float. 
        Changing data types to smaller size can reduce memory usage dramatically. 

        Parameters:
            df(DataFrame): dateframe to reduce memory usage
            verbose(bool): set to True to show how much memory has been reduced. 
                        set to False to disable the output.

        Returns:
            DataFrame: The memory reduced DataFrame.
        """
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        start_mem = df.memory_usage(index=True,deep=True).sum() / 1024**2    
        for col in df.columns:
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min >= np.iinfo(np.int64).min and c_max <= np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)  
                else:
                    if c_min >= np.finfo(np.float16).min and c_max <= np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)    
        end_mem = df.memory_usage(index=True).sum() / 1024**2
        if verbose: 
            logger.info('Mem. usage decreased from {:5.2f}Mb to {:5.2f} Mb ({:.1f}% reduction)'.format(start_mem,end_mem, 100 * (start_mem - end_mem) / start_mem))
        return df

    @staticmethod
    def check_columns_exist(df, columns):
        """check whether a list of columns exist in df. 

        Parameters:
            df(DataFrame): dataframe
            columns(list): columns to be checked 

        Returns:
            tuple: (cols which exist in df,cols which don't exist in df)
        """
        cols_exist = [f for f in columns if f in df.columns]
        cols_not_exist = [f for f in columns if f not in df.columns]
        return cols_exist, cols_not_exist

    @staticmethod
    def get_categorical_columns(df):
        """Get categorical columns from df. 

        Parameters:
            df(DataFrame): dateframe

        Returns:
            list: categorical cols.
        """
        return [col for col in df.columns if is_categorical_dtype(df[col])]
 
    @staticmethod
    def get_object_columns(df):
        """Get object columns from df. 

        Parameters:
            df(DataFrame): dateframe

        Returns:
            list: object cols.
        """
        return [col for col in df.columns if is_object_dtype(df[col])]
 
    @staticmethod
    def any_empty_dataframe(data):
        """Check whether there're empty dataframes

        Parameters:
            data(list,dict): list or dict of DataFrame 

        Returns:
            bool: return True if there's any empty dataframe.return False if all dataframes are not empty. 
        """
        if not data:
            return True
        elif isinstance(data, dict):
            return any([v.empty for k, v in data.items()])
        elif isinstance(data, list):
            return any([l.empty for l in data])
        return False

    @staticmethod
    def process_drop_columns(df,drop_columns):
        """Drop a list of columns from dataframe

        This method only drop those columns which are in df 

        Parameters:
            df(DataFrame) : dataframe to be processed
            drop_columns(list):list of columns

        Returns:
            DataFrame: dataframe with the remaining columns

        Examples:
        
        >>> import pandas as pd
        >>> import numpy as np
        >>> import logging
        >>> from lib.feature_transformer import FeatureTransformer
        >>> logging.disable()
        >>> df = pd.DataFrame(np.arange(12).reshape(3,4),columns=['A', 'B', 'C', 'D'])
        >>> df
           A  B   C   D
        0  0  1   2   3
        1  4  5   6   7
        2  8  9  10  11
        >>> FeatureTransformer.process_drop_columns(df,drop_columns=['B', 'C'])
           A   D
        0  0   3
        1  4   7
        2  8  11
        
        """

        logger.info("dropping columns......")
        cols_exist,cols_not_exist=FeatureTransformer.check_columns_exist(df, drop_columns)
        df_tmp=df.drop(cols_exist, axis='columns')
        logger.info('{} column(s) have been dropped:{}'.format(len(cols_exist),cols_exist))
        if cols_not_exist:
            logger.warning('{} column(s) do(es) NOT exist:{}'.format(len(cols_not_exist),cols_not_exist))
        return df_tmp

    @staticmethod
    def process_select_columns(df,select_columns):
        """Select a list of columns from dataframe

        This method only select those columns which are in df.

        Parameters:
            df(DataFrame) : dataframe to be processed
            select_columns(list):list of columns

        Returns:
            DataFrame: dataframe with the select columns

        """
        logger.info("selecting columns......")
        cols_exist,cols_not_exist=FeatureTransformer.check_columns_exist(df, select_columns)
        df_tmp=df[cols_exist]
        logger.info('{} column(s) have been select:{}'.format(len(cols_exist),cols_exist))
        if cols_not_exist:
            logger.warning('{} column(s) do(es) NOT exist:{}'.format(len(cols_not_exist),cols_not_exist))
        return df_tmp

    @staticmethod
    def process_drop_rows(df, action_configs):
        """drop rows which are too abnormal to be trained

        Rows matching any of action_configs conditions will be deleted.  

        Parameters:
            df(DataFrame) : dataframe to be processed
            action_configs(dict):keys are column names, and values are list of values for each key.   

        Returns:
            DataFrame: dataframe with remaining rows

        Examples:

        >>> import pandas as pd
        >>> import numpy as np
        >>> import logging
        >>> from lib.feature_transformer import FeatureTransformer
        >>> logging.disable()
        >>> df = pd.DataFrame(np.arange(24).reshape(-1,4),columns=['A', 'B', 'C', 'D'])
        >>> df
            A   B   C   D
        0   0   1   2   3
        1   4   5   6   7
        2   8   9  10  11
        3  12  13  14  15
        4  16  17  18  19
        5  20  21  22  23
        >>> FeatureTransformer.process_drop_rows(df,action_configs={'A':[0,4],'B':[5,9]})
            A   B   C   D
        3  12  13  14  15
        4  16  17  18  19
        5  20  21  22  23

        """
        logger.info("dropping rows......")
        columns = sorted(list(action_configs.keys()))
        cols_exist, cols_not_exist = FeatureTransformer.check_columns_exist(df, columns)
        configs = {k: v for k, v in action_configs.items() if k in cols_exist}
        inds = df[cols_exist].isin(configs)
        inds_sel = inds.any(axis=1)

        for f, series in inds.iteritems():
            logger.info("remove {} rows where column {} is any of {}".format(series.sum(),f,action_configs[f]))

        logger.info("remove {} rows from {} rows".format(inds_sel.astype(int).sum(), inds_sel.shape[0]))
        if cols_not_exist:
            logger.warning("missing {} columns: {}".format(len(cols_not_exist), cols_not_exist))
        return df.loc[~inds_sel]

    @staticmethod
    def process_replace_values(df, action_configs):
        """Replace values 

        This is usually used to replace two values(e.g. Y/N) to numbers(e.g. 1/0)

        Parameters:
            df(DataFrame) : dataframe to be processed
            action_configs(dict):keys are column names, and values are dicts in which keys are orignal values and values are replaced values for each key.   

        Returns:
            DataFrame: dataframe with remaining rows

        Examples:

        >>> import pandas as pd
        >>> import numpy as np
        >>> import logging
        >>> from lib.feature_transformer import FeatureTransformer
        >>> logging.disable()
        >>> df = pd.DataFrame(np.arange(24).reshape(-1,4),columns=['A', 'B', 'C', 'D'])
        >>> df
            A   B   C   D
        0   0   1   2   3
        1   4   5   6   7
        2   8   9  10  11
        3  12  13  14  15
        4  16  17  18  19
        5  20  21  22  23
        >>> FeatureTransformer.process_replace_values(df,action_configs={'A':{0:30,4:40},'B':{1:100,5:500}})
            A    B   C   D
        0  30  100   2   3
        1  40  500   6   7
        2   8    9  10  11
        3  12   13  14  15
        4  16   17  18  19
        5  20   21  22  23

        """        

        logger.info("replacing values......")
        columns = sorted(list(action_configs.keys()))
        cols_exist, cols_not_exist = FeatureTransformer.check_columns_exist(df, columns)

        configs = {k: v for k, v in action_configs.items() if k in cols_exist}
        df=df.replace(configs)

        for k, v in configs.items():
            logger.info("replace {} using {}".format(k, v))
        if cols_not_exist:
            logger.warning("{} column(s) do(es) NOT exist: {}".format(len(cols_not_exist), cols_not_exist))
        return df
    
    def process_simple_impute(self,df,process_key,action_key,action_configs,datatype):
        """Simple impute for both training data and testing data. 

        This is for imputing missing values. Though process_replace_values could impute np.NAN values with constant values, 
        this method is more advanced to impute missing values with simple functions on the related columns such as mean,median,most_frequent 
        as well as constant values.Thus process_replace_values are not recommended for imputing np.NAN value.

        Parameters:
            df(DataFrame) : dataframe to be processed
            process_key(str) : process_key in data config files which are values of process_sequence and keys of process dicts.
            action_key(str) : action_key in data config files which are values of action_sequence and keys of action dicts.
            action_configs(dict):keys are column names, and values are dicts in which keys are 'missing_values','strategy' and 'fill_value' and 
                                 the values are used as values of SimpleImputer

            datatype(str) : 'train' for training data, 'test' for testing data.
        
        Returns:
            DataFrame: processed dataframe

        Examples:

        >>> import pandas as pd
        >>> import numpy as np
        >>> import logging
        >>> from lib.feature_transformer import FeatureTransformer
        >>> logging.disable()
        >>> df_train = pd.DataFrame(np.arange(24).reshape(-1,4),columns=['A', 'B', 'C', 'D'])
        >>> df_train
            A   B   C   D
        0   0   1   2   3
        1   4   5   6   7
        2   8   9  10  11
        3  12  13  14  15
        4  16  17  18  19
        5  20  21  22  23
        >>> df_train.loc[0,'A']=np.nan
        >>> df_train.loc[1,'B']=np.nan
        >>> df_train
              A     B   C   D
        0   NaN   1.0   2   3
        1   4.0   NaN   6   7
        2   8.0   9.0  10  11
        3  12.0  13.0  14  15
        4  16.0  17.0  18  19
        5  20.0  21.0  22  23
        >>> ft=FeatureTransformer()
        >>> action_configs={
        ...    'A':{'missing_values':np.NaN,'strategy':'mean','fill_value':''},
        ...    'B':{'missing_values':np.NAN,'strategy':'constant','fill_value':100},
        ... }
        >>> ft.process_simple_impute(df_train,'AAA','BBB',action_configs,'train')
              A      B   C   D
        0  12.0    1.0   2   3
        1   4.0  100.0   6   7
        2   8.0    9.0  10  11
        3  12.0   13.0  14  15
        4  16.0   17.0  18  19
        5  20.0   21.0  22  23
        >>> df_test = pd.DataFrame(np.arange(36).reshape(-1,4),columns=['A', 'B', 'C', 'D'])
        >>> df_test
            A   B   C   D
        0   0   1   2   3
        1   4   5   6   7
        2   8   9  10  11
        3  12  13  14  15
        4  16  17  18  19
        5  20  21  22  23
        6  24  25  26  27
        7  28  29  30  31
        8  32  33  34  35
        >>> df_test.loc[2,'A']=np.nan
        >>> df_test.loc[3,'B']=np.nan
        >>> df_test
              A     B   C   D
        0   0.0   1.0   2   3
        1   4.0   5.0   6   7
        2   NaN   9.0  10  11
        3  12.0   NaN  14  15
        4  16.0  17.0  18  19
        5  20.0  21.0  22  23
        6  24.0  25.0  26  27
        7  28.0  29.0  30  31
        8  32.0  33.0  34  35
        >>> ft.process_simple_impute(df_test,'AAA','BBB',action_configs,'test')
              A      B   C   D
        0   0.0    1.0   2   3
        1   4.0    5.0   6   7
        2  12.0    9.0  10  11
        3  12.0  100.0  14  15
        4  16.0   17.0  18  19
        5  20.0   21.0  22  23
        6  24.0   25.0  26  27
        7  28.0   29.0  30  31
        8  32.0   33.0  34  35
		
        """
        logger.info("Processing simple_impute......")

        columns = sorted(list(action_configs.keys()))
        cols_exist, cols_not_exist = FeatureTransformer.check_columns_exist(df, columns)

        configs = {k: v for k, v in action_configs.items() if k in cols_exist}

        for k, v in configs.items():
            map_key='{}_{}_{}'.format(process_key,action_key,k)
            logger.info("impute column {} using {}".format(k, v))
            if datatype=='train':
                imp = SimpleImputer(missing_values=v.get('missing_values',np.nan),strategy=v.get('strategy','mean'),fill_value=v.get('fill_value'))
                df[k]=imp.fit_transform(df[[k]])
                self.imp_dict[map_key]=imp
            else:
                imp=self.imp_dict.get(map_key)
                if None==imp:
                    logger.warn("Ouch! There is no SimpleImputer() for {}".format(map_key))
                    exit(0)
                df[k]=imp.transform(df[[k]])
        
        if cols_not_exist:
            logger.warning("{} column(s) do(es) NOT exist: {}".format(len(cols_not_exist), cols_not_exist))

        return df

    @staticmethod
    def process_change_dtype(df, action_configs):
        """This is for changing dtype of DataFrame columns including changing date string to datetime. 

        Parameters:
            df(DataFrame) : dataframe to be processed
            action_configs(dict) : keys are column names, and values are either strings or dicts. 
                                When the value is strings, it indicates a data type, e.g. 'int'.
                                When the value is a dict, the key is either 'datetime' or 'unix'.
                                When the key is 'datetime', there's another key 'format' for date format.
                                When the key is 'unix', there's another key 'unit' which is D,s,ms,us or ns. 

        Returns:
            DataFrame: processed dataframe

        Examples:

        >>> import pandas as pd
        >>> import numpy as np
        >>> import logging
        >>> from lib.feature_transformer import FeatureTransformer
        >>> logging.disable()
        >>> df = pd.DataFrame(np.arange(24).reshape(-1,4),columns=['A', 'B', 'C', 'D'])
        >>> df['E']='2011-11-03 11:07:04'
        >>> df['F']='1549700152'
        >>> df
            A   B   C   D                    E           F
        0   0   1   2   3  2011-11-03 11:07:04  1549700152
        1   4   5   6   7  2011-11-03 11:07:04  1549700152
        2   8   9  10  11  2011-11-03 11:07:04  1549700152
        3  12  13  14  15  2011-11-03 11:07:04  1549700152
        4  16  17  18  19  2011-11-03 11:07:04  1549700152
        5  20  21  22  23  2011-11-03 11:07:04  1549700152
        >>> df.dtypes
        A     int32
        B     int32
        C     int32
        D     int32
        E    object
        F    object
        dtype: object
        >>> action_configs={
        ...    'A':'float',
        ...    'E':{'type':'datetime','format':'%Y-%m-%d %H:%M:%S'},
        ...    'F':{'type':'epoch','unit':'s'},
        ... }
        >>> FeatureTransformer.process_change_dtype(df,action_configs)
              A   B   C   D                   E                   F
        0   0.0   1   2   3 2011-11-03 11:07:04 2019-02-09 08:15:52
        1   4.0   5   6   7 2011-11-03 11:07:04 2019-02-09 08:15:52
        2   8.0   9  10  11 2011-11-03 11:07:04 2019-02-09 08:15:52
        3  12.0  13  14  15 2011-11-03 11:07:04 2019-02-09 08:15:52
        4  16.0  17  18  19 2011-11-03 11:07:04 2019-02-09 08:15:52
        5  20.0  21  22  23 2011-11-03 11:07:04 2019-02-09 08:15:52
        >>> df.dtypes
        A           float64
        B             int32
        C             int32
        D             int32
        E    datetime64[ns]
        F    datetime64[ns]
        dtype: object

        """

        logger.info("changing dtypes......")
        columns = sorted(list(action_configs.keys()))
        cols_exist, cols_not_exist = FeatureTransformer.check_columns_exist(df, columns)
        configs = {k: v for k, v in action_configs.items() if k in cols_exist}
        for k,v in configs.items():
            if isinstance(v,dict):
                t=v.get('type','datetime')
                if t=='datetime':
                    df[k]=pd.to_datetime(df[k],errors='ignore',format=v.get('format',None))
                elif t=='epoch':
                    df[k]=pd.to_datetime(df[k],errors='ignore',unit=v.get('unit','s'))
            else:
                df[k]=df[k].astype(v)
        
        for k, v in configs.items():
            logger.info("change {} to dtype {}".format(k, v))
        if cols_not_exist:
            logger.warning("{} column(s) do(es) NOT exist: {}".format(len(cols_not_exist), cols_not_exist))
        return df

    @staticmethod
    def process_interaction(df, action_configs):
        """Feature interaction.

        This method performs interaction between two columns. It also derives date part from 'datetime' and could derive
        more complicated interactions from functions. 

        Parameters:
            df(DataFrame) : dataframe to be processed
            action_configs(list) : Items in action_configs are dicts.In the dict,
                                key 'name' indicate new column names.
                                key 'mode' indicates interactino method including 'add','subtract','subtract_positive','multiply','divide','datetime' and 'function'.
                                key 'dtype' indicates dtype for the new column.
                                key 'a' indicates the first column.
                                key 'b' indicates the second column.
                                when 'mode' is 'datetime', 'f' indicates part of the datetime, including 'year','month','day','hour','weekday'.
                                when 'mode' is 'function', 'f' indicates the function for interaction.  

        Returns:
            DataFrame: processed dataframe

        Examples:
        
        >>> import pandas as pd
        >>> import numpy as np
        >>> import logging
        >>> from lib.feature_transformer import FeatureTransformer
        >>> logging.disable()
        >>> df = pd.DataFrame(np.arange(-10,14).reshape(-1,4),columns=['A', 'B', 'C', 'D'])
        >>> df['E']='2011-11-03 11:07:04'
        >>> action_configs={
        ...    'E':{'type':'datetime','format':'%Y-%m-%d %H:%M:%S'},
        ... }
        >>> FeatureTransformer.process_change_dtype(df,action_configs)
            A   B   C   D                   E
        0 -10  -9  -8  -7 2011-11-03 11:07:04
        1  -6  -5  -4  -3 2011-11-03 11:07:04
        2  -2  -1   0   1 2011-11-03 11:07:04
        3   2   3   4   5 2011-11-03 11:07:04
        4   6   7   8   9 2011-11-03 11:07:04
        5  10  11  12  13 2011-11-03 11:07:04
        >>> action_configs=[
        ...    {'name':'F','mode':'add','dtype':'int','a':'A','b':'B'},
        ...    {'name':'G','mode':'datetime','dtype':'int','a':'E','f':'year'},
        ...    {'name':'H','mode':'function','dtype':'int','a':'A','b':'B','f':np.add},
        ...    {'name':'I','mode':'function','dtype':'int','a':'A','f':np.abs},
        ... ]
        >>> FeatureTransformer.process_interaction(df,action_configs)
            A   B   C   D                   E   F     G   H   I
        0 -10  -9  -8  -7 2011-11-03 11:07:04 -19  2011 -19  10
        1  -6  -5  -4  -3 2011-11-03 11:07:04 -11  2011 -11   6
        2  -2  -1   0   1 2011-11-03 11:07:04  -3  2011  -3   2
        3   2   3   4   5 2011-11-03 11:07:04   5  2011   5   2
        4   6   7   8   9 2011-11-03 11:07:04  13  2011  13   6
        5  10  11  12  13 2011-11-03 11:07:04  21  2011  21  10
        """
 
        logger.info("Processing Interactions......")

        new_columns = []
        for v in action_configs:
            k = v['name']
            logger.info("process {}".format(k))

            interaction = v.get('mode', None)
            #check feature columns
            check_cols = [vv for kk, vv in v.items() if kk in ['a', 'b']]
            _, cols_not_exist = FeatureTransformer.check_columns_exist(df, check_cols)
            if cols_not_exist:
                logger.warning("{} column(s) do(es) NOT exist: {}".format(len(cols_not_exist), cols_not_exist))
                continue

            # process
            if 'add' == interaction:
                df[k] = df[v['a']] + df[v['b']]
            elif 'subtract' == interaction:
                df[k] = df[v['a']] - df[v['b']]
            elif 'subtract_positive' == interaction:
                df[k] = (df[v['a']] - df[v['b']]).apply(lambda x: x if x > 0 else 0)
            elif 'multiply' == interaction:
                df[k] = df[v['a']] * df[v['b']]
            elif 'divide' == interaction:
                df[k] = df[v['a']] / df[v['b']].apply(lambda x: x if x != 0 else 1.0)
            elif 'datetime' == interaction:
                formated_date = df[v['a']]
                if v['f']=='year':
                    df[k]=formated_date.apply(lambda x:x.year).astype(v['dtype'])
                elif v['f']=='month':
                    df[k]=formated_date.apply(lambda x:x.month).astype(v['dtype'])
                elif v['f']=='day':
                    df[k]=formated_date.apply(lambda x:x.day).astype(v['dtype'])
                elif v['f']=='hour':
                    df[k]=formated_date.apply(lambda x:x.hour).astype(v['dtype'])
                elif v['f']=='weekday':
                    df[k]=formated_date.apply(lambda x:x.isoweekday()).astype(v['dtype'])
                else:
                    logger.error('Only support year/month/day/weekday/hour at the moment for interacting datetime')
            elif 'function'==interaction:
                if v.get('b'):
                    df[k]= (v['f'](df[v['a']],df[v['b']])).astype(v['dtype'])
                else:
                    df[k]= (v['f'](df[v['a']])).astype(v['dtype'])
            new_columns.append(k)
        return df
    
    @staticmethod
    def process_aggregation(df, action_configs):
        """Aggregation on real-value columns.
        
        If categorical columns are not in main table, they usually should
        be converted to one hot encode.Then all the columns of this table could 
        be regarded as real-value columns.

        Parameters:
            df(DataFrame) : dataframe to be processed
            action_configs(list) : Items in action_configs are dicts.In the dict,
                                key 'groupby' indicate a list of group by column names.
                                key 'count' indicates whether count on this column.
                                key 'metrics' indicates the columns to aggregate on and related list of aggregation functions. 

        Returns:
            dict: aggregation results.The names of derived aggregation columns are generated automatically based on groupby, k and value in 
            metrics,concated by underscore('_').The format is groupby name+'_'+metrics key+'_' + metrics value.
            e.g. month_totals.pageviews_mean.For count,the derived column name is groupby name+'_'+'count',e.g.month_count;
            For percent,the derived column name is groupby name+'_'+'percent'.If the groupby is a list of several columns, 
            then the groupby name is concated by underscore('_')  .The aggregations results with same groupby columns are merged 
            into one dataframe, aggregations with different groupby columns are stored in the result dict and the key names 
            are groupby names if the groupby is one column, or the groupby name concated by underscore('_') if the groupby 
            is a list. 

        Examples:

        >>> import pandas as pd
        >>> import numpy as np
        >>> import logging
        >>> from lib.feature_transformer import FeatureTransformer
        >>> logging.disable()
        >>> df = pd.DataFrame(np.arange(24).reshape(-1,4),columns=['A', 'B', 'C', 'D'])
        >>> df['G']='A'
        >>> df.loc[3:,'G']='B'
        >>> action_configs=[
        ...     {
        ...         'groupby': ['G'],
        ...         'count' : True,
        ...         'metrics' : {
        ...                 'A': ['sum', 'mean'],
        ...                 'B': ['max'],
        ...            },
        ...     },
        ... ]
        >>> agg=FeatureTransformer.process_aggregation(df,action_configs)
        >>> agg
        {'G':    G  G_A_sum  G_A_mean  G_B_max  G_count
        0  A       12         4        9        3
        1  B       48        16       21        3}
        >>> agg['G']
           G  G_A_sum  G_A_mean  G_B_max  G_count
        0  A       12         4        9        3
        1  B       48        16       21        3

        """
    
        ret=list()
        for ac in action_configs:
            groupby_cols = ac.get('groupby', [])
            groupby_cols, cols_not_exist = FeatureTransformer.check_columns_exist(df, groupby_cols)

            if cols_not_exist:
                logger.warning("{} column(s) do(es) NOT exist: {}".format(len(cols_not_exist), cols_not_exist))

            if not groupby_cols:
                logger.warning("aggregate column {} does not exist".format(groupby_cols))
                continue
            else:
                logger.info("aggregate on {}".format(groupby_cols))

            aggregations = {}

            metrics_dict = ac.get('metrics', {})
            df_agg=pd.DataFrame()
            
            if metrics_dict:
                aggregations = {k:list(v) for k, v in metrics_dict.items() if k in df.columns and v}
                if aggregations:
                    for k, v in aggregations.items():
                        logger.info("aggregate {} ({}) with {}".format(k, df[k].dtype, v))
                    df_agg = df.groupby(groupby_cols).agg({**aggregations})
                    df_agg.index.name='_'.join(groupby_cols)
                    df_agg.columns = pd.Index(['{}_{}_{}'.format(df_agg.index.name,e[0], e[1]) for e in df_agg.columns.tolist()])
            
            if df_agg.empty: 
                df_agg.index.name='_'.join(groupby_cols)

            col_name_cnt='{}_count'.format('_'.join(groupby_cols))
            if ac.get('count', False):
                    logger.info("aggregate count on {}".format(groupby_cols))
                    df_agg[col_name_cnt] = df.groupby(groupby_cols).size()
            
            if ac.get('percent',False):
                col_name_percent='{}_percent'.format('_'.join(groupby_cols))
                logger.info("aggregate percent on {}".format(groupby_cols))
                if ac.get('count', False)==False:
                    df_agg[col_name_cnt] = df.groupby(groupby_cols).size()
                    df_agg[col_name_percent] = df_agg[col_name_cnt]/df_agg[col_name_cnt].sum()
                    df_agg=df_agg.drop(columns=col_name_cnt)
                else:
                    df_agg[col_name_percent] = df_agg[col_name_cnt]/df_agg[col_name_cnt].sum()
            ret.append(df_agg)

        ret = [r for r in ret if not r.empty]
        inds = sorted(list(set([r.index.name for r in ret])))
        ret = {ind: pd.concat([r for r in ret if r.index.name == ind], axis=1, join='outer') for ind in inds}
        for k, v in ret.items():
            logger.info("Aggregating Result on {}: {}".format(k, v.shape))
            ret[k]=v.reset_index()
            
        return ret
    
    def process_factorize(self,df,process_key,action_key,action_configs,datatype):
        """process factorize(label encoding).

        This method process label encoding. It also add unseen labels from test data to training classes.
        
        Parameters:
            df(DataFrame) : dataframe to be processed
            process_key(str) : process_key in data config files which are values of process_sequence and keys of process dicts.
            action_key(str) : action_key in data config files which are values of action_sequence and keys of action dicts.
            datatype(str) : 'train' for training data, 'test' for testing data.
            action_configs(dict) : key 'exclude_columns' indicate a list of columns.
            will be factorized except exclude_columns.

        Returns:
            DataFrame: processed dataframe. 

        Examples:

        >>> import pandas as pd
        >>> import numpy as np
        >>> import logging
        >>> from lib.feature_transformer import FeatureTransformer
        >>> logging.disable()
        >>> ft=FeatureTransformer()
        >>> df_train = pd.DataFrame(np.arange(24).reshape(-1,4),columns=['A', 'B', 'C', 'D'])
        >>> df_train['G']='A'
        >>> df_train.loc[3:,'G']='B'
        >>> df_train['G']=df_train['G'].astype('category')
        >>> df_train
            A   B   C   D  G
        0   0   1   2   3  A
        1   4   5   6   7  A
        2   8   9  10  11  A
        3  12  13  14  15  B
        4  16  17  18  19  B
        5  20  21  22  23  B
        >>> df_test = pd.DataFrame(np.arange(36).reshape(-1,4),columns=['A', 'B', 'C', 'D'])
        >>> df_test['G']='A'
        >>> df_test.loc[5:,'G']='B'
        >>> df_test['G']=df_test['G'].astype('category')
        >>> df_test
            A   B   C   D  G
        0   0   1   2   3  A
        1   4   5   6   7  A
        2   8   9  10  11  A
        3  12  13  14  15  A
        4  16  17  18  19  A
        5  20  21  22  23  B
        6  24  25  26  27  B
        7  28  29  30  31  B
        8  32  33  34  35  B
        >>> action_configs={'exclude_columns':['fullVisitorId','visitId']}
        >>> ft.process_factorize(df_train,'AAA','BBB',action_configs,'train')
            A   B   C   D  G
        0   0   1   2   3  0
        1   4   5   6   7  0
        2   8   9  10  11  0
        3  12  13  14  15  1
        4  16  17  18  19  1
        5  20  21  22  23  1
        >>> ft.process_factorize(df_test,'AAA','BBB',action_configs,'test')
            A   B   C   D  G
        0   0   1   2   3  0
        1   4   5   6   7  0
        2   8   9  10  11  0
        3  12  13  14  15  0
        4  16  17  18  19  0
        5  20  21  22  23  1
        6  24  25  26  27  1
        7  28  29  30  31  1
        8  32  33  34  35  1
		
        """

        logger.info("Processing label encoding......")
        exclude_cols=action_configs.get('exclude_columns',[])
        cols=FeatureTransformer.get_categorical_columns(df)
        cols=[c for c in cols if c not in exclude_cols]
        
        cols_exist, cols_not_exist = FeatureTransformer.check_columns_exist(df, cols)

        for bin_feature in cols_exist:
            map_key='{}_{}_{}'.format(process_key,action_key,bin_feature)
            logger.info('factorizing column {}'.format(bin_feature))
            if datatype=='train':
                le = _FlexibleLabelEncoder()
                df[bin_feature]=le.fit_transform(df[bin_feature])
                df[bin_feature]=df[bin_feature].astype('category')
                self.le_dict[map_key]=le
            else:
                le=self.le_dict.get(map_key)
                if None==le:
                    logger.warn("Ouch! There is no labelEncoder() for {}".format(map_key))
                    exit(0)
                df[bin_feature]=le.transform(df[bin_feature])
                df[bin_feature]=df[bin_feature].astype('category')
            
            logger.info("factorize {} items for {}".format(len(le.classes_), bin_feature))

        for k in cols_not_exist:
            logger.warning("missing {}".format(k))
        return df

    def process_one_hot_encoder(self,df,process_key,action_key,datatype):
        """process one-hot-encoder.

        This method process one-hot-encoder. An unknown category is encountered during transform, the resulting 
        one-hot encoded columns for this feature will be all zeros.
        
        Parameters:
            df(DataFrame) : dataframe to be processed
            process_key(str) : process_key in data config files which are values of process_sequence and keys of process dicts.
            action_key(str) : action_key in data config files which are values of action_sequence and keys of action dicts.
            datatype(str) : 'train' for training data, 'test' for testing data.

        Returns:
            DataFrame: processed dataframe. 

        Examples:

        >>> import pandas as pd
        >>> import numpy as np
        >>> import logging
        >>> from lib.feature_transformer import FeatureTransformer
        >>> logging.disable()
        >>> ft=FeatureTransformer()
        >>> df_train = pd.DataFrame(np.arange(24).reshape(-1,4),columns=['A', 'B', 'C', 'D'])
        >>> df_train['G']='A'
        >>> df_train.loc[3:,'G']='B'
        >>> df_train['G']=df_train['G'].astype('category')
        >>> df_train
            A   B   C   D  G
        0   0   1   2   3  A
        1   4   5   6   7  A
        2   8   9  10  11  A
        3  12  13  14  15  B
        4  16  17  18  19  B
        5  20  21  22  23  B
        >>> df_test = pd.DataFrame(np.arange(36).reshape(-1,4),columns=['A', 'B', 'C', 'D'])
        >>> df_test['G']='A'
        >>> df_test.loc[5:,'G']='B'
        >>> df_test['G']=df_test['G'].astype('category')
        >>> df_test
            A   B   C   D  G
        0   0   1   2   3  A
        1   4   5   6   7  A
        2   8   9  10  11  A
        3  12  13  14  15  A
        4  16  17  18  19  A
        5  20  21  22  23  B
        6  24  25  26  27  B
        7  28  29  30  31  B
        8  32  33  34  35  B
        >>> d1=ft.process_one_hot_encoder(df=df_train,process_key='AAA',action_key='BBB',datatype='train')
        >>> d1
            A   B   C   D  x0_A  x0_B
        0   0   1   2   3   1.0   0.0
        1   4   5   6   7   1.0   0.0
        2   8   9  10  11   1.0   0.0
        3  12  13  14  15   0.0   1.0
        4  16  17  18  19   0.0   1.0
        5  20  21  22  23   0.0   1.0
        >>> d2=ft.process_one_hot_encoder(df=df_test,process_key='AAA',action_key='BBB',datatype='train')
        >>> d2
            A   B   C   D  x0_A  x0_B
        0   0   1   2   3   1.0   0.0
        1   4   5   6   7   1.0   0.0
        2   8   9  10  11   1.0   0.0
        3  12  13  14  15   1.0   0.0
        4  16  17  18  19   1.0   0.0
        5  20  21  22  23   0.0   1.0
        6  24  25  26  27   0.0   1.0
        7  28  29  30  31   0.0   1.0
        8  32  33  34  35   0.0   1.0

        """
        logger.info("Processing One Hot Encoding......")

        categorical_columns = FeatureTransformer.get_categorical_columns(df)

        logger.info("identify {} categorical columns: {}".format(len(categorical_columns), categorical_columns))

        df_cat=df[categorical_columns].copy()
        df.drop(columns=categorical_columns,inplace=True)

        map_key='{}_{}_ohe'.format(process_key,action_key)

        if datatype=='train':
            ohe = OneHotEncoder(handle_unknown='ignore',sparse=False)
            df_cat=ohe.fit_transform(df_cat)
            self.ohe_dict[map_key]=ohe
        else:
            ohe=self.ohe_dict.get(map_key)
            if None==ohe:
                logger.warn("Ouch! There is no OneHotEncoder() for {}".format(map_key))
                exit(0)
            df_cat=ohe.transform(df_cat)
        df_cat=pd.DataFrame(df_cat,columns=ohe.get_feature_names())

        df=pd.concat([df,df_cat],axis=1)

        df=FeatureTransformer.reduce_mem_usage(df)
        
        return df

    def process_clip_outliers(self,df,process_key,action_key,action_configs,datatype):
        '''This method is for clipping outliers.

        This method clipping outliers to lower bound=q[0]-(q[1]-q[0])*1.5,upper bound=q[1]+(q[1]-q[0])*1.5,
        where q[0] means fist quantile, and1[1] means third quantile.
        
        Parameters:
            df(DataFrame) : dataframe to be processed
            process_key(str) : process_key in data config files which are values of process_sequence and keys of process dicts.
            action_key(str) : action_key in data config files which are values of action_sequence and keys of action dicts.
            datatype(str) : 'train' for training data, 'test' for testing data.
            action_configs(dict) : key 'include_columns' indicate a list of columns will be clipped.

        Returns:
            DataFrame: processed dataframe. 

        Examples:

        >>> import pandas as pd
        >>> import numpy as np
        >>> import logging
        >>> from lib.feature_transformer import FeatureTransformer
        >>> logging.disable()
        >>> df_train = pd.DataFrame(np.arange(12).reshape(-1,3),columns=['A', 'B','C'])
        >>> df_train
           A   B   C
        0  0   1   2
        1  3   4   5
        2  6   7   8
        3  9  10  11
        >>> ft=FeatureTransformer()
        >>> action_configs={
        ...    'include_columns':['A','B']
        ... }
        >>> ft.process_clip_outliers(df_train,'AAA','BBB',action_configs,'train')
             A     B   C
        0  0.0   1.0   2
        1  3.0   4.0   5
        2  6.0   7.0   8
        3  9.0  10.0  11
        >>> ft.clipping_dict
        {'AAA_BBB_A': (-4.5, 13.5), 'AAA_BBB_B': (-3.5, 14.5)}
        >>> df_test = pd.DataFrame(np.arange(36).reshape(-1,3),columns=['A', 'B', 'C'])
        >>> df_test
             A   B   C
        0    0   1   2
        1    3   4   5
        2    6   7   8
        3    9  10  11
        4   12  13  14
        5   15  16  17
        6   18  19  20
        7   21  22  23
        8   24  25  26
        9   27  28  29
        10  30  31  32
        11  33  34  35
        >>> ft.process_clip_outliers(df_test,'AAA','BBB',action_configs,'test')
               A     B   C
        0    0.0   1.0   2
        1    3.0   4.0   5
        2    6.0   7.0   8
        3    9.0  10.0  11
        4   12.0  13.0  14
        5   13.5  14.5  17
        6   13.5  14.5  20
        7   13.5  14.5  23
        8   13.5  14.5  26
        9   13.5  14.5  29
        10  13.5  14.5  32
        11  13.5  14.5  35
        '''

        logger.info("Processing clip outliers......")
        include_columns=action_configs.get('include_columns')
        cols_exist, cols_not_exist = FeatureTransformer.check_columns_exist(df, include_columns)
        if cols_not_exist:
            logger.warning('{} column(s) do(es) NOT exist:{}'.format(len(cols_not_exist),cols_not_exist))

        for colname in cols_exist:
            map_key='_'.join([process_key,action_key,colname])
            if datatype=='train':
                q=np.nanpercentile(df[colname],(25.0,75.0))
                l=q[0]-(q[1]-q[0])*1.5
                u=q[1]+(q[1]-q[0])*1.5
                self.clipping_dict[map_key]=(l,u)
            else:
                t=self.clipping_dict.get(map_key,None)
                if None==t:
                    logger.warn("Ouch! There is no clipping_dict for {}".format(map_key))
                    exit(0)
                else:
                    l,u=t
            df[colname].clip(lower=l,upper=u,inplace=True)
        return df

    def process_standardization(self,df,process_key,action_key,action_configs,datatype):
        '''This method is for standardizing features using zero mean.
        
        Parameters:
            df(DataFrame) : dataframe to be processed
            process_key(str) : process_key in data config files which are values of process_sequence and keys of process dicts.
            action_key(str) : action_key in data config files which are values of action_sequence and keys of action dicts.
            datatype(str) : 'train' for training data, 'test' for testing data.
            action_configs(dict) : key 'include_columns' indicate a list of columns will be normalized.

        Returns:
            DataFrame: processed dataframe. 

        Examples:

        >>> import pandas as pd
        >>> import numpy as np
        >>> import logging
        >>> from lib.feature_transformer import FeatureTransformer
        >>> logging.disable()
        >>> df_train = pd.DataFrame(np.arange(12).reshape(-1,3),columns=['A', 'B','C'])
        >>> df_train
           A   B   C
        0  0   1   2
        1  3   4   5
        2  6   7   8
        3  9  10  11
        >>> ft=FeatureTransformer()
        >>> action_configs={
        ...    'include_columns':['A','B']
        ... }
        >>> ft.process_standardization(df_train,'AAA','BBB',action_configs,'train')
                  A         B   C
        0 -1.341641 -1.341641   2
        1 -0.447214 -0.447214   5
        2  0.447214  0.447214   8
        3  1.341641  1.341641  11
        >>> ft.norm_dict
        {'AAA_BBB_A': (4.5, 3.3541019662496847), 'AAA_BBB_B': (5.5, 3.3541019662496847)}
        >>> df_test = pd.DataFrame(np.arange(36).reshape(-1,3),columns=['A', 'B', 'C'])
        >>> df_test
             A   B   C
        0    0   1   2
        1    3   4   5
        2    6   7   8
        3    9  10  11
        4   12  13  14
        5   15  16  17
        6   18  19  20
        7   21  22  23
        8   24  25  26
        9   27  28  29
        10  30  31  32
        11  33  34  35
        >>> ft.process_standardization(df_test,'AAA','BBB',action_configs,'test')
                   A         B   C
        0  -1.341641 -1.341641   2
        1  -0.447214 -0.447214   5
        2   0.447214  0.447214   8
        3   1.341641  1.341641  11
        4   2.236068  2.236068  14
        5   3.130495  3.130495  17
        6   4.024922  4.024922  20
        7   4.919350  4.919350  23
        8   5.813777  5.813777  26
        9   6.708204  6.708204  29
        10  7.602631  7.602631  32
        11  8.497058  8.497058  35

        '''

        logger.info("Processing normalization......")
        include_columns=action_configs.get('include_columns')
        cols_exist, cols_not_exist = FeatureTransformer.check_columns_exist(df, include_columns)
        if cols_not_exist:
            logger.warning('{} column(s) do(es) NOT exist:{}'.format(len(cols_not_exist),cols_not_exist))

        for colname in cols_exist:
            map_key='_'.join([process_key,action_key,colname])
            if datatype=='train':
                m=np.mean(df[colname].values)
                s=np.sqrt(np.mean((df[colname].values - m)**2))
                self.norm_dict[map_key]=(m,s)
            else:
                t=self.norm_dict.get(map_key)
                if None==t:
                    logger.warn("Ouch! There is no norm_dict for {}".format(map_key))
                    exit(0)
                else:
                    m,s=t

            if m==0 and s==0:
                logger.info("column {},both mean and std is zero,skipped".format(colname))
            elif s==0:
                logger.info("column {},mean is not zero but stddev is zero,skipped".format(colname))
            else:
                df[colname]=(df[colname].values-m)/s

        return df

    def process_pca(self,df,process_key,action_key,action_configs,datatype):
        """pca for both training data and testing data. 

        Parameters:
            df(DataFrame) : dataframe to be processed
            process_key(str) : process_key in data config files which are values of process_sequence and keys of process dicts.
            action_key(str) : action_key in data config files which are values of action_sequence and keys of action dicts.
            datatype(str) : 'train' for training data, 'test' for testing data.
            action_configs(dict):key 'exclude_columns' indicates a list of columns which are excluded for pca.

        Returns:
            DataFrame: processed dataframe

        Examples:

        >>> import pandas as pd
        >>> import numpy as np
        >>> import logging
        >>> from lib.feature_transformer import FeatureTransformer
        >>> logging.disable()
        >>> df_train = pd.DataFrame(np.arange(24).reshape(-1,4),columns=['A', 'B', 'C', 'D'])
        >>> df_train['E']=df_train['D']
        >>> df_train
            A   B   C   D   E
        0   0   1   2   3   3
        1   4   5   6   7   7
        2   8   9  10  11  11
        3  12  13  14  15  15
        4  16  17  18  19  19
        5  20  21  22  23  23
        >>> ft=FeatureTransformer()
        >>> action_configs={
        ...    'exclude_columns':['A','B'],
        ... }
        >>> ft.process_pca(df_train,'AAA','BBB',action_configs,'train')
               pca_0   A   B
        0  17.320508   0   1
        1  10.392305   4   5
        2   3.464102   8   9
        3  -3.464102  12  13
        4 -10.392305  16  17
        5 -17.320508  20  21
        >>> df_test = pd.DataFrame(np.arange(36).reshape(-1,4),columns=['A', 'B', 'C', 'D'])
        >>> df_test['E']=df_test['D']
        >>> df_test
            A   B   C   D   E
        0   0   1   2   3   3
        1   4   5   6   7   7
        2   8   9  10  11  11
        3  12  13  14  15  15
        4  16  17  18  19  19
        5  20  21  22  23  23
        6  24  25  26  27  27
        7  28  29  30  31  31
        8  32  33  34  35  35
        >>> ft.process_pca(df_test,'AAA','BBB',action_configs,'test')
               pca_0   A   B
        0  17.320508   0   1
        1  10.392305   4   5
        2   3.464102   8   9
        3  -3.464102  12  13
        4 -10.392305  16  17
        5 -17.320508  20  21
        6 -24.248711  24  25
        7 -31.176915  28  29
        8 -38.105118  32  33
        """
        logger.info("Processing pca......")
        exclude_columns=action_configs.get('exclude_columns')
        cols_exist, cols_not_exist = FeatureTransformer.check_columns_exist(df, exclude_columns)
        
        df_tmp=df[cols_exist].copy()
        df=df.drop(columns=cols_exist)
        map_key='{}_{}_{}'.format(process_key,action_key,'pca')
        if datatype=='train':
            pca=PCA(n_components=0.99,svd_solver='full')
            pca.fit(df)
            self.pca_dict[map_key]=pca
        else:
            pca=self.pca_dict.get(map_key)
            if None==pca:
                logger.warn("Ouch! There is no PCA() for {}".format(map_key))
                exit(0)
        
        column_names = ['pca_%i' %i for i in range(pca.n_components_)]

        df=pd.DataFrame(pca.transform(df),columns=column_names)

        if cols_not_exist:
            logger.warning("{} column(s) do(es) NOT exist: {}".format(len(cols_not_exist), cols_not_exist))

        df=pd.concat(objs=[df,df_tmp],axis=1)

        return df

    def process_kbins(self,df,process_key,action_key,action_configs,datatype):
        """KBinsDiscretizer for both training data and testing data. 

        Parameters:
            df(DataFrame) : dataframe to be processed
            process_key(str) : process_key in data config files which are values of process_sequence and keys of process dicts.
            action_key(str) : action_key in data config files which are values of action_sequence and keys of action dicts.
            datatype(str) : 'train' for training data, 'test' for testing data.
            action_configs(dict):key are columns and values are dicts in which key 'n_bins','strategy','encode' is default to 'ordinal' 
            have the same meaning as sklearn.preprocessing.KBinsDiscretizer
        Returns:
            DataFrame: processed dataframe

        Examples:
        
        >>> import pandas as pd
        >>> import numpy as np
        >>> import logging
        >>> from lib.feature_transformer import FeatureTransformer
        >>> logging.disable()
        >>> df_train = pd.DataFrame(np.arange(24).reshape(-1,4),columns=['A', 'B', 'C', 'D'])
        >>> df_train
            A   B   C   D
        0   0   1   2   3
        1   4   5   6   7
        2   8   9  10  11
        3  12  13  14  15
        4  16  17  18  19
        5  20  21  22  23
        >>> ft=FeatureTransformer()
        >>> action_configs={
        ...    'A':{'n_bins':5,'strategy':'uniform'}
        ... }
        >>> ft.process_kbins(df_train,'AAA','BBB',action_configs,'train')
             A   B   C   D
        0  0.0   1   2   3
        1  1.0   5   6   7
        2  2.0   9  10  11
        3  3.0  13  14  15
        4  4.0  17  18  19
        5  4.0  21  22  23
        >>> df_test = pd.DataFrame(np.arange(36).reshape(-1,4),columns=['A', 'B', 'C', 'D'])
        >>> df_test
            A   B   C   D
        0   0   1   2   3
        1   4   5   6   7
        2   8   9  10  11
        3  12  13  14  15
        4  16  17  18  19
        5  20  21  22  23
        6  24  25  26  27
        7  28  29  30  31
        8  32  33  34  35
        >>> ft.process_kbins(df_test,'AAA','BBB',action_configs,'test')
             A   B   C   D
        0  0.0   1   2   3
        1  1.0   5   6   7
        2  2.0   9  10  11
        3  3.0  13  14  15
        4  4.0  17  18  19
        5  4.0  21  22  23
        6  4.0  25  26  27
        7  4.0  29  30  31
        8  4.0  33  34  35
        """
        logger.info("Processing binning......")
        columns = sorted(list(action_configs.keys()))
        cols_exist, cols_not_exist = FeatureTransformer.check_columns_exist(df, columns)

        configs = {k: v for k, v in action_configs.items() if k in cols_exist}

        for k, v in configs.items():
            map_key='{}_{}_{}'.format(process_key,action_key,k)
            logger.info("binning column {} using {}".format(k, v))
            if datatype=='train':
                kbins=KBinsDiscretizer(n_bins=v.get('n_bins',10),encode='ordinal',strategy=v.get('strategy','uniform'))
                df[k]=kbins.fit_transform(df[[k]])
                self.kbins_dict[map_key]=kbins
            else:
                kbins=self.kbins_dict.get(map_key)
                if None==kbins:
                    logger.warn("Ouch! There is no KBinsDiscretizer() for {}".format(map_key))
                    exit(0)
                df[k]=kbins.transform(df[[k]])
        
        if cols_not_exist:
            logger.warning("{} column(s) do(es) NOT exist: {}".format(len(cols_not_exist), cols_not_exist))
        
        return df
