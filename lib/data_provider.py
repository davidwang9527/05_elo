import numpy as np
import pandas as pd
import lib.config as config 
from pandas.core.dtypes.common import is_dict_like
from lib.utility import logger,check_file_exist,save_pickle,load_pickle
from lib.feature_transformer import FeatureTransformer

class DataProvider():
    """Class DataProvider is built for providing transformed data.
    
    This class utilizes elementary functions from feature_transformer and configs
    in data configs(default file name is default_data_configs.py) to prepare data
    for training or testing.

    Parameters:
        data_configs(dict) : the dict loaded from data configs
        cache_prefix(str) : cache prefix for cache files.It replaces {prefix} 
            of data_refresh_configs in config.py
        datatype(str) : 'train' for training data,'test' for testing data.
            It also replaces  {datatype} of data_refresh_configs in config.py
        debug(bool) : 'True' for debug mode

    """

    def __init__(self,data_configs,cache_prefix,datatype,debug):
        self.input_path  = config.file_dir_path.get('input', './input')
        self.cache_path  = config.file_dir_path.get('cache', './cache')
        self.data_refresh_configs = config.data_refresh_configs
        self.data_configs = data_configs
        self.cache_prefix = cache_prefix
        self.datatype = datatype
        self.feature_transformer_filename = '{cache_path}/{prefix}_feature_transformer.pkl'
        self.feature_transformer =FeatureTransformer()
        self.debug  =   debug
        self.data_raw         = {}
        self.data_processed   = {}
        self.data_train_test  = {}
        self.cols_categorical = {}
        self.cols_one_hot     = {}

    def get_debug(self,df):
        '''Get rows from df for debugging

        Parameters:
            df(DataFrame) : the original DataFrame
        
        Returns:
            DataFrame : If debug mode is ON,the final debug rows is the minimum of debug_rows in module config 
              and df rows.If debug mode is OFF, return df.
        '''

        if self.debug:
            debug_num=min(df.shape[0],config.debug_nrows)
            logger.warning('Debug mode, get {} records'.format(debug_num))
            return df.iloc[:debug_num]
        else:
            return df

    def check_syntax(self):
        #check input
        action_configs=self.data_configs.get('input')
        if not action_configs.get('fact_train',{}) or not action_configs.get('fact_test',{}):
            logger.error('Both fact_train and fact_test must be configured for {}!'.format('input')) 
            exit(0)
        are_mappings = [is_dict_like(v) for k,v in action_configs.items()]
        if not all(are_mappings):
            logger.error('items and nested items for input must be dictionaries!')
            exit(0)

        #check process_sequence
        process_sequence=self.data_configs.get('process_sequence',[])
        if not process_sequence:
            logger.error('process_sequence is mandantory!')
            exit(0)
        for process_key in process_sequence:
            process_configs=self.data_configs.get(process_key,{})
            if not process_configs:
                logger.error('There is no configruations for process_key {}!'.format(process_key))
                exit(0)
            
            #check action sequence
            action_sequence=process_configs.get('action_sequence',[])
            if not action_sequence:
                logger.error('action_sequence is mandantory for process {}!'.format(process_key))
                exit(0)
            
            if 'get_data' not in action_sequence or 'result' not in action_sequence:
                 logger.error('get_data and result must be in action_sequence for {}'.format(process_key))
                 exit(0)
            
            possible_actions = ['aggregations','change_dtype','clip_outliers','drop_columns','drop_rows',
                                'factorize_columns','get_data','interaction_columns','kbins','one_hot_encoder',
                                'onehot_encoding','pca','reduce_mem_usage','remove_duplicate','replace_values',
                                'result','select_columns','simple_impute','standardization']
            
            #check actions
            for action_key in action_sequence:
                action_configs=process_configs.get(action_key)
                if action_configs is None:
                    logger.error('No {} configuration for {}'.format(action_key,process_key))
                    exit(0)
                
                ac=[x for x in possible_actions if x in action_key]
                if not ac:
                    logger.error('{} is not supported. Only below actions are supported at the moment:{}'.format(action_key,possible_actions))
                    exit(0)
                
                #check every action
                if 'aggregations' in action_key:
                    action_configs=process_configs.get(action_key,[])
                    for action_config in action_configs:
                        groupby_cols = action_config.get('groupby', [])
                        if not groupby_cols:
                            logger.error("No columns for groupby field")
                            exit(0)
                        metrics_cols = action_config.get('metrics', {})
                        if  not (metrics_cols or action_config.get('count', False) or action_config.get('percent',False)):
                            logger.error("There should be at least one of below three: columns for metrics field , count or percent ")
                            exit(0)

                elif 'clip_outliers' in action_key:
                    action_configs=process_configs.get(action_key,[])
                    if type(action_config)!='list':
                        logger.error("clip_outliers should be a list!")

                elif 'get_data' in action_key:
                    are_mappings = [is_dict_like(v) for v in action_configs]
                    if not all(are_mappings):
                        logger.error('items in {} must be dictionaries!'.format(action_key))
                        exit(0)

                elif 'replace_values' in action_key:
                    action_configs=process_configs.get(action_key,[])
                    are_mappings = [is_dict_like(v) for k,v in action_configs.items()]
                    if not all(are_mappings):
                        logger.error('items in replace_values must be dictionaries!')
                        exit(0)
                
                elif 'interaction_columns' in action_key:
                    action_configs=process_configs.get(action_key,[])
                    possible_interactions = ['add', 'subtract', 'subtract_positive','multiply','divide',
                                'datetime','function']
                    for v in action_configs:
                        interaction = v.get('mode', None)
                        if interaction not in possible_interactions:
                            logger.error("interaction {} is not supported. Only below interactions are supported at the moment:{}".format(v,possible_interactions))
                            exit(0)
        return
    
    def load_data_from_csv(self):
        '''Load data from csv

        There must be 'fact_train' for main table of training data and 'fact_test' for main table of test data.
        When the datatype is 'train','fact_train' will be loaded. Otherwise, fact_test will be loaded. 
        
        csv files are loaded by chunks. The chunk size is determined by variable chunksize in config module.
        
        '''
        process_configs=self.data_configs['input']
        data_dict = {k: '{}/{}'.format(self.input_path, data.get('name', None)) for k, data in process_configs.items()}
        if self.datatype=='test':
            data_dict.pop('fact_train',None)
        else:
            data_dict.pop('fact_test',None)
        
        for k, f_path in data_dict.items():
            if not check_file_exist(f_path):
                logger.error("file {} doesn't exist!".format(f_path))
                continue

            df=pd.DataFrame()
            chunk_no=0
            
            for df_chunk in pd.read_csv(f_path,
                                    chunksize=config.chunksize):
                df_chunk.reset_index(drop=True,inplace=True)
                chunk_no=chunk_no + 1
                logger.info('loading {} chunk(s) from {}, df_chunk shape={}'.format(chunk_no,f_path, df_chunk.shape))
                df_chunk=FeatureTransformer.reduce_mem_usage(df_chunk)

                df=pd.concat([df,df_chunk], ignore_index=True,sort=False)
                logger.info(" {} records has been loaded in total".format(df.shape[0]))
                logger.info('memory usage on {} is {:.3f} MB'.format(k, df.memory_usage(index=True,deep=True).sum() / 1024. ** 2))
                if self.debug==True and df.shape[0]>=config.debug_nrows:
                    break
            
            df=FeatureTransformer.reduce_mem_usage(df)
            if k=='fact_train' or k=='fact_test':
                k='fact'   
            self.data_raw[k]=df
        return

    def load_and_transform_data(self,  source):
        '''This is the method exposed to caller.

            This is a 'controller' method:it invokes 'check_syntax' to check whether there're some errors 
            in data configs.Then it decides where to load the sources(e.g. from csv or from cached files):   
            When refresh_cache is set to False, data is loaded from train_test first(level=3),then data will 
            be loaded from level 2 if there's no files in level 3, and so on.When refresh_cache is set to True, 
            data will be reloaded from csv(level 1) to refresh the cache from scratch.
            Then it invokes 'preprocess' to do feature engineering if there's no files in level 3 or refresh_cache 
            is set to True. 
            Finally it return (x,y).

        Parameters:
            source('str') : source='train' means load and transform training data.source='test' means loading 
                            and transforming testing data.

        Returns:
             tuple: the first item is X, the second item is Y.
        '''
        configs_table = pd.DataFrame(self.data_refresh_configs).T
        configs_table['level'] = configs_table['level'].astype(int)
        configs_table.set_index('level', inplace=True)
        configs_table['filename'] = configs_table['filename'].apply(lambda x: x.format(prefix=self.cache_prefix,datatype=self.datatype) if isinstance(x, str) else None)
        
        self.feature_transformer_filename=self.feature_transformer_filename.format(cache_path=self.cache_path,prefix=self.cache_prefix)

        #check syntax of config files 
        self.check_syntax()

        refresh_level = self.data_refresh_configs.get(source).get('level')
        if refresh_level == 3:
            logger.info("loading data from train_test......")
            filename = '{}/{}'.format(self.cache_path, configs_table.loc[refresh_level, 'filename'])
            self.data_train_test = load_pickle(filename=filename)
            if  FeatureTransformer.any_empty_dataframe(self.data_train_test):
                refresh_level = 2
                logger.warning('No train_test cache for {}  to load. Trying to refresh at level {}'.format(self.datatype,refresh_level))
                self.data_train_test  = {}
            else:
                self.data_train_test['x']=self.get_debug(self.data_train_test['x'])
                self.data_train_test['y']=self.get_debug(self.data_train_test['y'])

        if refresh_level==2:
            logger.info("loading data from raw......")
            filename = '{}/{}'.format(self.cache_path, configs_table.loc[refresh_level, 'filename'])
            self.data_raw = load_pickle(filename=filename)
            if  FeatureTransformer.any_empty_dataframe(self.data_raw):
                refresh_level = 1
                logger.warning('No raw cache for {} to load. Trying to refresh at level {}'.format(self.datatype,refresh_level))
                self.data_raw  = {}
            else:
                for k,v in self.data_raw.items():
                    self.data_raw[k]=self.get_debug(v)
        
        if refresh_level == 1:
            logger.info("loading data from csv......")
            self.load_data_from_csv()
            filename = '{}/{}'.format(self.cache_path, configs_table.loc[2, 'filename'])
            save_pickle(filename, self.data_raw)

        if refresh_level <= 2:
            if self.datatype=='test':
                self.feature_transformer = load_pickle(filename=self.feature_transformer_filename)
                if self.feature_transformer==None:
                    logger.error('No feature transformer for transforming test data!')
                    exit(0)
            self.preprocess()
            filename = '{}/{}'.format(self.cache_path, configs_table.loc[3, 'filename'])
            save_pickle(filename, self.data_train_test)
            if self.datatype=='train':
                save_pickle(self.feature_transformer_filename, self.feature_transformer)

        if self.datatype=='train':
            self.data_train_test['x']=FeatureTransformer.process_drop_columns(self.data_train_test['x'],config.main_table_pk)
        
        return (self.data_train_test['x'], self.data_train_test['y'])

    def preprocess(self):
        '''This method preprocess actions given by data configs.

        It  sequentially processes the value list whose dict key is  'process_sequence'.
        In every dict whose dict key is the items of the above value list,it processes the value list 
        whose dict key is 'action_sequence'.Configurations of actions are documented at module default_data_configs.
        Below are supported actions:aggregations,change_dtype,clip_outliers,drop_columns,drop_rows,factorize_columns,
        get_data,interaction_columns,kbins,one_hot_encoding,pca,remove_duplicate,
        replace_values,result,select_columns,simple_impute,standardization.

        Note that this method supports those action names as the base name. That menas you could use 'replace_values1' and 
        'replace_values2' in the same action_sequence.
        '''

        process_sequence=self.data_configs.get('process_sequence',[])

        for process_key in process_sequence:
            logger.info("processing {}......".format(process_key))
            process_configs=self.data_configs[process_key]
            action_sequence=process_configs.get('action_sequence',[])
            for action_key in action_sequence:
                if 'aggregations' in action_key:
                    action_configs=process_configs.get(action_key,[])
                    if action_configs:
                        df = FeatureTransformer.process_aggregation(df, action_configs)
             
                elif 'change_dtype' in action_key:
                    action_configs=process_configs.get(action_key,[])
                    if action_configs:
                        df = FeatureTransformer.process_change_dtype(df, action_configs=action_configs)
             
                elif 'clip_outliers' in action_key:
                    action_configs=process_configs.get(action_key,[])
                    if action_configs:
                        df = self.feature_transformer.process_clip_outliers(df, process_key,action_key,action_configs,self.datatype)
               
                elif 'drop_columns' in action_key:
                    cols_to_drop=process_configs.get(action_key,[])
                    if cols_to_drop:
                        df = FeatureTransformer.process_drop_columns(df, drop_columns=cols_to_drop)
               
                elif 'drop_rows' in action_key:
                    action_configs=process_configs.get(action_key,{})
                    if action_configs:
                        df = FeatureTransformer.process_drop_rows(df, action_configs=action_configs)
                
                elif 'factorize_columns' in action_key:
                    action_configs=process_configs.get(action_key,[])
                    if action_configs:
                        df = self.feature_transformer.process_factorize(df,process_key,action_key,action_configs,self.datatype)                
                
                elif 'get_data' in action_key:
                    action_configs=process_configs.get(action_key,[])
                    for v in  action_configs:
                        if v.get('dict','')=='raw':
                            df_tmp=self.data_raw[v.get('key')].copy()
                        else:
                            df_tmp=self.data_processed[v.get('key')].copy()
                        
                        how_to=v.get('how_to','')
                        if how_to=='first_table':
                            df=df_tmp.copy()
                        elif how_to=='merge':
                            df = df.merge(df_tmp, how=v.get('how_to_merge'),left_on=v.get('left_on'),right_on=v.get('right_on'))
                            logger.info('After {} merge:{}'.format(v.get('how_to_merge'),df.shape))
                        elif how_to=='append':
                            df = df.append(df_tmp,sort=False,ignore_index=True)
                            logger.info("After append: {}".format(df.shape))
                
                elif 'interaction_columns' in action_key:
                    action_configs=process_configs.get(action_key,[])
                    if action_configs:
                        df = FeatureTransformer.process_interaction(df, action_configs=action_configs)

                elif 'kbins' in action_key:
                    action_configs=process_configs.get(action_key,[])
                    if action_configs:
                        df = self.feature_transformer.process_kbinsprocess_kbins(df,process_key,action_key,action_configs,self.datatype)

                elif 'onehot_encoding' in action_key:
                    action_configs=process_configs.get(action_key,[])
                    df = self.feature_transformer.process_one_hot_encoder(df,process_key,action_key,self.datatype)

                elif 'pca' in action_key:
                    action_configs=process_configs.get(action_key,[])
                    df = self.feature_transformer.process_pca(df,process_key,action_key,action_configs,self.datatype)

                elif 'remove_duplicate' in action_key:
                    action_configs=process_configs.get(action_key,{})
                    logger.info("df shape before removing duplcates:{}".format(df.shape))
                    df=df[~df.duplicated(subset=action_configs['duplicated_index_columns'],keep=action_configs['keep'])]
                    logger.info("df shape after removing duplcates:{}".format(df.shape))

                elif 'replace_values' in action_key:
                    action_configs=process_configs.get(action_key,{})
                    if action_configs:
                        df = FeatureTransformer.process_replace_values(df, action_configs=action_configs)

                elif 'result' in action_key:
                    action_configs=process_configs.get(action_key,[])
                    for f in action_configs:
                        t=f.get('dict','')
                        key=f.get('key','')
                        include_columns=f.get('include_columns','')
                        exclude_columns=f.get('exclude_columns','')
                        if(t=='train_test'):
                            if include_columns:
                                self.data_train_test[key]=FeatureTransformer.process_select_columns(df,include_columns)
                            elif exclude_columns:
                                self.data_train_test[key]=FeatureTransformer.process_drop_columns(df,exclude_columns)
                            else:
                                self.data_train_test[key]=df
                            self.data_train_test[key]=FeatureTransformer.reduce_mem_usage(self.data_train_test[key])
                        else:
                            if include_columns:
                                self.data_processed[key]=FeatureTransformer.process_select_columns(df,include_columns)
                            elif exclude_columns:
                                self.data_processed[key]=FeatureTransformer.process_drop_columns(df,exclude_columns)
                            else:
                                self.data_processed[key]=df
                            self.data_processed[key]=FeatureTransformer.reduce_mem_usage(self.data_processed[key])
                
                elif 'select_columns' in action_key:
                    cols_to_select=process_configs.get(action_key,[])
                    if cols_to_select:
                        df = FeatureTransformer.process_select_columns(df, select_columns=cols_to_select)
                    
                elif 'simple_impute' in action_key:
                    action_configs=process_configs.get(action_key,[])
                    if action_configs:
                        df = self.feature_transformer.process_simple_impute(df, process_key,action_key,action_configs,self.datatype)

                elif 'standardization' in action_key:
                    action_configs=process_configs.get(action_key,[])
                    if action_configs:
                        df = self.feature_transformer.process_standardization(df, process_key,action_key,action_configs,self.datatype)

                else:
                    logger.error('This action {} is not supported!'.format(action_key))
        return
