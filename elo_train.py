import pickle
import argparse
import sys
import numpy as np
from lib.config import file_dir_path,debug_nrows,filename_model_result,main_table_pk
from lib.utility import logger,initialize_configs,load_pickle
from lib.data_provider import DataProvider
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

def parse_command_line():
    default_cache_prefix    = 'default'
    default_data_configs    = 'default_data_configs.py'
    default_model_configs   = 'default_model_configs.py'

    arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arg_parser.add_argument('-p', '--cache_prefix',    type=str, default=default_cache_prefix, help='cache file prefix')
    arg_parser.add_argument('-d', '--data_configs',    type=str, default=default_data_configs,  help='data provider configs')
    arg_parser.add_argument('-m', '--model_configs',   type=str, default=default_model_configs, help='model training configs')
    arg_parser.add_argument('--refresh_cache', action='store_true', default=False,  help='refresh cache from orignal data')
    arg_parser.add_argument('--debug', action='store_true', default=False, help='debug mode using {} samples'.format(debug_nrows))
    args = arg_parser.parse_args()

    configs_loc = file_dir_path.get('configs', './configs')
    args.data_configs='{}/{}'.format(configs_loc,args.data_configs)
    args.model_configs='{}/{}'.format(configs_loc,args.model_configs)

    logger.info('-' * 30)
    logger.info('running task with prefix={}'.format(args.cache_prefix))
    logger.info('running task with data configs={}'.format(args.data_configs))
    logger.info('running task with model configs={}'.format(args.model_configs))
    logger.info('Refreshing cache from original data?:{}'.format(args.refresh_cache))
    if args.debug:
        logger.warning('**Debug Mode**')
    return args

def load_and_transform_data(args,datatype):
    data_configs    = initialize_configs(args.data_configs).data_configs
    dp = DataProvider(data_configs=data_configs,cache_prefix=args.cache_prefix,datatype=datatype,debug=args.debug)

    if args.refresh_cache:
        x, y = dp.load_and_transform_data(source='from_csv')
    else:
        x, y = dp.load_and_transform_data(source='from_train_test') 
    
    if args.debug:
        logger.warning('debug mode,x={}'.format(x.shape))
    else:
        logger.info('normal mode,x={}'.format(x.shape))
    return (x,y)

def train(args):
    #loading training data
    X, y=load_and_transform_data(args,'train')
    train_x, test_x, train_y, test_y = train_test_split(X, y,test_size=0.2, random_state=40)

    #training model
    model_configs = initialize_configs(args.model_configs).model_configs
    logger.info('load hpo configs of {} models'.format(len(model_configs)))
    rmse_best = 100
    for model_name,model_config in model_configs.items():
        logger.info('training model {}'.format(model_name))
        regression=model_config.get('model',None)(model_configs.get('initialize'))
        param_grid=model_config.get('search_space')
        search = GridSearchCV(estimator=regression,param_grid=param_grid,cv=5,scoring='neg_root_mean_squared_error',refit=True,n_jobs=-1,verbose=True)
        search.fit(train_x,train_y)
        logger.info('best hyperparameters for {}:{}'.format(model_name,search.best_params_))
        
        score_train=search.score(train_x,train_y)
        score_test=search.score(test_x,test_y)
        logger.info('score:{:.2f}/{:.2f}'.format(score_train,score_test))
        
        pred_train=search.predict(train_x)
        pred_test=search.predict(test_x)
        rmse_train=np.sqrt(metrics.mean_squared_error(pred_train,train_y))
        rmse_test=np.sqrt(metrics.mean_squared_error(pred_test,test_y))
        logger.info('RMSE:{:.2f}/{:.2f}'.format(rmse_train,rmse_test))

        if rmse_test < rmse_best:
            logger.info('New test rmse benchmark! rmse_test:{},rmse_best:{}'.format(rmse_test,rmse_best))
            rmse_best = rmse_test
            model_filename="./model/best_model.pkl"
            with open(model_filename, "wb") as f:
                pickle.dump(search, f)
                logger.info("Model has been written to " + model_filename)
    return

if __name__ == '__main__':
    try:
        #Two ways to change cache_prefix to gs: 1).sys.argv.append('-pgs') ; 2).sys.argv.append('--cache_prefix=gs')
        sys.argv.append('-pelo')
        #sys.argv.append('--debug')

        args=parse_command_line()

        train(args)
    except BaseException as e:
        logger.exception(e)
