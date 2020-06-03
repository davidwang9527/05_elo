import argparse
import sys
from lib.config import file_dir_path,debug_nrows,filename_model_result,main_table_pk
from lib.utility import logger,initialize_configs,load_pickle
from lib.data_provider import DataProvider

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
    arg_parser.add_argument('--training',   action='store_true', default=False, help='training model')
    arg_parser.add_argument('--predict',   action='store_true', default=False, help='predicting')
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

#This functions usually needs to modified for every project
def predict(args):
    #loading test data
    test, _=load_and_transform_data(args,'test')

    #loading model
    model=load_pickle('./model/best_model.pkl')
    submission_filename='{}/{}'.format(file_dir_path.get('output','./output'),'elo.csv')
    
    #predict
    preds = model.predict(test.drop(labels=main_table_pk,axis=1))        
    
    test["target"] = preds
    submission = test[['card_id','target']]
    submission.to_csv(submission_filename, index=False)
    logger.info(submission.head())

if __name__ == '__main__':
    try:
        #Two ways to change cache_prefix to gs: 1).sys.argv.append('-pgs') ; 2).sys.argv.append('--cache_prefix=gs')
        sys.argv.append('-pelo')

        args=parse_command_line()
        predict(args)
    except BaseException as e:
        logger.exception(e)
