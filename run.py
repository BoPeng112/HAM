from interactions import Interactions
from eval_metrics import *
from train import *

import argparse
import logging
from time import time
import datetime
import torch
import pdb
import pickle

if __name__ == '__main__':

    #FIXME single config function 
    parser = argparse.ArgumentParser()

    # data arguments
    parser.add_argument('--L', type=int, default=5)
    parser.add_argument('--T', type=int, default=3)
    parser.add_argument('--P', type=int, default=1)
    parser.add_argument('--data', type=str, default='CDs')

    # train arguments
    parser.add_argument('--n_iter', type=int, default=300)
    parser.add_argument('--setting', type=str, default='CUT')
    parser.add_argument('--isTrain', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--l2', type=float, default=1e-3)
    parser.add_argument('--neg_samples', type=int, default=3)
    parser.add_argument('--order', type=int, default=2)
    parser.add_argument('--sets_of_neg_samples', type=int, default=50)
    parser.add_argument('--abla', type=str, default='no')

    # model dependent arguments
    parser.add_argument('--model', type=str, default='xHAM')
    parser.add_argument('--d', type=int, default=50)

    config = parser.parse_args()

    #the code below is used to specify the directories to store the results
    #resultsName = 'all_results'
    #logName = resultsName+'/'+config.model+'/'+config.setting+'/'+config.data+'/'+config.data+'_'+str(config.d)+'_'+str(config.L)+'_'+str(config.T)+'_'+str(config.P)+'_'+str(config.l2)+'_'+str(config.order)+'_'+config.abla+'.'+config.setting

    ##logging.basicConfig(filename=logName, level=logging.DEBUG)
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    #FIXME input file
    if config.data == 'CDs':
        from data import Amazon
        data_set = Amazon.CDs()
    elif config.data == 'Books':
        from data import Amazon
        data_set = Amazon.Books()
    elif config.data == 'Children':
        from data import GoodReads
        data_set = GoodReads.Children()
    elif config.data == 'Comics':
        from data import GoodReads
        data_set = GoodReads.Comics()
    elif config.data == 'ML20M':
        from data import MovieLens
        data_set = MovieLens.ML20M()
    elif config.data == 'ML1M':
        from data import MovieLens
        data_set = MovieLens.ML1M()

    #generate datasets in the 80-20-CUT setting
    train_set, val_set, train_val_set, test_set, num_users, num_items = data_set.generate_dataset(index_shift=1)

    #generate datasets in the 3-LOS setting
    if config.setting == 'LOS':
        assert len(train_set) == len(val_set) and len(test_set) == len(train_set)

        for i in range(len(train_set)):
            user = train_set[i] + val_set[i] + test_set[i]
            train_set[i]     = user[:-6]
            train_val_set[i] = user[:-3]
            val_set[i]       = user[-6:-3]
            test_set[i]      = user[-3:]

    #generate datasets in the 3-80-CUT setting
    if config.setting == 'CUS':
        test_set = [eachlist[:3] for eachlist in test_set]

    #training or testing mode
    if config.isTrain:
        train = Interactions(train_set, num_users, num_items)
    else:
        train = Interactions(train_val_set, num_users, num_items)
    
    train.to_sequence(config.L, config.T)

    logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info(config)

    #FIXME comments
    if config.isTrain:
        train_model(train, val_set, config, logger)
    else:
        train_model(train, test_set, config, logger)





