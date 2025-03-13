import os
import random
import numpy as np
import pandas as pd
from loguru import logger
from torch.utils.data import Dataset, DataLoader, Subset

# 定义闭包 worker_init_fn
def create_worker_init_fn(seed):
    def worker_init_fn(worker_id):
        random.seed(seed + worker_id)
        np.random.seed(seed + worker_id)
    return worker_init_fn

    
def get_dataloader(config, mydataset=None):
    worker_init_fn = create_worker_init_fn(config["seed"]["random_seed"])
    my_data_train = mydataset(flag='train',config=config)
    logger.info(f"my_data_train.__len__():{my_data_train.__len__()}")
    logger.info(f'config["train"]["iter_num"]*config["train"]["batch_size"]:{config["train"]["iter_num"]*config["train"]["batch_size"]}')
    train_subset_indices =  np.random.choice(my_data_train.__len__(), min(config["train"]["iter_num"]*config["train"]["batch_size"], my_data_train.__len__()), replace=False)
    my_data_train_subsets = Subset(my_data_train, train_subset_indices)
    train_loader = DataLoader(my_data_train_subsets, batch_size=config["train"]["batch_size"], num_workers=config["train"]["num_workers"], shuffle=config["train"]["shuffle"], worker_init_fn=worker_init_fn)

    my_data_train_eval = mydataset(flag='train',config=config) 
    logger.info(f"my_data_train_eval.__len__():{my_data_train_eval.__len__()}")
    logger.info(f'config["infer"]["iter_num"]*config["infer"]["batch_size"]:{config["infer"]["iter_num"]*config["infer"]["batch_size"]}')
    train_eval_subset_indices =  np.random.choice(my_data_train_eval.__len__(), min(config["infer"]["iter_num"]*config["infer"]["batch_size"], my_data_train_eval.__len__()), replace=False)
    my_data_train_eval_subsets = Subset(my_data_train_eval, train_eval_subset_indices)
    train_loader_eval = DataLoader(my_data_train_eval_subsets, batch_size=config["infer"]["batch_size"], num_workers=config["infer"]["num_workers"], shuffle=config["infer"]["shuffle"], worker_init_fn=worker_init_fn)

    my_data_valid = mydataset(flag='valid',config=config)
    logger.info(f"my_data_valid.__len__():{my_data_valid.__len__()}")
    logger.info(f'config["infer"]["iter_num"]*config["infer"]["batch_size"]:{config["infer"]["iter_num"]*config["infer"]["batch_size"]}')
    valid_subset_indices =  np.random.choice(my_data_valid.__len__(), min(config["infer"]["iter_num"]*config["infer"]["batch_size"], my_data_valid.__len__()), replace=False)
    my_data_valid_subsets = Subset(my_data_valid, valid_subset_indices)
    valid_loader = DataLoader(my_data_valid_subsets, batch_size=config["infer"]["batch_size"], num_workers=config["infer"]["num_workers"], shuffle=config["infer"]["shuffle"], worker_init_fn=worker_init_fn)

    my_data_test = mydataset(flag='test',config=config)
    logger.info(f"my_data_test.__len__():{my_data_test.__len__()}")
    logger.info(f'config["infer"]["iter_num"]*config["infer"]["batch_size"]:{config["infer"]["iter_num"]*config["infer"]["batch_size"]}')
    test_subset_indices =  np.random.choice(my_data_test.__len__(), min(config["infer"]["iter_num"]*config["infer"]["batch_size"], my_data_test.__len__()), replace=False)
    my_data_test_subsets = Subset(my_data_test, test_subset_indices)
    test_loader = DataLoader(my_data_test_subsets, batch_size=config["infer"]["batch_size"], num_workers=config["infer"]["num_workers"], shuffle=config["infer"]["shuffle"], worker_init_fn=worker_init_fn)

    return train_loader, train_loader_eval, valid_loader, test_loader