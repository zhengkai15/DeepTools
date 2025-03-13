import os
import torch
import torch.optim as optim
from .config import save_yaml
from loguru import logger
from .utils import time_me
from .utils import print_variable_info
    
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler


# Setup optimizer and learning rate scheduler
@time_me
def get_optim(config, model):
    if config["optim"]["name"] == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config["lr"]["value"],
            weight_decay=config["optim"]["weight_decay"]
        )
    return optimizer


@time_me
def get_scheduler(config, optimizer):
    scheduler_name = config["lr"]["scheduler"]["name"]
    scheduler_params = config["lr"]["scheduler"]

    if scheduler_name == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_params["mode"],
            factor=scheduler_params["factor"],
            patience=scheduler_params["patience"],
            threshold=scheduler_params["threshold"],
            min_lr=scheduler_params["min_lr"],
            verbose=True
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_name}")
    return scheduler


import signal
class EarlyStopping:
    def __init__(self, patience=5, mode='max'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.stopped_epoch = 0
        self.early_stop = False
        self.mode = mode
        self.best_model_state = None  # Store best model state
        self.config = None
        self.model = None

        # Register signal handler for Ctrl+C
        signal.signal(signal.SIGINT, self.handle_signal)

    def save_checkpoint(self):
        """Save the best model and update config."""
        if self.config is None or self.model is None:
            logger.error("Config or model is not set; cannot save checkpoint.")
            return

        model_dir = os.path.join(self.config["exp"]["dir"], "model")
        os.makedirs(model_dir, exist_ok=True)
        best_model_path = os.path.join(model_dir, "best_model.pth")
        torch.save(self.best_model_state, best_model_path)
        self.config['exp']['best_model_path'] = best_model_path
        save_yaml(self.config)
        logger.info(f"Best model saved to {best_model_path}")

    def handle_signal(self, signum, frame):
        """Handle Ctrl+C signal to save the best model."""
        logger.info(f"Signal {signum} received. Saving the best model before exiting...")
        self.save_checkpoint()
        exit(0)

    def __call__(self, val_loss, epoch, config, model):
        """Check for early stopping conditions."""
        self.config = config  # Set config
        self.model = model  # Set model

        if self.mode == "max":
            improve_condition = self.best_score is None or val_loss >= self.best_score
        elif self.mode == "min":
            improve_condition = self.best_score is None or val_loss <= self.best_score
        else:
            logger.error(f"Invalid mode: {self.mode}")
            return

        if improve_condition:
            self.best_score = val_loss
            self.counter = 0
            self.best_model_state = model.state_dict()  # Update best model state
            self.save_checkpoint()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped_epoch = epoch
                self.early_stop = True
                logger.info(f"Early stopping triggered at epoch {epoch}.")

        logger.info(f"Early stopping counter: {self.counter}, Best score: {self.best_score:.4f}")

  
def get_early_stopping(config):
    early_stopping = EarlyStopping(patience=config["train"]["early_stopping"]["patience"], mode=config["train"]["early_stopping"]["mode"])
    return early_stopping


from .utils import get_color_codes
def check_nan_inf(x):
    BG_RED, BG_GREEN, BG_YELLOW, BG_BLUE, BG_RESET = get_color_codes()
    if torch.isnan(x).any() or torch.isinf(x).any():
        logger.info(f"{BG_RED} *********** NaN or Inf detected in input tensor. {BG_RESET}")



import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch.nn.functional as F

def update_tqdm(total_num, index, pbar, last_update_time=None, update_type='by_time'):
    if update_type == 'by_time':
        # 每30s更新一次进度条
        current_time = time.time()
        if current_time - last_update_time >= 30 or (index + 1) == total_num:
            # 计算已完成的步数，并更新进度条
            pbar.update(index + 1 - pbar.n)  # 更新到当前步
            last_update_time = current_time  # 更新时间戳
    else:
        # 每 10 次更新一次进度条
        if (index + 1) % 10 == 0 or (index + 1) == total_num:
            pbar.update(10 if (index + 1) % 10 == 0 else total_num % 10)
    return last_update_time

## save each model
def save_model(model, exp, epoch):
    os.makedirs(f'{exp}/model/', exist_ok=True)
    torch.save(model.state_dict(), f'{exp}/model/model_{epoch:03d}.pth')     
    
def save_train_metrics(metric_epoch,exp, optimizer, lrs, losses):
    metric = np.mean(metric_epoch)
    losses.append(metric)
    lr_cur = optimizer.param_groups[0]['lr']
    lrs.append(lr_cur)
    df = pd.DataFrame({
    'lr': lrs,    # 第一列
    'loss': losses      # 第二列
    })
    df.to_csv(os.path.join(exp, f"loss_train_all.csv"))
    
def save_eval_metrics(lrs, losses_valid, task_metrics, exp, flag):
    df = pd.DataFrame({'lr': lrs,'loss': losses_valid, 'task_metrics': task_metrics})
    df.to_csv(f'{exp}/loss_{flag}.csv')


@time_me
def train_epoches(model, 
    train_loader, 
    sample_num,
    loss_func, 
    optimizer, 
    epoch, 
    losses, 
    lrs, 
    config, funcs=None):
    ft_gt_process, train_result_process = funcs
    loss_epoch = []
    model.cuda()
    model.train()
    # 获取损失阈值
    loss_threshold = config['train']['loss_threshold']
    # with tqdm(total=train_loader.__len__(), desc=f'trainning epoch:{epoch}') as pbar:
    with tqdm(total=sample_num, desc=f'trainning epoch:{epoch}') as pbar:
        last_update_time = time.time()  # 记录上次更新时间
        for index, (ft_item, gt_item, _) in enumerate(train_loader):
            ft_item,gt_item = ft_item.cuda().float(), gt_item.cuda().float()
            # input gt process 
            ft_item, gt_item = ft_gt_process(ft_item, gt_item, config)
            
            
            baseline, output_item = model(ft_item)
            
            # pred process
            output_item = train_result_process(output_item)
            
            if config["debug"]["verbose"]:
                print_variable_info({"ft_item":ft_item, "gt_item":gt_item, "output_item":output_item})
            
            # calculate loss
            loss = loss_func(output_item, gt_item)
            
            loss_epoch.append(loss.item())
            optimizer.zero_grad()
            # 检查损失值是否超过阈值
            if loss.item() <= loss_threshold:
                loss.backward()
                # 检查是否启用梯度裁剪
                if config['train']['gradient_clipping']['enabled']:
                    method = config['train']['gradient_clipping']['method']
                    if method == 'value':
                        # 按值裁剪
                        min_value = config['train']['gradient_clipping']['clip_value']['min']
                        max_value = config['train']['gradient_clipping']['clip_value']['max']
                        for p in model.parameters():
                            p.grad.data.clamp_(min=min_value, max=max_value)
                    elif method == 'norm':
                        # 按范数裁剪
                        max_norm = config['train']['gradient_clipping']['max_norm']
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                optimizer.step()
            else:
                logger.info(f"损失值 {loss.item()} 超过阈值 {loss_threshold}，跳过此次反向传播和参数更新。")
            if index >= sample_num:
                break
            
            # update qtdm
            last_update_time = update_tqdm(total_num=len(train_loader), index=index, pbar=pbar, last_update_time=last_update_time, update_type='by_time')

            if config["debug"]["verbose"] and index == 9:
                break
    if config["train"]["save_epoch_model"]:
        save_model(model, config["exp"]["dir"], epoch)
    save_train_metrics(loss_epoch, config["exp"]["dir"], optimizer, lrs, losses)
    
    return losses, lrs
        

@time_me
def eval_epoches(
    model,
    eval_loader,
    sample_num=0,
    flag='test',
    epoch=0,
    loss_func=None,
    losses_valid=[],
    scheduler=None,
    lrs=[],
    task_metrics=[],
    config=None, res_epoch=None, funcs=None):
    
    ft_gt_process, infer_result_process, cal_target_metric, print_func = funcs
    model.eval()
    
    loss_epoch = []
    
    with tqdm(total=sample_num, desc=f'evaluate {flag} epoch:{epoch}') as pbar:
        last_update_time = time.time()  # 记录上次更新时间
        for index, (ft_item, gt_item, timestamp) in enumerate(eval_loader):
            
            ft_item, gt_item = ft_item.cuda().float(), gt_item.cuda().float()
            ft_item, gt_item = ft_gt_process(ft_item, gt_item, config)

            baseline, output_item = model(ft_item)
            
            if config["debug"]["verbose"]:
                print_variable_info({"ft_item":ft_item, "gt_item":gt_item, "output_item":output_item})
                
            # ************** 测评开始 ***************
            for b_i in range(config["infer"]["batch_size"]):
                output_item_i, gt_item_i, baseline_i = output_item[b_i:b_i+1], gt_item[b_i:b_i+1], baseline[b_i:b_i+1]
                data, loss_epoch = infer_result_process(output_item_i, gt_item_i, baseline_i, config, loss_func, loss_epoch)
                res_epoch = cal_target_metric(res_epoch, data)
            # ************** 测评结束 ***************

            if index >= sample_num:
                break
            
            # 更新进度条
            last_update_time = update_tqdm(total_num=len(eval_loader), index=index, pbar=pbar, last_update_time=last_update_time, update_type='by_time')

            if config["debug"]["verbose"]  and index == 9:
                break     
        
        # 更新当前epoch的指标, res_epoch的最后一个位置
        task_metric_ls = res_epoch[-1]
        metric = np.nanmean(loss_epoch)
        losses_valid.append(metric)
        task_metric = np.nanmean(task_metric_ls)
        task_metrics.append(task_metric)
        
        print_func(res_epoch)
        
        # save_eval_metrics
        save_eval_metrics(lrs, losses_valid, task_metrics, config["exp"]["dir"], flag)
        
        # print info
        BG_RED, BG_GREEN, BG_YELLOW, BG_BLUE, BG_RESET = get_color_codes()
        if flag == "valid":
            logger.info(f'{BG_RED} losses_{flag}: {losses_valid} {BG_RESET} ')
            logger.info(f'{BG_RED} task_metrics_{flag}: {task_metrics} {BG_RESET} ')
        
        # update scheduler
        if config["lr"]["scheduler"]["monitor"] == 'task_metric':
            monitor_metric = task_metric
        elif config["lr"]["scheduler"]["monitor"] == 'loss':
            monitor_metric = metric
        else:
            logger.info(f"wrong {config['lr']['scheduler']['monitor']}")
        
        if scheduler is not None:
            scheduler.step(monitor_metric)
        # return
        return losses_valid, task_metrics
    
    
    
import matplotlib.pyplot as plt
import pandas as pd
@time_me
def plot_metrics(exp_path):
    """
    Plot training/validation/test metrics from CSV files
    
    Args:
        exp_path: Experiment path containing the CSV files
        flag: Current mode (train/valid/test) 
    """
    # Read train/valid/test results
    train_df = pd.read_csv(f'{exp_path}/loss_train.csv', index_col=0)
    valid_df = pd.read_csv(f'{exp_path}/loss_valid.csv', index_col=0)
    test_df = pd.read_csv(f'{exp_path}/loss_test.csv', index_col=0)

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot loss curves
    ax1.plot(train_df.index, train_df['loss'], label='Train')
    ax1.plot(valid_df.index, valid_df['loss'], label='Valid') 
    ax1.plot(test_df.index, test_df['loss'], label='Test')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot task_metrics curves
    ax2.plot(train_df.index, train_df['task_metrics'], label='Train')
    ax2.plot(valid_df.index, valid_df['task_metrics'], label='Valid')
    ax2.plot(test_df.index, test_df['task_metrics'], label='Test')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('task_metrics')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f'{exp_path}/metrics.png')
    plt.close()