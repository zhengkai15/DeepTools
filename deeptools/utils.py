import os
import sys
from datetime import datetime, timedelta
from loguru import logger


def time_me(func):
    '''
    @summary: cal the time of the fucntion
    @param : None
    @return: return the res of the func
    '''
    def wrapper(*args,**kw):
        start_time = datetime.now()
        res = func(*args, **kw)
        over_time = datetime.now()
        logger.info('current Function: {0} :run time is :{1} seconds'.format(func.__name__ , (over_time - start_time).total_seconds()))
        return res
    return wrapper


def create_logger(log_level="INFO", log_path=None):
    """
    创建日志记录器，支持控制是否写入文件。

    Args:
        write_to_file (bool): 是否将日志写入文件。默认 True。
    
    Returns:
        logger: 配置完成的 loguru 日志记录器。
    """
    # 移除默认配置
    logger.remove()
    
    # 配置控制台日志
    logger.add(sys.stderr, level=log_level, format="{time:YYYY-MM-DD HH:mm:ss} - {level}: {message}")
    
    # 配置文件日志（可选）
    if log_path is not None:
        now = (datetime.now() + timedelta(hours=8)).strftime("%Y%m%d_%H-%M-%S")
        # script_path = os.path.abspath(__file__)
        # script_dir = os.path.dirname(script_path)
        # log_file = f'{script_dir}/log_statistic_{now}.log'
        log_file = os.path.join(log_path, f"logs/log_statistic_{now}.log")        
        # 创建日志文件目录
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        # 添加日志文件记录器
        logger.add(log_file, rotation="10 MB", level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} - {level}: {message}")
    return logger


def get_color_codes():
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_RESET = '\033[49m'  # 重置背景颜色
    return BG_RED, BG_GREEN, BG_YELLOW, BG_BLUE, BG_RESET


def print_variable_info(variable_dict):
    BG_RED, BG_GREEN, BG_YELLOW, BG_BLUE, BG_RESET = get_color_codes()
    
    logger.info(f"{10*'%'} echo {10*'%'}")
    for var_name, var in variable_dict.items():
        device = str(var.device) if hasattr(var, 'device') else 'CPU'
        if hasattr(var, 'shape'):
            contents = str(var.shape)
            logger.info(f"{BG_GREEN} Variable Name: {var_name:<15}, Shape    : {contents:<50}, Device: {device:<15} {BG_RESET}")
        else:
            contents = str(var)
            logger.info(f"{BG_GREEN} Variable Name: {var_name:<15}, Conetents: {contents:<50}, Device: {device:<15} {BG_RESET}")
            