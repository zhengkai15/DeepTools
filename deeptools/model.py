import torch
from loguru import logger
from .config import set_random_seed
from torch import nn

def initialize_weights(model):
    """
    Initialize weights for all layers in the model.
    Uses appropriate initialization methods based on layer type.

    Args:
        model (torch.nn.Module): The model to initialize.
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Conv1d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose3d)):
            # Kaiming normal initialization for convolutional layers
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d, nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
            if m.bias is not None:
                # Constant initialization for batch normalization layers
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            if m.bias is not None:
                # Kaiming normal initialization for linear layers
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            if m.bias is not None:
                # Normal initialization for embedding layers
                nn.init.normal_(m.weight, mean=0, std=1)
        elif isinstance(m, (nn.LSTM, nn.GRU)):
            if m.bias is not None:
                # Kaiming normal initialization for LSTM/GRU layers
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

def get_model_params(config):
    """
    Get model parameters from the configuration.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        dict: Dictionary containing input and output channel sizes.
    """
    params = {}
    params["in_variable_num"] = config["model"]["in_variable_num"]
    params["in_times_num"]  = config["model"]["in_times_num"]
    params["out_variable_num"]  = config["model"]["out_variable_num"]
    params["out_times_num"]  = config["model"]["out_times_num"]
    in_channels =  params["in_times_num"] * params["in_variable_num"]
    out_channels = params["out_times_num"] * params["out_variable_num"] 
    params["in_channels"] = in_channels
    params["out_channels"] = out_channels
    
    return params

def load_model(config, ModelClass, additional_params=None):
    """
    Load the model with the given configuration and initialize weights if necessary.

    Args:
        config (dict): Configuration dictionary.
        ModelClass (type): The model class to instantiate.
        additional_params (dict, optional): Additional parameters for the model.

    Returns:
        torch.nn.Module: The instantiated and initialized model.
    """
    best_model_path = config["exp"]["best_model_path"]
    params = get_model_params(config)
    
    if additional_params:
        params.update(additional_params)
    model = ModelClass(config=params)
    
    if best_model_path:
        logger.info(f"Loading best model from: {best_model_path}")
        model.load_state_dict(torch.load(best_model_path))
        logger.info(f"模型 {best_model_path} 已加载。")
        
    else:
        model.apply(initialize_weights)
        logger.info(f"模型已初始化。")
    return model

def get_model(config):
    """
    根据配置获取模型。
    设置随机种子，初始化模型，并加载预训练权重（如果可用）。

    参数:
        config (dict): 配置字典。

    返回:
        torch.nn.Module: 实例化并初始化的模型。
    """
    model_name = config["model"]["name"]

    # 模型名称到其相应类的映射
    model_mapping = {
        "unet": "deeptools.models.unet.Model",
    }

    if model_name not in model_mapping:
        raise ValueError(f"不支持的模型名称: {model_name}")

    # 动态导入模型类
    module_name, class_name = model_mapping[model_name].rsplit(".", 1)
    ModelClass = getattr(__import__(module_name, fromlist=[class_name]), class_name)
    
    # 加载模型
    additional_params = None    
    model = load_model(config, ModelClass, additional_params)

    return model