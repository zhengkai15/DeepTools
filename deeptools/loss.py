import torch
import torch.nn as nn
from loguru import logger

class mse_loss(nn.Module):
    def __init__(self, config):
        super(mse_loss, self).__init__()
    
    def forward(self, outputs, targets):
        assert outputs.shape == targets.shape, "Prediction and label must have the same shape"
        loss = nn.functional.mse_loss(outputs, targets, reduction='mean')
        return loss
    
def get_loss(config):
    """Get loss function based on loss name.
    
    Args:
        loss_name (str): Name of the loss function.
        
    Returns:
        loss_func: The loss function object.
    """
    loss_name = config["loss"]["name"]
    logger.info(f"loss_name: {loss_name}")
    
    if loss_name == 'mse':
        loss_func = mse_loss(config)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
        
    return loss_func