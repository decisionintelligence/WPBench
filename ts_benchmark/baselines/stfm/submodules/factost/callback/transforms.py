import torch
import torch.nn as nn
from .core import Callback
from src.models.layers.revin import RevIN

class RevInCB(Callback):
    def __init__(self, num_features: int, eps=1e-5, affine:bool=False, denorm:bool=True, device=None):
        """
        Callback for Reversible Instance Normalization (RevIN)
        
        RevIN is specifically designed for time series data, allowing for both normalization
        during forward pass and denormalization during inference.
        
        Args:
            num_features (int): Number of input features/channels
            eps (float): Small constant for numerical stability
            affine (bool): If True, adds learnable affine transformation parameters.
                         Note: Currently only works with affine=False
            denorm (bool): If True, performs denormalization after forward pass
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.denorm = denorm
        self.device = device
        self.revin = RevIN(num_features, eps, affine, device=device)
        
        # Initialize parameters if affine is enabled
        if self.affine:
            self.revin._init_params()
    
    def before_fit(self):
        """Add RevIN parameters to optimizer if affine is enabled"""
        if self.affine:
            # 检查参数是否已在 optimizer 中
            existing_params = set()
            for group in self.learner.opt.param_groups:
                for p in group['params']:
                    existing_params.add(p)
            params_to_add = []
            for p in [self.revin.affine_weight, self.revin.affine_bias]:
                if p not in existing_params:
                    params_to_add.append(p)
            if params_to_add:
                self.learner.opt.add_param_group({
                    'params': params_to_add,
                    'lr': self.learner.lr
                })
    
    def before_forward(self): 
        self.revin_norm()
        
    def after_forward(self): 
        if self.denorm: 
            self.revin_denorm() 
        
    def revin_norm(self):
        """
        Normalize input data before forward pass
        Transforms input shape: [batch_size x sequence_length x num_features]
        Uses instance normalization along the sequence dimension
        """
        xb_revin = self.revin(self.xb, 'norm')  
        self.learner.xb = xb_revin

    def revin_denorm(self):
        """
        Denormalize predictions after forward pass
        Transforms predictions shape: [batch_size x target_window x num_features]
        Restores the original scale of the data
        """
        # Handle case where pred is a dictionary (multiple horizons)
        if isinstance(self.pred, dict):
            # For zero-shot evaluation, we typically use the first horizon
            first_horizon_key = list(self.pred.keys())[0]
            pred_tensor = self.pred[first_horizon_key]
            denorm_pred = self.revin(pred_tensor, 'denorm')
            self.learner.pred = denorm_pred
        else:
            # Handle case where pred is already a tensor
            pred = self.revin(self.pred, 'denorm')      
            self.learner.pred = pred
    

